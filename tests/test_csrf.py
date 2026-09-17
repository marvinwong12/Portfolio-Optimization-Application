"""CSRF protection is disabled in the shared `app` fixture (see
conftest.py) so the rest of the suite can POST without scraping a token on
every request - those tests exercise route/ownership/business logic, not
CSRF plumbing. This module explicitly turns it back on (the real default)
to verify the protection actually works end-to-end, rather than just
trusting that CSRFProtect is wired up correctly.
"""
import re

import pytest

from portfolio_optimizer import create_app
from portfolio_optimizer.models import db, User


def _extract_csrf_token(html_bytes):
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html_bytes)
    assert match, "Expected a csrf_token hidden input on the rendered page"
    return match.group(1).decode()


@pytest.fixture
def csrf_app():
    """Same shape as the shared `app` fixture, but with CSRF protection
    actually turned on, to test it for real."""
    application = create_app({
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'TESTING': True,
        'WTF_CSRF_ENABLED': True,
    })
    yield application


@pytest.fixture
def csrf_client(csrf_app):
    return csrf_app.test_client()


def _register(csrf_client, username='carol', email='carol@example.com'):
    """Register a user through the real form flow (token scraped from the
    rendered page), returning the client already logged in."""
    get_response = csrf_client.get('/register')
    token = _extract_csrf_token(get_response.data)
    return csrf_client.post('/register', data={
        'username': username,
        'email': email,
        'password': 'password123',
        'confirm_password': 'password123',
        'csrf_token': token,
    })


class TestLoginCSRF:
    def test_post_without_csrf_token_is_rejected(self, csrf_app, csrf_client):
        with csrf_app.app_context():
            user = User(username='alice', email='alice@example.com')
            user.set_password('correcthorse')
            db.session.add(user)
            db.session.commit()

        response = csrf_client.post('/login', data={'username': 'alice', 'password': 'correcthorse'})
        assert response.status_code == 400

    def test_post_with_valid_csrf_token_succeeds(self, csrf_app, csrf_client):
        with csrf_app.app_context():
            user = User(username='alice', email='alice@example.com')
            user.set_password('correcthorse')
            db.session.add(user)
            db.session.commit()

        get_response = csrf_client.get('/login')
        token = _extract_csrf_token(get_response.data)

        response = csrf_client.post('/login', data={
            'username': 'alice',
            'password': 'correcthorse',
            'csrf_token': token,
        })
        assert response.status_code == 302

    def test_post_with_garbage_csrf_token_is_rejected(self, csrf_app, csrf_client):
        with csrf_app.app_context():
            user = User(username='alice', email='alice@example.com')
            user.set_password('correcthorse')
            db.session.add(user)
            db.session.commit()

        response = csrf_client.post('/login', data={
            'username': 'alice',
            'password': 'correcthorse',
            'csrf_token': 'not-a-real-token',
        })
        assert response.status_code == 400


class TestRegisterCSRF:
    def test_post_without_csrf_token_is_rejected(self, csrf_client):
        response = csrf_client.post('/register', data={
            'username': 'bob',
            'email': 'bob@example.com',
            'password': 'password123',
            'confirm_password': 'password123',
        })
        assert response.status_code == 400

    def test_post_with_valid_csrf_token_succeeds(self, csrf_client):
        response = _register(csrf_client, username='bob', email='bob@example.com')
        assert response.status_code == 302


class TestPortfolioRoutesCSRF:
    def test_create_portfolio_without_csrf_token_is_rejected_even_when_logged_in(self, csrf_client):
        _register(csrf_client)  # logs in as 'carol'

        response = csrf_client.post('/', data={'name': 'Bad', 'stocks': 'AAPL'})
        assert response.status_code == 400

    def test_delete_without_csrf_token_is_rejected(self, csrf_app, csrf_client):
        _register(csrf_client)  # logs in as 'carol'

        response = csrf_client.post('/delete/1')
        assert response.status_code == 400

    def test_delete_get_is_rejected_regardless_of_csrf(self, csrf_client):
        """Even with a would-be-valid session, /delete must not accept GET
        at all - CSRFProtect only guards state-changing HTTP methods, so a
        GET-based delete would stay forgeable no matter what."""
        _register(csrf_client)  # logs in as 'carol'

        response = csrf_client.get('/delete/1')
        assert response.status_code == 405
