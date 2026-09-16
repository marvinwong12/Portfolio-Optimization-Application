import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer import create_app
from portfolio_optimizer.models import db, User
from portfolio_optimizer.cache import price_cache, risk_free_rate_cache


@pytest.fixture(autouse=True)
def _clear_caches():
    """Prevent cache state from one test leaking into the next."""
    price_cache.clear()
    risk_free_rate_cache.clear()
    yield
    price_cache.clear()
    risk_free_rate_cache.clear()


@pytest.fixture
def app():
    """A Flask app instance backed by an in-memory database, isolated from
    the real portfolios.db."""
    application = create_app({
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'TESTING': True,
    })
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def create_user(app):
    """Factory fixture for creating a User directly in the database
    (bypassing the /register route) with a known plaintext password, so
    tests can log in as them. Returns the new user's id."""
    def _create_user(username='testuser', email='testuser@example.com', password='password123'):
        with app.app_context():
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            return user.id
    return _create_user


@pytest.fixture
def logged_in_client(app, client, create_user):
    """A test client already logged in as a fresh user. Yields (client, user_id)."""
    user_id = create_user()
    response = client.post('/login', data={'username': 'testuser', 'password': 'password123'})
    assert response.status_code == 302  # sanity check the login actually worked
    return client, user_id


@pytest.fixture
def synthetic_returns():
    """Deterministic daily returns for 4 assets over ~300 trading days,
    used across analyzer tests so results are reproducible."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range('2023-01-02', periods=300)
    data = rng.normal(loc=0.0006, scale=0.012, size=(300, 4))
    return pd.DataFrame(data, index=dates, columns=['AAA', 'BBB', 'CCC', 'DDD'])
