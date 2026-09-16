import json

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.models import db, Portfolios
from portfolio_optimizer import data_fetcher as data_fetcher_module


# ---------------------------------------------------------------------------
# CRUD routes - no network involved. All exercised as a logged-in user;
# anonymous-access redirects are covered in test_auth.py.
# ---------------------------------------------------------------------------

def _create_portfolio(app, user_id, name='Tech', stocks='AAPL,MSFT', long_only=True):
    with app.app_context():
        portfolio = Portfolios(user_id=user_id, name=name, stocks=stocks, long_only=long_only)
        db.session.add(portfolio)
        db.session.commit()
        return portfolio.id


class TestIndexRoute:
    def test_get_renders_only_current_users_portfolios(self, app, logged_in_client, create_user):
        client, user_id = logged_in_client
        _create_portfolio(app, user_id, name='Mine')

        other_user_id = create_user(username='other', email='other@example.com')
        _create_portfolio(app, other_user_id, name='NotMine')

        response = client.get('/')
        assert response.status_code == 200
        assert b'Mine' in response.data
        assert b'NotMine' not in response.data

    def test_post_creates_portfolio_owned_by_current_user(self, app, logged_in_client):
        client, user_id = logged_in_client
        response = client.post('/', data={
            'name': 'New Portfolio',
            'stocks': 'AAPL, MSFT, GOOG',
            'long_only': 'true',
        })
        assert response.status_code == 302

        with app.app_context():
            created = Portfolios.query.filter_by(name='New Portfolio').first()
            assert created is not None
            assert created.stocks == 'AAPL, MSFT, GOOG'
            assert created.user_id == user_id

    def test_post_with_blank_stocks_is_rejected(self, logged_in_client):
        client, _ = logged_in_client
        response = client.post('/', data={'name': 'Bad', 'stocks': '   ,, '})
        assert response.status_code == 200
        assert b'valid stock symbols' in response.data


class TestDeleteRoute:
    def test_delete_removes_own_portfolio(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='ToDelete')

        response = client.get(f'/delete/{portfolio_id}')
        assert response.status_code == 302

        with app.app_context():
            assert db.session.get(Portfolios, portfolio_id) is None

    def test_delete_missing_portfolio_404s(self, logged_in_client):
        client, _ = logged_in_client
        response = client.get('/delete/99999')
        assert response.status_code == 404

    def test_cannot_delete_another_users_portfolio(self, app, logged_in_client, create_user):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine')

        response = client.get(f'/delete/{other_portfolio_id}')
        assert response.status_code == 403

        with app.app_context():
            assert db.session.get(Portfolios, other_portfolio_id) is not None


class TestUpdateRoute:
    def test_get_renders_update_form_for_own_portfolio(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Editable', stocks='AAPL')
        response = client.get(f'/update/{portfolio_id}')
        assert response.status_code == 200
        assert b'AAPL' in response.data

    def test_post_updates_stocks_and_long_only(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Editable', stocks='AAPL', long_only=True)
        response = client.post(f'/update/{portfolio_id}', data={
            'stocks': 'AAPL,MSFT,GOOG',
            'long_only': 'false',
        })
        assert response.status_code == 302

        with app.app_context():
            updated = db.session.get(Portfolios, portfolio_id)
            assert updated.stocks == 'AAPL,MSFT,GOOG'
            assert updated.long_only is False

    def test_cannot_update_another_users_portfolio(self, app, logged_in_client, create_user):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine', stocks='AAPL')

        response = client.get(f'/update/{other_portfolio_id}')
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# /access - exercises the full fetch -> analyze -> render pipeline with
# yfinance mocked out, so it runs offline and deterministically.
# ---------------------------------------------------------------------------

def _fake_history_frame(n, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2022-01-03', periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({'Close': close}, index=dates)


class _FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, interval=None, period=None):
        return _fake_history_frame(750, seed=abs(hash(self.symbol)) % (2**32))


def _fake_download(symbols_or_ticker, **kwargs):
    if isinstance(symbols_or_ticker, (list, tuple)):
        symbols = symbols_or_ticker
        dates = pd.bdate_range('2022-01-03', periods=750)
        frames = {}
        for symbol in symbols:
            rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
            close = 100 + np.cumsum(rng.normal(0, 1, 750))
            frames[(symbol, 'Close')] = close
        columns = pd.MultiIndex.from_tuples(frames.keys())
        return pd.DataFrame(np.column_stack(list(frames.values())), index=dates, columns=columns)
    else:
        # e.g. "^IRX" risk-free rate lookup. yfinance returns single-ticker
        # downloads with a MultiIndex (field, ticker) column - see the
        # matching comment in test_data_fetcher.py.
        dates = pd.bdate_range('2024-01-02', periods=5)
        columns = pd.MultiIndex.from_tuples([('Close', str(symbols_or_ticker))])
        values = np.array([5.0, 5.1, 5.2, 5.25, 5.25]).reshape(-1, 1)
        return pd.DataFrame(values, index=dates, columns=columns)


@pytest.fixture
def mocked_yfinance(monkeypatch):
    monkeypatch.setattr(data_fetcher_module.yf, 'Ticker', _FakeTicker)
    monkeypatch.setattr(data_fetcher_module.yf, 'download', _fake_download)


class TestAccessRoute:
    def test_access_renders_analysis_for_own_portfolio(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        response = client.get(f'/access/{portfolio_id}')

        assert response.status_code == 200
        assert b'Tech' in response.data

    def test_access_persists_tangency_weights_to_db(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        client.get(f'/access/{portfolio_id}')

        with app.app_context():
            portfolio = db.session.get(Portfolios, portfolio_id)
            assert portfolio.weights is not None
            weights = json.loads(portfolio.weights)
            assert set(weights.keys()) == {'AAA', 'BBB'}
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_access_missing_portfolio_404s(self, logged_in_client, mocked_yfinance):
        client, _ = logged_in_client
        response = client.get('/access/99999')
        assert response.status_code == 404

    def test_cannot_access_another_users_portfolio(self, app, logged_in_client, create_user, mocked_yfinance):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine', stocks='AAA,BBB')

        response = client.get(f'/access/{other_portfolio_id}')
        assert response.status_code == 403


class TestBacktestRoute:
    def test_backtest_renders_for_own_portfolio(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        response = client.get(f'/backtest/{portfolio_id}')

        assert response.status_code == 200
        assert b'Tech' in response.data
        assert b'Equal Weight' in response.data
        assert b'Tangency' in response.data

    def test_backtest_missing_portfolio_404s(self, logged_in_client, mocked_yfinance):
        client, _ = logged_in_client
        response = client.get('/backtest/99999')
        assert response.status_code == 404

    def test_cannot_backtest_another_users_portfolio(self, app, logged_in_client, create_user, mocked_yfinance):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine', stocks='AAA,BBB')

        response = client.get(f'/backtest/{other_portfolio_id}')
        assert response.status_code == 403
