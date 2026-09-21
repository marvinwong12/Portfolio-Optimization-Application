import json

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.models import db, Portfolios, PortfolioSnapshot
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


class TestLoadingOverlayWiring:
    """Smoke tests that the loading-overlay opt-in markers are actually
    present on the slow actions - catches a template regression that would
    silently bring back the "page looks frozen" UX bug."""

    def test_index_page_wires_up_access_link_and_forms(self, app, logged_in_client):
        client, user_id = logged_in_client
        _create_portfolio(app, user_id, name='Tech')

        response = client.get('/')
        html = response.data.decode()
        assert 'js-loading-trigger' in html
        assert 'data-loading-message="Analyzing Tech' in html
        assert 'data-loading-message="Checking ticker symbols and creating your portfolio' in html
        assert 'data-loading-message="Fetching and analyzing stock data' in html

    def test_update_page_wires_up_form(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Editable')

        response = client.get(f'/update/{portfolio_id}')
        assert b'js-loading-trigger' in response.data


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

    def test_post_creates_portfolio_owned_by_current_user(self, app, logged_in_client, mocked_yfinance):
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
        assert b'at least one stock symbol' in response.data

    def test_post_with_invalid_ticker_is_rejected_and_not_saved(self, app, logged_in_client, mocked_yfinance):
        """Regression test: creating a portfolio with a bad ticker used to
        save it anyway and only fail later when the user clicked Access."""
        client, _ = logged_in_client
        response = client.post('/', data={
            'name': 'Bad Portfolio',
            'stocks': f'AAPL,{INVALID_SYMBOL}',
            'long_only': 'true',
        })
        assert response.status_code == 200
        assert INVALID_SYMBOL.encode() in response.data
        assert b'Could not find data for' in response.data

        with app.app_context():
            assert Portfolios.query.filter_by(name='Bad Portfolio').first() is None

    def test_post_with_invalid_ticker_preserves_entered_form_values(self, logged_in_client, mocked_yfinance):
        client, _ = logged_in_client
        response = client.post('/', data={
            'name': 'Bad Portfolio',
            'stocks': f'AAPL,{INVALID_SYMBOL}',
            'long_only': 'true',
        })
        assert b'value="Bad Portfolio"' in response.data
        assert f'value="AAPL,{INVALID_SYMBOL}"'.encode() in response.data


class TestDeleteRoute:
    def test_delete_removes_own_portfolio(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='ToDelete')

        response = client.post(f'/delete/{portfolio_id}')
        assert response.status_code == 302

        with app.app_context():
            assert db.session.get(Portfolios, portfolio_id) is None

    def test_delete_missing_portfolio_404s(self, logged_in_client):
        client, _ = logged_in_client
        response = client.post('/delete/99999')
        assert response.status_code == 404

    def test_cannot_delete_another_users_portfolio(self, app, logged_in_client, create_user):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine')

        response = client.post(f'/delete/{other_portfolio_id}')
        assert response.status_code == 403

    def test_delete_no_longer_accepts_get(self, app, logged_in_client):
        """Regression test: /delete used to be a plain GET link, which is
        forgeable via e.g. an <img src="..."> on another site regardless of
        CSRF protection (CSRFProtect only guards state-changing HTTP
        methods). It must be POST-only."""
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='StillThere')

        response = client.get(f'/delete/{portfolio_id}')
        assert response.status_code == 405

        with app.app_context():
            assert db.session.get(Portfolios, portfolio_id) is not None


class TestUpdateRoute:
    def test_get_renders_update_form_for_own_portfolio(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Editable', stocks='AAPL')
        response = client.get(f'/update/{portfolio_id}')
        assert response.status_code == 200
        assert b'AAPL' in response.data

    def test_post_updates_stocks_and_long_only(self, app, logged_in_client, mocked_yfinance):
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

    def test_post_with_invalid_ticker_is_rejected_and_leaves_portfolio_unchanged(
        self, app, logged_in_client, mocked_yfinance
    ):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Editable', stocks='AAPL', long_only=True)

        response = client.post(f'/update/{portfolio_id}', data={
            'stocks': f'AAPL,{INVALID_SYMBOL}',
            'long_only': 'true',
        })
        assert response.status_code == 200
        assert INVALID_SYMBOL.encode() in response.data
        assert b'Could not find data for' in response.data

        with app.app_context():
            unchanged = db.session.get(Portfolios, portfolio_id)
            assert unchanged.stocks == 'AAPL'  # not overwritten with the bad value

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


# A symbol name reserved in these fakes to simulate a ticker yfinance can't
# find (delisted / typo) - i.e. a real column that's entirely NaN, since
# that's what triggers the "invalid symbol" branches in the app code.
INVALID_SYMBOL = 'BADTICKER'


def _fake_download(symbols_or_ticker, **kwargs):
    if isinstance(symbols_or_ticker, (list, tuple)):
        symbols = symbols_or_ticker
        dates = pd.bdate_range('2022-01-03', periods=750)
        frames = {}
        for symbol in symbols:
            if symbol == INVALID_SYMBOL:
                close = np.full(750, np.nan)
            else:
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
        assert b'js-loading-trigger' in response.data  # backtest link wired up

    def test_access_shows_portfolio_level_risk_analysis(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        response = client.get(f'/access/{portfolio_id}')

        assert b'Risk Analysis' in response.data
        assert b'1-Day VaR (95%)' in response.data
        assert b'1-Day CVaR (95%)' in response.data
        assert b'Max Drawdown' in response.data
        assert b'Share of Risk' in response.data

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

    def test_access_creates_a_snapshot_for_every_strategy(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        client.get(f'/access/{portfolio_id}')

        with app.app_context():
            snapshots = PortfolioSnapshot.query.filter_by(portfolio_id=portfolio_id).all()
            strategies = {s.strategy for s in snapshots}
            assert strategies == {'Minimum Variance', 'Tangency', 'Equal Weight', 'Monte Carlo Optimal'}
            for snapshot in snapshots:
                weights = json.loads(snapshot.weights)
                assert set(weights.keys()) == {'AAA', 'BBB'}
                assert snapshot.sharpe_ratio is not None

    def test_access_accumulates_snapshots_across_multiple_runs(self, app, logged_in_client, mocked_yfinance):
        """Regression-shaped test: re-running Access must add new snapshot
        rows, not overwrite/replace the previous run's - that history is
        the entire point of the feature."""
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        client.get(f'/access/{portfolio_id}')
        client.get(f'/access/{portfolio_id}')

        with app.app_context():
            tangency_snapshots = (PortfolioSnapshot.query
                                   .filter_by(portfolio_id=portfolio_id, strategy='Tangency')
                                   .all())
            assert len(tangency_snapshots) == 2


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


def _add_snapshot(app, portfolio_id, strategy, weights, created_at=None,
                   portfolio_return=0.1, volatility=0.15, sharpe_ratio=0.5):
    with app.app_context():
        snapshot = PortfolioSnapshot(
            portfolio_id=portfolio_id,
            strategy=strategy,
            weights=json.dumps(weights),
            portfolio_return=portfolio_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
        )
        if created_at is not None:
            snapshot.created_at = created_at
        db.session.add(snapshot)
        db.session.commit()
        return snapshot.id


class TestHistoryRoute:
    def test_redirects_with_flash_when_no_snapshots_yet(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech')

        response = client.get(f'/history/{portfolio_id}')
        assert response.status_code == 302
        assert response.headers['Location'] == f'/access/{portfolio_id}'

    def test_missing_portfolio_404s(self, logged_in_client):
        client, _ = logged_in_client
        response = client.get('/history/99999')
        assert response.status_code == 404

    def test_cannot_view_another_users_history(self, app, logged_in_client, create_user):
        client, _ = logged_in_client
        other_user_id = create_user(username='other', email='other@example.com')
        other_portfolio_id = _create_portfolio(app, other_user_id, name='NotMine')
        _add_snapshot(app, other_portfolio_id, 'Tangency', {'AAPL': 1.0})

        response = client.get(f'/history/{other_portfolio_id}')
        assert response.status_code == 403

    def test_renders_snapshot_table_and_chart_with_multiple_snapshots(self, app, logged_in_client):
        import datetime as dt

        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAPL,MSFT')
        _add_snapshot(app, portfolio_id, 'Tangency', {'AAPL': 0.7, 'MSFT': 0.3},
                      created_at=dt.datetime(2024, 1, 1))
        _add_snapshot(app, portfolio_id, 'Tangency', {'AAPL': 0.5, 'MSFT': 0.5},
                      created_at=dt.datetime(2024, 4, 1))

        response = client.get(f'/history/{portfolio_id}')
        assert response.status_code == 200
        assert b'Tech' in response.data
        assert b'Tangency' in response.data
        assert b'<img src="data:image/png;base64,' in response.data  # chart rendered

    def test_single_snapshot_shows_table_without_a_chart(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAPL')
        _add_snapshot(app, portfolio_id, 'Tangency', {'AAPL': 1.0})

        response = client.get(f'/history/{portfolio_id}')
        assert response.status_code == 200
        assert b'<img src="data:image/png;base64,' not in response.data
        assert b'only one snapshot' in response.data.lower()

    def test_strategy_query_param_switches_the_view(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAPL,MSFT')
        _add_snapshot(app, portfolio_id, 'Tangency', {'AAPL': 0.7, 'MSFT': 0.3})
        _add_snapshot(app, portfolio_id, 'Equal Weight', {'AAPL': 0.5, 'MSFT': 0.5})

        default_response = client.get(f'/history/{portfolio_id}')
        assert b'Tangency Weights Over Time' in default_response.data

        switched_response = client.get(f'/history/{portfolio_id}?strategy=Equal Weight')
        assert b'Equal Weight Weights Over Time' in switched_response.data

    def test_unknown_strategy_query_param_falls_back_to_first_available(self, app, logged_in_client):
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAPL')
        _add_snapshot(app, portfolio_id, 'Tangency', {'AAPL': 1.0})

        response = client.get(f'/history/{portfolio_id}?strategy=NotARealStrategy')
        assert response.status_code == 200
        assert b'Tangency Weights Over Time' in response.data


class TestBacktestBenchmarkCostsAndConfidence:
    def _get(self, app, client, user_id, query=''):
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)
        return client.get(f'/backtest/{portfolio_id}{query}')

    def test_includes_the_spy_benchmark(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        response = self._get(app, client, user_id)
        assert response.status_code == 200
        assert b'SPY (Buy &amp; Hold)' in response.data

    def test_shows_confidence_intervals_and_significance_verdicts(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        html = self._get(app, client, user_id).data.decode()
        assert 'Sharpe Ratio (95% CI)' in html
        assert 'Are the Differences Real?' in html
        assert 'Compared with Equal Weight' in html
        assert 'Compared with SPY (Buy &amp; Hold)' in html
        assert any(verdict in html for verdict in
                   ('Significantly better', 'Significantly worse', 'Not distinguishable'))

    def test_shows_turnover_column(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        assert b'Annual Turnover' in self._get(app, client, user_id).data

    def test_cost_defaults_to_10_bps(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        assert b'name="cost_bps"' in self._get(app, client, user_id).data
        assert b'value="10"' in self._get(app, client, user_id).data

    def test_cost_query_param_is_used(self, app, logged_in_client, mocked_yfinance):
        client, user_id = logged_in_client
        response = self._get(app, client, user_id, '?cost_bps=50')
        assert b'value="50"' in response.data

    @pytest.mark.parametrize('bad_value', ['abc', '-5', '9999', 'nan', 'inf', ''])
    def test_invalid_cost_falls_back_to_the_default(self, app, logged_in_client, mocked_yfinance, bad_value):
        client, user_id = logged_in_client
        response = self._get(app, client, user_id, f'?cost_bps={bad_value}')
        assert response.status_code == 200
        assert b'value="10"' in response.data

    def test_higher_cost_lowers_reported_returns(self, app, logged_in_client, mocked_yfinance):
        import re
        client, user_id = logged_in_client
        portfolio_id = _create_portfolio(app, user_id, name='Tech', stocks='AAA,BBB', long_only=True)

        def tangency_total_return(cost):
            html = client.get(f'/backtest/{portfolio_id}?cost_bps={cost}').data.decode()
            row = html.split('<strong>Tangency</strong>')[1]
            return float(re.search(r'(-?\d+\.\d+)%', row).group(1))

        assert tangency_total_return(200) < tangency_total_return(0)

    def test_page_still_works_when_the_benchmark_cannot_be_loaded(
        self, app, logged_in_client, mocked_yfinance, monkeypatch
    ):
        from portfolio_optimizer import routes

        def broken(*args, **kwargs):
            raise RuntimeError('benchmark feed down')

        monkeypatch.setattr(routes, '_fetch_benchmark_returns', broken)
        client, user_id = logged_in_client
        response = self._get(app, client, user_id)

        assert response.status_code == 200
        assert b'SPY (Buy &amp; Hold)' not in response.data
        assert b"couldn't be loaded" in response.data
        assert b'Compared with Equal Weight' in response.data


class TestParseCostBps:
    @pytest.mark.parametrize('raw, expected', [
        (None, 10.0), ('', 10.0), ('abc', 10.0), ('-1', 10.0), ('201', 10.0),
        ('nan', 10.0), ('inf', 10.0), ('0', 0.0), ('25', 25.0), ('7.5', 7.5), ('200', 200.0),
    ])
    def test_parsing_and_fallback(self, raw, expected):
        from portfolio_optimizer.routes import _parse_cost_bps
        assert _parse_cost_bps(raw) == expected
