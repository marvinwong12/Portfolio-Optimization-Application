import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.portfolio_service import PortfolioApp
from portfolio_optimizer.analyzer import PortfolioAnalyzer


@pytest.fixture
def portfolio_app_with_data(synthetic_returns):
    """A PortfolioApp with returns/analyzer set directly, bypassing the
    network-dependent fetch_data so this stays a pure unit test."""
    app = PortfolioApp()
    app.returns = synthetic_returns
    app.risk_free_rate = 0.04
    app.long_only = True
    app.analyzer = PortfolioAnalyzer(synthetic_returns, app.risk_free_rate, long_only=True)
    return app


class TestStoreAndRetrievePortfolio:
    def test_store_then_get_portfolio(self, portfolio_app_with_data):
        weights = np.ones(4) / 4
        portfolio_app_with_data.store_portfolio('Equal Weight', weights, 'benchmark')

        stored = portfolio_app_with_data.get_portfolio('Equal Weight')
        assert stored is not None
        assert np.array_equal(stored['weights'], weights)
        assert stored['description'] == 'benchmark'
        assert stored['assets'] == list(portfolio_app_with_data.returns.columns)

    def test_get_missing_portfolio_returns_none(self, portfolio_app_with_data):
        assert portfolio_app_with_data.get_portfolio('Nonexistent') is None

    def test_remove_portfolio(self, portfolio_app_with_data):
        portfolio_app_with_data.store_portfolio('P1', np.ones(4) / 4)
        assert portfolio_app_with_data.remove_portfolio('P1') is True
        assert portfolio_app_with_data.get_portfolio('P1') is None
        assert portfolio_app_with_data.remove_portfolio('P1') is False

    def test_get_portfolio_names(self, portfolio_app_with_data):
        portfolio_app_with_data.store_portfolio('P1', np.ones(4) / 4)
        portfolio_app_with_data.store_portfolio('P2', np.ones(4) / 4)
        assert set(portfolio_app_with_data.get_portfolio_names()) == {'P1', 'P2'}


class TestPortfolioPerformance:
    def test_get_portfolio_metrics_matches_analyzer(self, portfolio_app_with_data):
        weights = np.ones(4) / 4
        portfolio_app_with_data.store_portfolio('Equal', weights)

        metrics = portfolio_app_with_data.get_portfolio_metrics('Equal')
        expected = portfolio_app_with_data.analyzer.calculate_portfolio_metrics(weights)
        assert metrics['sharpe_ratio'] == pytest.approx(expected['sharpe_ratio'])

    def test_compare_portfolios_returns_all_stored(self, portfolio_app_with_data):
        portfolio_app_with_data.store_portfolio('A', np.ones(4) / 4)
        portfolio_app_with_data.store_portfolio('B', np.array([0.4, 0.3, 0.2, 0.1]))

        comparison = portfolio_app_with_data.compare_portfolios(['A', 'B', 'Missing'])
        assert set(comparison.keys()) == {'A', 'B'}


class TestExportImportPortfolios:
    def test_export_then_import_round_trip(self, portfolio_app_with_data, tmp_path):
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        portfolio_app_with_data.store_portfolio('P1', weights, 'test portfolio')

        out_file = tmp_path / 'export.csv'
        assert portfolio_app_with_data.export_portfolios(str(out_file)) is True
        assert out_file.exists()

        fresh_app = PortfolioApp()
        fresh_app.returns = portfolio_app_with_data.returns  # needed for store_portfolio's asset list
        assert fresh_app.import_portfolios(str(out_file)) is True

        imported = fresh_app.get_portfolio('P1')
        assert imported is not None
        np.testing.assert_allclose(imported['weights'], weights)

    def test_export_with_no_portfolios_returns_false(self):
        app = PortfolioApp()
        assert app.export_portfolios('unused.csv') is False

    def test_import_missing_file_returns_false(self):
        app = PortfolioApp()
        assert app.import_portfolios('does_not_exist.csv') is False
