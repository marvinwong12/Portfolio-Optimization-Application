"""Matplotlib chart rendering, converted to base64 PNGs for embedding in HTML.

Every chart shares one dark theme that matches the app's UI, a colorblind-safe
palette (Okabe-Ito), and percent/dollar axis formatting. Each asset and
strategy keeps the same color across every chart it appears in.

Charts are built on `matplotlib.figure.Figure` objects rather than the global
`pyplot` state machine: pyplot is not thread-safe, and the app can be served
by a threaded server, so two requests drawing at once must not share state.
"""
import io
import base64

import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter
import seaborn as sns

# --- Theme ---------------------------------------------------------------
BG = '#1e1e1e'            # matches the app's card background
FG = '#ececec'
MUTED = '#a0a0a0'
SPINE = '#444444'
GRID = '#ffffff'
ACCENT = '#4a86e8'
GOLD = '#ffd54a'

# Okabe-Ito, brightened for dark backgrounds; cycled for more than 8 assets.
ASSET_COLORS = [
    '#56B4E9', '#E69F00', '#009E73', '#F0E442',
    '#CC79A7', '#D55E00', '#7aa6ff', '#b8b8b8',
    '#8dd3c7', '#bebada', '#fb8072', '#b3de69',
]
STRATEGY_COLORS = {
    'Tangency': '#E69F00',
    'Minimum Variance': '#56B4E9',
    'Equal Weight': '#009E73',
    'Monte Carlo Optimal': '#CC79A7',
}
BENCHMARK_COLOR = '#b8b8b8'
POSITIVE = '#009E73'
NEGATIVE = '#D55E00'

PNG_DPI = 150


def _asset_color(index):
    return ASSET_COLORS[index % len(ASSET_COLORS)]


def _figure(figsize, **subplots_kwargs):
    """A themed Figure and its axes (a single Axes, or an array for grids)."""
    fig = Figure(figsize=figsize, facecolor=BG)
    axes = fig.subplots(**subplots_kwargs)
    for ax in np.atleast_1d(axes).ravel():
        _style_axes(ax)
    return fig, axes


def _style_axes(ax, grid_axis='both'):
    ax.set_facecolor(BG)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, color=SPINE)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(True, axis=grid_axis, color=GRID, alpha=0.08, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, text, subtitle=None):
    ax.set_title(text, loc='left', color=FG, fontsize=14, fontweight='bold', pad=26 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9.5, va='bottom')


def _style_legend(legend):
    if legend is None:
        return
    legend.get_frame().set_facecolor(BG)
    legend.get_frame().set_edgecolor(SPINE)
    for text in legend.get_texts():
        text.set_color(FG)


def _style_colorbar(cbar, label):
    cbar.set_label(label, color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=2)
    cbar.outline.set_visible(False)


def _percent_axis(axis, decimals=None):
    """Percent tick labels. decimals=None lets matplotlib choose the precision
    from the tick spacing, so ticks 2.5% apart aren't rounded to whole
    percents (which would label 32.5% as "33%")."""
    axis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def _dollar_axis(axis):
    axis.set_major_formatter(FuncFormatter(lambda value, _: f'${value:,.2f}'))


def _date_axis(ax, start=None, end=None):
    """Concise date ticks. When start/end are given, ticks are restricted to
    that range so the empty margin reserved for end-of-line labels doesn't
    get misleading ticks (e.g. a "2027" tick beyond the last data point)."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    if start is not None and end is not None:
        low, high = mdates.date2num(start), mdates.date2num(end)
        ticks = [t for t in locator.tick_values(start, end) if low <= t <= high]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
    else:
        ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _spread_labels(positions, min_gap):
    """Nudge label y-positions apart so end-of-line labels don't overlap,
    keeping their order. Returns adjusted positions in the input order."""
    order = np.argsort(positions)
    adjusted = np.array(positions, dtype=float)
    for previous, current in zip(order[:-1], order[1:]):
        if adjusted[current] - adjusted[previous] < min_gap:
            adjusted[current] = adjusted[previous] + min_gap
    # Re-centre so the whole stack stays close to the original positions.
    adjusted += (np.mean(positions) - np.mean(adjusted))
    return adjusted


class PortfolioVisualizer:
    """
    A class to create visualizations for portfolio analysis.

    Static Methods:
        plot_efficient_frontier: Monte Carlo cloud + exact frontier, assets, capital market line
        plot_weights: Portfolio weights as a donut (long-only) or sorted bars (long-short)
        plot_correlation_matrix: Correlation heatmap
        plot_returns_time_series: Cumulative returns per asset, directly labeled
        plot_backtest_comparison: Walk-forward equity curves with a drawdown panel
        plot_weight_history: How a strategy's weights drifted over time
    """

    @staticmethod
    def plot_to_base64(fig):
        """Render a Figure to a base64-encoded PNG string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=PNG_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        return base64.b64encode(buffer.getvalue()).decode('utf8')

    @staticmethod
    def plot_efficient_frontier(returns, volatilities, sharpe_ratios, optimal_portfolio=None,
                                 frontier=None, save_path=None, min_variance_portfolio=None,
                                 assets=None, risk_free_rate=None):
        """
        Plot the Monte Carlo cloud of random portfolios with the exact
        efficient frontier (solved, not sampled) on top, the tangency and
        minimum-variance portfolios, the capital market line, and where each
        individual asset sits.

        Args:
            returns, volatilities, sharpe_ratios (np.array): Monte Carlo cloud
            optimal_portfolio (dict): Tangency portfolio metrics
                ('volatility', 'return', and optionally 'sharpe_ratio')
            frontier (dict): {'returns', 'volatilities'} from
                PortfolioAnalyzer.efficient_frontier()
            save_path (str): Unused; kept for backward compatibility
            min_variance_portfolio (dict): Minimum-variance portfolio metrics
            assets (dict): {symbol: {'volatility', 'return'}} for each asset
            risk_free_rate (float): Draws the capital market line from the
                risk-free rate through the tangency portfolio

        Returns:
            str: Base64 encoded image data
        """
        fig, ax = _figure((11, 7))
        volatilities = np.asarray(volatilities)
        returns = np.asarray(returns)

        cloud = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', s=12,
                           alpha=0.5, linewidths=0, zorder=1, rasterized=True)
        _style_colorbar(fig.colorbar(cloud, ax=ax, pad=0.02, fraction=0.04), 'Sharpe ratio')

        xs, ys = [volatilities], [returns]

        if frontier is not None and len(frontier.get('returns', [])) > 0:
            ax.plot(frontier['volatilities'], frontier['returns'], color='white', linewidth=2.6,
                    label='Efficient frontier (exact)', zorder=3)
            xs.append(np.asarray(frontier['volatilities']))
            ys.append(np.asarray(frontier['returns']))

        if optimal_portfolio:
            vol, ret = optimal_portfolio['volatility'], optimal_portfolio['return']
            sharpe = optimal_portfolio.get('sharpe_ratio')
            label = 'Max Sharpe (tangency)' + (f', {sharpe:.2f}' if sharpe is not None else '')
            ax.scatter([vol], [ret], color=GOLD, s=280, marker='*', edgecolors='black',
                       linewidths=0.8, label=label, zorder=6)
            xs.append(np.array([vol]))
            ys.append(np.array([ret]))

            if risk_free_rate is not None and vol > 0 and ret > risk_free_rate:
                x_max = float(max(np.max(volatilities), vol) * 1.15)
                slope = (ret - risk_free_rate) / vol
                ax.plot([0, x_max], [risk_free_rate, risk_free_rate + slope * x_max],
                        linestyle='--', color=ACCENT, linewidth=1.4,
                        label='Capital market line', zorder=2)

        if min_variance_portfolio:
            vol, ret = min_variance_portfolio['volatility'], min_variance_portfolio['return']
            ax.scatter([vol], [ret], color='#56B4E9', s=110, marker='D', edgecolors='black',
                       linewidths=0.8, label='Minimum variance', zorder=6)
            xs.append(np.array([vol]))
            ys.append(np.array([ret]))

        for index, (symbol, stats) in enumerate((assets or {}).items()):
            ax.scatter([stats['volatility']], [stats['return']], s=70, marker='o',
                       facecolors=BG, edgecolors=_asset_color(index), linewidths=1.8, zorder=5)
            ax.annotate(symbol, (stats['volatility'], stats['return']), xytext=(7, 6),
                        textcoords='offset points', color=FG, fontsize=9, zorder=7)
            xs.append(np.array([stats['volatility']]))
            ys.append(np.array([stats['return']]))

        all_x, all_y = np.concatenate(xs), np.concatenate(ys)
        x_pad = (all_x.max() - all_x.min()) * 0.06 or 0.01
        y_pad = (all_y.max() - all_y.min()) * 0.06 or 0.01
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

        ax.set_xlabel('Annualized volatility (risk)')
        ax.set_ylabel('Annualized return')
        _percent_axis(ax.xaxis)
        _percent_axis(ax.yaxis)
        _title(ax, 'Risk vs. Return',
               'Each dot is a random portfolio; the white line is the best return available at each level of risk')
        if ax.get_legend_handles_labels()[0]:
            _style_legend(ax.legend(loc='lower right', fontsize=9))

        return PortfolioVisualizer.plot_to_base64(fig)

    @staticmethod
    def plot_weights(weights, symbols, title, long_only=True):
        """
        Plot portfolio weights: a donut with a percentage legend for
        long-only portfolios, sorted horizontal bars (which can show negative
        weights) for long-short ones.

        Args:
            weights (np.array): Portfolio weights
            symbols (list): Asset symbols
            title (str): Chart title
            long_only (bool): Whether the portfolio is long-only

        Returns:
            str: Base64 encoded image data
        """
        weights = np.asarray(weights, dtype=float)
        symbols = [str(symbol) for symbol in symbols]
        colors = [_asset_color(i) for i in range(len(symbols))]  # stable per asset
        order = np.argsort(-weights)

        if long_only:
            fig, ax = _figure((9, 5.2))
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            held = [i for i in order if weights[i] > 1e-4]
            wedges, _ = ax.pie(
                weights[held], colors=[colors[i] for i in held], startangle=90, counterclock=False,
                wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=2.5),
            )
            total = weights[held].sum()
            for wedge, i in zip(wedges, held):
                share = weights[i] / total
                if share < 0.06:
                    continue  # too thin to label inside; the legend has it
                angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
                ax.text(0.81 * np.cos(angle), 0.81 * np.sin(angle), f'{share:.0%}', ha='center',
                        va='center', color='#111111', fontsize=10, fontweight='bold')
            ax.text(0, 0, f'{len(held)}\nholdings', ha='center', va='center', color=FG,
                    fontsize=13, fontweight='bold', linespacing=1.2)

            handles = [Patch(facecolor=colors[i], edgecolor='none') for i in order]
            labels = [f'{symbols[i]}   {weights[i]:.1%}' for i in order]
            _style_legend(ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
                                    frameon=False, fontsize=10, handlelength=1.1))
            ax.set_title(title, loc='left', color=FG, fontsize=14, fontweight='bold', pad=10)
        else:
            fig, ax = _figure((10, max(3.2, 0.5 * len(symbols) + 1.6)))
            ax.grid(True, axis='x', color=GRID, alpha=0.08)
            ax.grid(False, axis='y')
            positions = np.arange(len(order))[::-1]
            values = weights[order]
            ax.barh(positions, values, color=[POSITIVE if v >= 0 else NEGATIVE for v in values],
                    height=0.62, zorder=3)
            span = max(np.abs(values).max(), 0.01)
            for position, value in zip(positions, values):
                ax.text(value + (0.015 * span if value >= 0 else -0.015 * span), position,
                        f'{value:.1%}', va='center', ha='left' if value >= 0 else 'right',
                        color=FG, fontsize=10)
            ax.axvline(0, color=MUTED, linewidth=1, zorder=4)
            ax.set_yticks(positions)
            ax.set_yticklabels([symbols[i] for i in order], color=FG, fontsize=10)
            ax.set_xlim(min(values.min(), 0) - 0.18 * span, max(values.max(), 0) + 0.18 * span)
            _percent_axis(ax.xaxis)
            ax.set_title(title, loc='left', color=FG, fontsize=14, fontweight='bold', pad=10)

        return PortfolioVisualizer.plot_to_base64(fig)

    @staticmethod
    def plot_correlation_matrix(correlation_matrix, save_path=None):
        """
        Plot the correlation matrix as a lower-triangle heatmap using a
        blue-to-red diverging palette (safe for red-green color blindness).

        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            save_path (str): Unused; kept for backward compatibility

        Returns:
            str: Base64 encoded image data
        """
        # Only the pairs: each asset's correlation with itself is always 1.00
        # and, drawn as the darkest cells, would dominate the picture while
        # saying nothing. Dropping the diagonal leaves a clean staircase
        # (rows 2..n by columns 1..n-1). A lone asset has no pairs, so it
        # falls back to showing its single self-correlation.
        pairs = correlation_matrix.iloc[1:, :-1] if len(correlation_matrix) > 1 else correlation_matrix
        mask = np.triu(np.ones(pairs.shape, dtype=bool), k=1) if len(correlation_matrix) > 1 else None

        size = max(5.5, 0.85 * len(correlation_matrix) + 2.5)
        fig, ax = _figure((size + 1, size))
        ax.grid(False)
        sns.heatmap(
            pairs, ax=ax, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True, linewidths=2.5, linecolor=BG,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.75, 'pad': 0.03},
        )
        ax.tick_params(colors=FG, labelsize=10, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        _style_colorbar(ax.collections[0].colorbar, 'Correlation')
        _title(ax, 'Correlation Between Assets', 'Lower means better diversification')

        return PortfolioVisualizer.plot_to_base64(fig)

    @staticmethod
    def plot_returns_time_series(returns_data, save_path=None):
        """
        Plot cumulative returns per asset, each line labeled at its end
        instead of through a legend.

        Args:
            returns_data (pd.DataFrame): Daily returns
            save_path (str): Unused; kept for backward compatibility

        Returns:
            str: Base64 encoded image data
        """
        fig, ax = _figure((11, 5.8))
        cumulative = (1 + returns_data).cumprod() - 1
        ax.axhline(0, color=MUTED, linewidth=0.9, alpha=0.6)

        for index, column in enumerate(cumulative.columns):
            ax.plot(cumulative.index, cumulative[column], color=_asset_color(index), linewidth=1.8)

        finals = np.array([cumulative[column].iloc[-1] for column in cumulative.columns])
        y_range = (cumulative.values.max() - cumulative.values.min()) or 1.0
        label_y = _spread_labels(finals, min_gap=0.045 * y_range)
        label_pad = (cumulative.index[-1] - cumulative.index[0]) * 0.012
        for index, column in enumerate(cumulative.columns):
            ax.annotate(f'{column}  {finals[index]:+.0%}', xy=(cumulative.index[-1], finals[index]),
                        xytext=(cumulative.index[-1] + label_pad, label_y[index]),
                        textcoords=('data', 'data'), xycoords='data',
                        color=_asset_color(index), fontsize=9.5, fontweight='bold',
                        va='center', ha='left',
                        annotation_clip=False,
                        arrowprops=None)
        # Room on the right for the end labels.
        span = cumulative.index[-1] - cumulative.index[0]
        ax.set_xlim(cumulative.index[0], cumulative.index[-1] + span * 0.13)

        _percent_axis(ax.yaxis)
        _date_axis(ax, cumulative.index[0], cumulative.index[-1])
        ax.set_ylabel('Cumulative return')
        _title(ax, 'Cumulative Returns')

        return PortfolioVisualizer.plot_to_base64(fig)

    @staticmethod
    def plot_backtest_comparison(equity_curves):
        """
        Plot walk-forward backtest equity curves on top, with an "underwater"
        drawdown panel beneath, so realized (out-of-sample) return and the
        pain of getting it can be read together. Benchmarks (names starting
        with "SPY") are drawn dashed and grey as a reference line.

        Args:
            equity_curves (dict): {name: pd.Series} of growth-of-$1 curves.

        Returns:
            str: Base64 encoded image data
        """
        fig, (ax, ax_dd) = _figure((11, 7.6), nrows=2, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1.15], 'hspace': 0.08})

        def color_for(name, index):
            if name.startswith('SPY'):
                return BENCHMARK_COLOR
            return STRATEGY_COLORS.get(name, _asset_color(index))

        ax.axhline(1.0, color=MUTED, linewidth=0.9, alpha=0.6)
        finals = []
        for index, (name, curve) in enumerate(equity_curves.items()):
            benchmark = name.startswith('SPY')
            color = color_for(name, index)
            ax.plot(curve.index, curve.values, color=color, linewidth=1.6 if benchmark else 2.1,
                    linestyle='--' if benchmark else '-', zorder=2 if benchmark else 3)
            drawdown = curve / curve.cummax() - 1
            ax_dd.plot(drawdown.index, drawdown.values, color=color, linewidth=1.1,
                       linestyle='--' if benchmark else '-', alpha=0.95)
            finals.append(float(curve.iloc[-1]))

        first_curve = next(iter(equity_curves.values()))
        y_min = min(c.min() for c in equity_curves.values())
        y_range = (max(finals) - y_min) or 1.0
        label_y = _spread_labels(finals, min_gap=0.075 * y_range)
        label_pad = (first_curve.index[-1] - first_curve.index[0]) * 0.012
        for index, (name, curve) in enumerate(equity_curves.items()):
            ax.annotate(f'{name}  ${finals[index]:.2f}', xy=(curve.index[-1], finals[index]),
                        xytext=(curve.index[-1] + label_pad, label_y[index]), xycoords='data',
                        textcoords='data', color=color_for(name, index), fontsize=9.5,
                        fontweight='bold', va='center', ha='left', annotation_clip=False)
        span = first_curve.index[-1] - first_curve.index[0]
        ax.set_xlim(first_curve.index[0], first_curve.index[-1] + span * 0.20)

        _dollar_axis(ax.yaxis)
        ax.set_ylabel('Growth of $1')
        ax.tick_params(labelbottom=False)
        _title(ax, 'Walk-Forward Backtest (Out-of-Sample)',
               'Growth of $1 after transaction costs; lower panel shows the decline from each running peak')

        _percent_axis(ax_dd.yaxis, decimals=0)
        ax_dd.set_ylabel('Drawdown')
        ax_dd.axhline(0, color=MUTED, linewidth=0.9, alpha=0.6)
        _date_axis(ax_dd, first_curve.index[0], first_curve.index[-1])

        return PortfolioVisualizer.plot_to_base64(fig)

    @staticmethod
    def plot_weight_history(dates, weights_matrix, symbols, title):
        """
        Plot how one strategy's weight allocation has drifted across
        successive analysis runs - a line per asset, not just the most
        recent snapshot.

        Args:
            dates (list): Snapshot timestamps, ascending
            weights_matrix (np.array): Shape (len(dates), len(symbols))
            symbols (list): Asset symbols, matching weights_matrix's columns
            title (str): Chart title

        Returns:
            str: Base64 encoded image data
        """
        weights_matrix = np.asarray(weights_matrix, dtype=float)
        fig, ax = _figure((11, 5.6))
        for index, symbol in enumerate(symbols):
            ax.plot(dates, weights_matrix[:, index], color=_asset_color(index), linewidth=2,
                    marker='o', markersize=5, markeredgecolor=BG, markeredgewidth=1.2, label=symbol)

        ax.set_ylim(-0.02, max(1.0, weights_matrix.max() + 0.05))
        _percent_axis(ax.yaxis)
        _date_axis(ax)
        ax.set_ylabel('Weight')
        _title(ax, title, 'Each point is one analysis run')
        _style_legend(ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10))

        return PortfolioVisualizer.plot_to_base64(fig)
