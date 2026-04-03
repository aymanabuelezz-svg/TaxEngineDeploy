"""
S&P 500 Projection Model
========================
Provides multiple projection methodologies for S&P 500 forecasting:
  - Geometric Brownian Motion (GBM)
  - Monte Carlo simulation
  - Scenario analysis (bull / base / bear)
  - Historical percentile bands

Optionally fetches live data via yfinance; falls back to hard-coded
long-run historical parameters when yfinance is unavailable.

Usage
-----
    python sp500_projection.py

    # or import and use programmatically:
    from sp500_projection import SP500ProjectionModel
    model = SP500ProjectionModel()
    results = model.run(years=10, simulations=1000)
    model.plot(results)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Historical constants (S&P 500, ~1928-2024)
# ---------------------------------------------------------------------------
HISTORICAL_ANNUAL_RETURN = 0.1012   # ~10.1% nominal CAGR
HISTORICAL_ANNUAL_VOLATILITY = 0.1965  # ~19.7% annualised std-dev
RISK_FREE_RATE = 0.045              # approximate current 10-yr Treasury yield

# Scenario overrides (annual drift, annual vol)
SCENARIOS: Dict[str, Tuple[float, float]] = {
    "bull":  (0.14, 0.15),
    "base":  (HISTORICAL_ANNUAL_RETURN, HISTORICAL_ANNUAL_VOLATILITY),
    "bear":  (0.04, 0.25),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectionParams:
    start_price: float = 5_000.0        # starting index level
    years: int = 10                     # projection horizon
    simulations: int = 2_000            # Monte Carlo paths
    steps_per_year: int = 252           # trading days per year
    annual_drift: float = HISTORICAL_ANNUAL_RETURN
    annual_volatility: float = HISTORICAL_ANNUAL_VOLATILITY
    seed: Optional[int] = 42


@dataclass
class ProjectionResults:
    params: ProjectionParams
    paths: np.ndarray                   # shape (steps+1, simulations)
    time_axis: np.ndarray               # years from today
    scenario_paths: Dict[str, np.ndarray] = field(default_factory=dict)
    percentiles: Dict[int, np.ndarray] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class SP500ProjectionModel:
    """S&P 500 projection model using Geometric Brownian Motion."""

    def __init__(self, params: Optional[ProjectionParams] = None):
        self.params = params or ProjectionParams()
        self._try_live_data()

    # ------------------------------------------------------------------
    # Optional live data fetch
    # ------------------------------------------------------------------

    def _try_live_data(self) -> None:
        """Attempt to update start_price from yfinance; silently skip on failure."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("^GSPC")
            hist = ticker.history(period="5d")
            if not hist.empty:
                self.params.start_price = float(hist["Close"].iloc[-1])
                print(f"[data] Live S&P 500 price fetched: {self.params.start_price:,.2f}")
        except Exception:
            print(f"[data] yfinance unavailable — using start price {self.params.start_price:,.0f}")

    # ------------------------------------------------------------------
    # GBM simulation
    # ------------------------------------------------------------------

    def _gbm_paths(
        self,
        start: float,
        drift: float,
        vol: float,
        total_steps: int,
        simulations: int,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return GBM price paths, shape (total_steps+1, simulations)."""
        z = rng.standard_normal((total_steps, simulations))
        log_returns = (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
        log_paths = np.vstack([
            np.zeros(simulations),
            np.cumsum(log_returns, axis=0),
        ])
        return start * np.exp(log_paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        years: Optional[int] = None,
        simulations: Optional[int] = None,
    ) -> ProjectionResults:
        """Run all projections and return a ProjectionResults object."""
        p = self.params
        if years is not None:
            p.years = years
        if simulations is not None:
            p.simulations = simulations

        rng = np.random.default_rng(p.seed)
        total_steps = p.years * p.steps_per_year
        dt = 1.0 / p.steps_per_year
        time_axis = np.linspace(0, p.years, total_steps + 1)

        # Main Monte Carlo paths
        paths = self._gbm_paths(
            p.start_price, p.annual_drift, p.annual_volatility,
            total_steps, p.simulations, dt, rng,
        )

        # Scenario single-path projections (deterministic GBM median)
        scenario_paths: Dict[str, np.ndarray] = {}
        for name, (sc_drift, sc_vol) in SCENARIOS.items():
            # Median path: exp((mu - 0.5*sigma^2) * t)
            t = time_axis
            scenario_paths[name] = p.start_price * np.exp(
                (sc_drift - 0.5 * sc_vol**2) * t
            )

        # Percentile bands from Monte Carlo
        percentile_levels = [5, 10, 25, 50, 75, 90, 95]
        percentiles = {
            pct: np.percentile(paths, pct, axis=1)
            for pct in percentile_levels
        }

        # Summary statistics at each whole-year milestone
        yearly_indices = [i * p.steps_per_year for i in range(p.years + 1)]
        year_labels = list(range(p.years + 1))
        rows = []
        for yr, idx in zip(year_labels, yearly_indices):
            slice_ = paths[idx]
            cagr = (slice_ / p.start_price) ** (1 / yr) - 1 if yr > 0 else np.nan
            rows.append({
                "year": yr,
                "median": np.median(slice_),
                "mean": np.mean(slice_),
                "p5": np.percentile(slice_, 5),
                "p25": np.percentile(slice_, 25),
                "p75": np.percentile(slice_, 75),
                "p95": np.percentile(slice_, 95),
                "median_cagr_pct": np.median(cagr) * 100 if yr > 0 else 0.0,
                "prob_above_start_pct": np.mean(slice_ > p.start_price) * 100,
            })
        summary = pd.DataFrame(rows).set_index("year")

        return ProjectionResults(
            params=p,
            paths=paths,
            time_axis=time_axis,
            scenario_paths=scenario_paths,
            percentiles=percentiles,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, results: ProjectionResults) -> None:
        """Print a formatted summary table."""
        p = results.params
        print("\n" + "=" * 70)
        print(f"  S&P 500 PROJECTION SUMMARY")
        print(f"  Start price : {p.start_price:>10,.2f}")
        print(f"  Horizon     : {p.years} years")
        print(f"  Simulations : {p.simulations:,}")
        print(f"  Drift (ann) : {p.annual_drift:.2%}")
        print(f"  Vol   (ann) : {p.annual_volatility:.2%}")
        print("=" * 70)
        df = results.summary
        print(
            df[["median", "p5", "p95", "median_cagr_pct", "prob_above_start_pct"]]
            .rename(columns={
                "median": "Median",
                "p5": "5th Pct",
                "p95": "95th Pct",
                "median_cagr_pct": "Med CAGR %",
                "prob_above_start_pct": "P(> Start) %",
            })
            .to_string(float_format=lambda x: f"{x:,.1f}")
        )
        print("=" * 70)

        # Scenario endpoint comparison
        print("\n  Scenario Endpoints (median GBM path):")
        for name, path in results.scenario_paths.items():
            end = path[-1]
            total_return = (end / p.start_price - 1) * 100
            cagr = ((end / p.start_price) ** (1 / p.years) - 1) * 100
            print(f"    {name:<6}  {end:>10,.0f}  ({total_return:+.1f}% total, {cagr:+.2f}% CAGR)")
        print()

    def plot(self, results: ProjectionResults, save_path: Optional[str] = None) -> None:
        """Plot Monte Carlo fan chart + scenario lines."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            print("[plot] matplotlib not installed — skipping plot.")
            return

        p = results.params
        t = results.time_axis

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"S&P 500 Projection Model — {p.years}-Year Horizon\n"
            f"Start: {p.start_price:,.0f}  |  {p.simulations:,} Monte Carlo simulations",
            fontsize=13,
        )

        # ---- left: fan chart ------------------------------------------
        ax = axes[0]
        ax.fill_between(t, results.percentiles[5],  results.percentiles[95],
                        alpha=0.15, color="steelblue", label="5–95th pct")
        ax.fill_between(t, results.percentiles[10], results.percentiles[90],
                        alpha=0.20, color="steelblue", label="10–90th pct")
        ax.fill_between(t, results.percentiles[25], results.percentiles[75],
                        alpha=0.30, color="steelblue", label="25–75th pct (IQR)")
        ax.plot(t, results.percentiles[50], color="steelblue", lw=2, label="Median")

        colors = {"bull": "green", "base": "black", "bear": "red"}
        for name, path in results.scenario_paths.items():
            ax.plot(t, path, color=colors[name], lw=1.5,
                    linestyle="--", label=f"{name.capitalize()} scenario")

        ax.axhline(p.start_price, color="gray", lw=0.8, linestyle=":")
        ax.set_xlabel("Years from today")
        ax.set_ylabel("Index Level")
        ax.set_title("Monte Carlo Fan Chart + Scenarios")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ---- right: terminal distribution histogram ---------------------
        ax2 = axes[1]
        terminal = results.paths[-1]
        ax2.hist(terminal, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
        for pct, color, lw in [(5, "red", 1.5), (50, "black", 2), (95, "red", 1.5)]:
            val = results.percentiles[pct][-1]
            ax2.axvline(val, color=color, lw=lw,
                        linestyle="--", label=f"P{pct}: {val:,.0f}")
        ax2.set_xlabel("Terminal Index Level")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Terminal Distribution (Year {p.years})")
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[plot] Saved to {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Tax-planning helpers
# ---------------------------------------------------------------------------

def capital_gains_projection(
    cost_basis: float,
    shares: float,
    index_path: np.ndarray,
    start_price: float,
    holding_years: int,
    lt_rate: float = 0.20,
    st_rate: float = 0.37,
    niit_rate: float = 0.038,
) -> pd.DataFrame:
    """
    Estimate potential capital gains tax for a portfolio tracking the S&P 500.

    Parameters
    ----------
    cost_basis    : average cost basis per share (or index unit)
    shares        : number of shares / units held
    index_path    : 1-D array of index values at each milestone year
    start_price   : current index level
    holding_years : array index positions to evaluate (use range(years+1))
    lt_rate       : long-term capital gains tax rate
    st_rate       : short-term capital gains tax rate
    niit_rate     : net investment income tax rate (applies to LT gains above threshold)

    Returns
    -------
    DataFrame with year, portfolio_value, unrealised_gain, lt_tax, net_proceeds
    """
    rows = []
    for yr, price in enumerate(index_path):
        portfolio_value = shares * (price / start_price) * cost_basis * (price / cost_basis)
        # Simplified: assume proportional value growth
        portfolio_value = shares * cost_basis * (price / start_price)
        unrealised_gain = portfolio_value - (shares * cost_basis)
        if yr == 0:
            rows.append({
                "year": yr,
                "portfolio_value": portfolio_value,
                "unrealised_gain": 0.0,
                "lt_tax_if_sold": 0.0,
                "net_proceeds_if_sold": portfolio_value,
            })
            continue
        rate = lt_rate + niit_rate if yr >= 1 else st_rate
        tax = max(0.0, unrealised_gain) * rate
        rows.append({
            "year": yr,
            "portfolio_value": round(portfolio_value, 2),
            "unrealised_gain": round(unrealised_gain, 2),
            "lt_tax_if_sold": round(tax, 2),
            "net_proceeds_if_sold": round(portfolio_value - tax, 2),
        })
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = ProjectionParams(
        start_price=5_000.0,
        years=10,
        simulations=2_000,
        seed=42,
    )
    model = SP500ProjectionModel(params)
    results = model.run()
    model.print_summary(results)
    model.plot(results, save_path="sp500_projection.png")

    # Example tax projection using the median path
    median_path = results.percentiles[50][:: params.steps_per_year]
    tax_df = capital_gains_projection(
        cost_basis=3_500.0,
        shares=100.0,
        index_path=median_path,
        start_price=params.start_price,
        holding_years=params.years,
    )
    print("\n  Capital Gains Tax Projection (median path, 100 shares @ $3,500 basis):")
    print(tax_df.to_string(float_format=lambda x: f"{x:,.0f}"))
    print()


if __name__ == "__main__":
    main()
