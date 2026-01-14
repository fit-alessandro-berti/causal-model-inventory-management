#!/usr/bin/env python3
"""
simulate_ocel_inventory_sem.py

Structural Equation Model (SEM) with latent factors for:
"Causal Model for Inventory Management Undesirable States"
(Understock / Overstock as undesirable inventory states).

This script:
  1) Defines an SEM in semopy syntax (latent + observed variables)
  2) Defines the random variables / distributions (a generative SCM)
  3) Simulates a dataset of *observable* event-log-derived variables
  4) Saves the simulated data to CSV
  5) (Optional) Fits the SEM to the simulated data with semopy

-----------------------------------------------------------------------
Observable variables INCLUDED are intentionally restricted to those
computable from an object-centric event log (OCEL) with:
  - activities
  - timestamps
  - quantities (e.g., GI / GR / PO quantities)
  - stock level snapshots OR reconstructable stock levels

Each row in the simulated dataset corresponds to one object instance,
e.g., a product-location (SKU-Plant, MaterialPlant_ID, etc.).

-----------------------------------------------------------------------
Event-log computable observables (per product-location object)
-----------------------------------------------------------------------
demand_total
  Sum of demand quantities over the window:
    sum(|Quantity|) where activity == PostGoodsIssue

demand_cv
  Coefficient of variation of demand aggregated into time buckets:
    bucket demand by week/month -> totals per bucket -> std/mean

lead_time_mean, lead_time_cv
  Distribution of replenishment lead times (PO create -> GR):
    for each PO item: GR.timestamp - POCreate.timestamp
    lead_time_mean = mean(lead_times), lead_time_cv = std/mean

eoq_mean, eoq_cv
  Distribution of order quantities at PO creation:
    activity == CreatePurchaseOrderItem, use PO quantity
    eoq_mean = mean(qty), eoq_cv = std/mean(qty)

interorder_time_mean_days
  Mean time between consecutive PO creation events:
    mean(diff(sorted(POCreate.timestamps))) in days

rop_mean, rop_cv
  Proxy for reorder point as stock level observed at PO creation times:
    take stock_level snapshots at POCreate timestamps
    rop_mean = mean(stock_at_order), rop_cv = std/mean(stock_at_order)

under_time_pct, over_time_pct, normal_time_pct
  Time-weighted exposure to each state using the stock-level time series:
    Integrate a step function stock(t) between events (timestamp deltas)

  A threshold-based labeling consistent with data-driven ROP/EOQ inference:
    rop_std ≈ rop_cv * rop_mean
    eoq_std ≈ eoq_cv * eoq_mean
    Under if stock(t) < rop_mean - k*rop_std
    Over  if stock(t) > rop_mean + eoq_mean + k*eoq_std
    Normal otherwise
  Then:
    under_time_pct = 100 * (time_under / total_time), etc.

under_logodds_vs_normal, over_logodds_vs_normal
  Log-odds relative to normal exposure:
    log(under_prop / normal_prop), log(over_prop / normal_prop)
  These are still observable: computed directly from the three time shares.
  (We use them because they are unbounded and pair well with linear SEM.)
-----------------------------------------------------------------------

Install:
  pip install semopy pandas numpy

Run:
  python simulate_ocel_inventory_sem.py --n 2000 --out simulated_ocel_sem.csv --fit
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# -----------------------------
# SEM model (latent + observed)
# -----------------------------
# NOTE: No comments inside the model string to avoid parser issues.
SEM_MODEL_DESC = r"""
Demand =~ 1*log_demand_total + log_demand_cv
Supply =~ 1*log_lead_time_mean + log_lead_time_cv
Batchiness =~ 1*log_eoq_mean + log_eoq_cv + log_interorder_time_days
Buffering =~ 1*log_rop_mean + log_rop_cv

Batchiness ~ Demand + Supply
Buffering ~ Demand + Supply + Batchiness

under_logodds_vs_normal ~ Demand + Supply + Batchiness + Buffering
over_logodds_vs_normal  ~ Demand + Supply + Batchiness + Buffering

Demand ~~ Supply
under_logodds_vs_normal ~~ over_logodds_vs_normal
""".strip()


@dataclass(frozen=True)
class SimConfig:
    n: int = 2000
    seed: int = 7
    rho_demand_supply: float = 0.30  # correlation between exogenous latent factors
    k_threshold: float = 1.0         # used in the (event-log-computable) under/over rule
    horizon_days: float = 180.0      # reference window for rates/counts


def _softmax3(u: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stable 3-class softmax for [Under, Over, Normal] using:
      normal logit fixed to 0
    Returns: under_prop, over_prop, normal_prop, all strictly in (0, 1) and sum to 1.
    """
    z0 = u
    z1 = o
    z2 = np.zeros_like(u)

    m = np.maximum(np.maximum(z0, z1), z2)
    e0 = np.exp(z0 - m)
    e1 = np.exp(z1 - m)
    e2 = np.exp(z2 - m)
    denom = e0 + e1 + e2
    return e0 / denom, e1 / denom, e2 / denom


def simulate_ocel_sem_dataset(cfg: SimConfig) -> pd.DataFrame:
    """
    Simulate observable event-log-derived variables from a latent-factor SCM
    consistent with the SEM model definition above.

    Latent random variables:
      (Demand, Supply) ~ MVN(0, Σ) with corr = rho_demand_supply
      Batchiness, Buffering: linear structural equations + Gaussian noise
      under/over log-odds (vs normal): linear structural equations + Gaussian noise

    Measurement model:
      Observed log-metrics are linear in their corresponding latent factor + Gaussian noise.
      Raw metrics are exp(log-metric) (lognormal, positive), matching typical OCEL-derived metrics.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    # ---- Exogenous latent factors: Demand, Supply ----
    cov = np.array([[1.0, cfg.rho_demand_supply],
                    [cfg.rho_demand_supply, 1.0]])
    ds = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    Demand = ds[:, 0]
    Supply = ds[:, 1]

    # ---- Structural latent variables ----
    # Batchiness: captures "large & variable order quantities + longer order cycles"
    Batchiness = 0.70 * Demand + 0.60 * Supply + rng.normal(0.0, 0.60, size=n)

    # Buffering: proxy for reorder-point / safety-stock regime
    Buffering = 0.50 * Demand + 0.80 * Supply + 0.50 * Batchiness + rng.normal(0.0, 0.60, size=n)

    # ---- Outcomes: log-odds of Under/Over vs Normal exposure ----
    # Signs encode causal intuition:
    #   - Demand and Supply uncertainty increase understock pressure
    #   - Batchiness increases both tails (under & over)
    #   - Buffering reduces Under but increases Over
    under_true = (
        -0.30
        + 0.80 * Demand
        + 0.70 * Supply
        + 0.50 * Batchiness
        - 0.80 * Buffering
        + rng.normal(0.0, 0.70, size=n)
    )
    over_true = (
        -0.60
        - 0.20 * Demand
        + 0.40 * Supply
        + 0.60 * Batchiness
        + 0.60 * Buffering
        + rng.normal(0.0, 0.70, size=n)
    )

    # Convert the two logits into 3 proportions (Under/Over/Normal) that sum to 1
    under_prop, over_prop, normal_prop = _softmax3(under_true, over_true)

    # These are the observable exposure shares (e.g., time-in-state %)
    under_time_pct = 100.0 * under_prop
    over_time_pct = 100.0 * over_prop
    normal_time_pct = 100.0 * normal_prop

    # Observable log-odds vs normal (computable from the 3 shares)
    eps = 1e-9
    under_logodds_vs_normal = np.log((under_prop + eps) / (normal_prop + eps))
    over_logodds_vs_normal = np.log((over_prop + eps) / (normal_prop + eps))

    # ---- Measurement model: log-observed indicators ----
    # Intercepts set realistic magnitudes for raw metrics after exp().
    # All of these are still event-log computable; we model their logs for SEM stability.

    # Demand indicators
    log_demand_total = 6.0 + 1.00 * Demand + rng.normal(0.0, 0.25, size=n)   # exp(6) ~ 403
    log_demand_cv = -0.5 + 0.70 * Demand + rng.normal(0.0, 0.30, size=n)     # exp(-0.5) ~ 0.61

    # Supply indicators
    log_lead_time_mean = 2.0 + 1.00 * Supply + rng.normal(0.0, 0.25, size=n)  # exp(2) ~ 7.4 days
    log_lead_time_cv = -1.2 + 0.70 * Supply + rng.normal(0.0, 0.25, size=n)   # exp(-1.2) ~ 0.30

    # Batchiness indicators
    log_eoq_mean = 3.7 + 1.00 * Batchiness + rng.normal(0.0, 0.25, size=n)     # exp(3.7) ~ 40
    log_eoq_cv = -0.9 + 0.60 * Batchiness + rng.normal(0.0, 0.30, size=n)      # exp(-0.9) ~ 0.41
    log_interorder_time_days = 2.5 + 0.70 * Batchiness + rng.normal(0.0, 0.25, size=n)  # exp(2.5) ~ 12.2 days

    # Buffering indicators
    log_rop_mean = 5.1 + 1.00 * Buffering + rng.normal(0.0, 0.25, size=n)      # exp(5.1) ~ 164
    log_rop_cv = -1.6 + 0.70 * Buffering + rng.normal(0.0, 0.25, size=n)       # exp(-1.6) ~ 0.20

    # ---- Convert log-metrics back to raw metrics (also event-log computable) ----
    demand_total = np.exp(log_demand_total)
    demand_cv = np.exp(log_demand_cv)

    lead_time_mean_days = np.exp(log_lead_time_mean)
    lead_time_cv = np.exp(log_lead_time_cv)

    eoq_mean = np.exp(log_eoq_mean)
    eoq_cv = np.exp(log_eoq_cv)

    interorder_time_mean_days = np.exp(log_interorder_time_days)

    rop_mean = np.exp(log_rop_mean)
    rop_cv = np.exp(log_rop_cv)

    # Optional integer counts that are also OCEL-computable (not used in SEM):
    # number_poi: approximate PO count from inter-order time and horizon.
    expected_poi = np.maximum(cfg.horizon_days / np.maximum(interorder_time_mean_days, 1e-6), 0.1)
    number_poi = rng.poisson(lam=expected_poi)

    df = pd.DataFrame(
        {
            # ---- Raw, event-log-computable metrics ----
            "demand_total": demand_total,
            "demand_cv": demand_cv,
            "lead_time_mean_days": lead_time_mean_days,
            "lead_time_cv": lead_time_cv,
            "eoq_mean": eoq_mean,
            "eoq_cv": eoq_cv,
            "interorder_time_mean_days": interorder_time_mean_days,
            "rop_mean": rop_mean,
            "rop_cv": rop_cv,
            "number_poi": number_poi,

            # ---- Exposure outcomes (event-log time-in-state) ----
            "under_time_pct": under_time_pct,
            "over_time_pct": over_time_pct,
            "normal_time_pct": normal_time_pct,

            # ---- Log-odds outcomes used in SEM (derived from shares) ----
            "under_logodds_vs_normal": under_logodds_vs_normal,
            "over_logodds_vs_normal": over_logodds_vs_normal,

            # ---- Logged indicators used in SEM (directly derived from the raw metrics) ----
            "log_demand_total": log_demand_total,
            "log_demand_cv": log_demand_cv,
            "log_lead_time_mean": log_lead_time_mean,
            "log_lead_time_cv": log_lead_time_cv,
            "log_eoq_mean": log_eoq_mean,
            "log_eoq_cv": log_eoq_cv,
            "log_interorder_time_days": log_interorder_time_days,
            "log_rop_mean": log_rop_mean,
            "log_rop_cv": log_rop_cv,
        }
    )
    return df


def fit_semopy_model(df: pd.DataFrame) -> None:
    """
    Fit the SEM model using semopy to the simulated dataset.
    (This is optional; simulation + CSV export is the core deliverable.)
    """
    try:
        from semopy import Model, calc_stats  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "semopy is required for fitting. Install with: pip install semopy"
        ) from e

    # Use only variables referenced by the SEM model
    sem_cols = [
        "log_demand_total",
        "log_demand_cv",
        "log_lead_time_mean",
        "log_lead_time_cv",
        "log_eoq_mean",
        "log_eoq_cv",
        "log_interorder_time_days",
        "log_rop_mean",
        "log_rop_cv",
        "under_logodds_vs_normal",
        "over_logodds_vs_normal",
    ]
    data = df[sem_cols].copy()

    # (Optional) mean-center for numerical stability (SEM is usually covariance-based)
    data = data - data.mean(axis=0)

    model = Model(SEM_MODEL_DESC)
    model.fit(data)

    print("\n=== SEM parameter estimates (head) ===")
    est = model.inspect()
    print(est.head(30).to_string(index=False))

    print("\n=== Global fit statistics ===")
    stats = calc_stats(model)
    print(stats.T.to_string())

    from semopy import semplot
    semplot(
        model,
        filename="sem_inventory_model_fitted.pdf",
        plot_covs=True,
        plot_exos=True,
        std_ests=True,  # <- standardized coefficients
        show=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000, help="Number of product-location objects to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--rho", type=float, default=0.30, help="Correlation between latent Demand and Supply.")
    parser.add_argument("--horizon-days", type=float, default=180.0, help="Reference observation window length in days.")
    parser.add_argument("--out", type=str, default="simulated_ocel_sem_inventory.csv", help="Output CSV path.")
    parser.add_argument("--fit", action="store_true", help="Also fit the SEM using semopy and print results.")
    args = parser.parse_args()

    cfg = SimConfig(
        n=args.n,
        seed=args.seed,
        rho_demand_supply=args.rho,
        horizon_days=args.horizon_days,
    )

    df = simulate_ocel_sem_dataset(cfg)
    df.to_csv(args.out, index=False)
    print(f"Saved simulated dataset: {args.out}")
    print(f"Rows: {len(df):,d} | Columns: {len(df.columns):,d}")

    if args.fit:
        fit_semopy_model(df)


if __name__ == "__main__":
    main()
