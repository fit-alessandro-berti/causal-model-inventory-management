#!/usr/bin/env python3
"""
estimation.py

Fit the SEM to the simulated observables and save a visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def fit_semopy_model(
    df: pd.DataFrame,
    output_path: str,
    params_out: str | None = None,
    mean_center: bool = True,
) -> None:
    """
    Fit the SEM model using semopy to the simulated dataset and save a PDF plot.
    """
    try:
        from semopy import Model, calc_stats, semplot  # type: ignore
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

    if mean_center:
        # SEM is usually covariance-based; mean-centering helps stability.
        data = data - data.mean(axis=0)

    model = Model(SEM_MODEL_DESC)
    model.fit(data)

    est = model.inspect()
    if params_out:
        params_path = Path(params_out)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        est.to_csv(params_path, index=False)
        print(f"Saved SEM parameter estimates: {params_path}")

    print("\n=== SEM parameter estimates (head) ===")
    print(est.head(30).to_string(index=False))

    print("\n=== Global fit statistics ===")
    stats = calc_stats(model)
    print(stats.T.to_string())

    semplot(
        model,
        filename=output_path,
        plot_covs=True,
        plot_exos=True,
        std_ests=True,  # standardized coefficients
        show=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input CSV with simulated observables.")
    parser.add_argument("--out", type=str, default="output_data/sem_inventory_model_fitted.pdf", help="Output PDF path.")
    parser.add_argument(
        "--params-out",
        type=str,
        default="output_data/sem_inventory_model_params.csv",
        help="Output CSV path for fitted parameters.",
    )
    parser.add_argument("--no-mean-center", action="store_true", help="Skip mean-centering of indicators.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fit_semopy_model(
        df,
        output_path=args.out,
        params_out=args.params_out,
        mean_center=not args.no_mean_center,
    )


if __name__ == "__main__":
    main()
