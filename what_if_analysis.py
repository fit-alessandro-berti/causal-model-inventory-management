#!/usr/bin/env python3
"""
what_if_analysis.py

Compute "what-if" effects on overstock log-odds from doubling observables,
based on fitted SEM parameters exported by estimation.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from estimation import SEM_MODEL_DESC  # noqa: F401


LOG_EPS = 1e-9

INDICATOR_TO_FACTOR = {
    "log_demand_total": "Demand",
    "log_demand_cv": "Demand",
    "log_lead_time_mean": "Supply",
    "log_lead_time_cv": "Supply",
    "log_eoq_mean": "Batchiness",
    "log_eoq_cv": "Batchiness",
    "log_interorder_time_days": "Batchiness",
    "log_rop_mean": "Buffering",
    "log_rop_cv": "Buffering",
}

RAW_TO_LOG = {
    "demand_total": "log_demand_total",
    "demand_cv": "log_demand_cv",
    "lead_time_mean_days": "log_lead_time_mean",
    "lead_time_cv": "log_lead_time_cv",
    "eoq_mean": "log_eoq_mean",
    "eoq_cv": "log_eoq_cv",
    "interorder_time_mean_days": "log_interorder_time_days",
    "rop_mean": "log_rop_mean",
    "rop_cv": "log_rop_cv",
}

ANCHOR_LOADINGS = {
    "log_demand_total": 1.0,
    "log_lead_time_mean": 1.0,
    "log_eoq_mean": 1.0,
    "log_rop_mean": 1.0,
}


def _col_name(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing columns: {candidates}")


def _get_param(df: pd.DataFrame, lhs: str, op: str, rhs: str) -> float | None:
    lcol = _col_name(df, ["lval", "lhs"])
    ocol = _col_name(df, ["op", "operator"])
    rcol = _col_name(df, ["rval", "rhs"])
    ecol = _col_name(df, ["Estimate", "estimate", "est"])
    match = df[(df[lcol] == lhs) & (df[ocol] == op) & (df[rcol] == rhs)]
    if match.empty:
        return None
    return float(match[ecol].iloc[0])


def _get_loading(df: pd.DataFrame, indicator: str, factor: str) -> float | None:
    candidates = [
        (indicator, "~", factor),
        (factor, "=~", indicator),
        (indicator, "=~", factor),
        (factor, "~", indicator),
    ]
    for lhs, op, rhs in candidates:
        val = _get_param(df, lhs, op, rhs)
        if val is not None:
            return val
    return None


def _safe_log_ratio(multiplier: float) -> float:
    return math.log(max(multiplier, LOG_EPS))


def _compute_total_effects(params: pd.DataFrame) -> dict[str, float]:
    b_batch_d = _get_param(params, "Batchiness", "~", "Demand") or 0.0
    b_batch_s = _get_param(params, "Batchiness", "~", "Supply") or 0.0
    b_buff_d = _get_param(params, "Buffering", "~", "Demand") or 0.0
    b_buff_s = _get_param(params, "Buffering", "~", "Supply") or 0.0
    b_buff_batch = _get_param(params, "Buffering", "~", "Batchiness") or 0.0

    b_over_d = _get_param(params, "over_logodds_vs_normal", "~", "Demand") or 0.0
    b_over_s = _get_param(params, "over_logodds_vs_normal", "~", "Supply") or 0.0
    b_over_batch = _get_param(params, "over_logodds_vs_normal", "~", "Batchiness") or 0.0
    b_over_buff = _get_param(params, "over_logodds_vs_normal", "~", "Buffering") or 0.0

    total_d = b_over_d + b_over_batch * b_batch_d + b_over_buff * (b_buff_d + b_buff_batch * b_batch_d)
    total_s = b_over_s + b_over_batch * b_batch_s + b_over_buff * (b_buff_s + b_buff_batch * b_batch_s)
    total_batch = b_over_batch + b_over_buff * b_buff_batch
    total_buff = b_over_buff

    return {
        "Demand": total_d,
        "Supply": total_s,
        "Batchiness": total_batch,
        "Buffering": total_buff,
    }


def run_what_if(params_path: Path, multiplier: float) -> pd.DataFrame:
    params = pd.read_csv(params_path)
    total_effects = _compute_total_effects(params)
    delta_log = _safe_log_ratio(multiplier)

    rows = []
    for raw_var, log_var in RAW_TO_LOG.items():
        factor = INDICATOR_TO_FACTOR[log_var]
        loading = ANCHOR_LOADINGS.get(log_var)
        if loading is None:
            loading = _get_loading(params, log_var, factor)

        if loading is None or loading == 0:
            delta_latent = np.nan
        else:
            delta_latent = delta_log / loading

        total_effect = total_effects[factor]
        delta_over = delta_latent * total_effect if not np.isnan(delta_latent) else np.nan

        rows.append(
            {
                "observable": raw_var,
                "log_indicator": log_var,
                "latent_factor": factor,
                "loading": loading,
                "delta_log_indicator": delta_log,
                "total_effect_on_over": total_effect,
                "delta_over_logodds": delta_over,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        default="output_data/sem_inventory_model_params.csv",
        help="CSV path with fitted SEM parameters (from estimation.py).",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=2.0,
        help="Multiply each observable by this factor (default: 2.0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output_data/what_if_overstock.csv",
        help="Output CSV path for what-if results.",
    )
    args = parser.parse_args()

    params_path = Path(args.params)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_what_if(params_path, multiplier=args.multiplier)
    results.to_csv(out_path, index=False)
    print(f"Saved what-if analysis: {out_path}")


if __name__ == "__main__":
    main()
