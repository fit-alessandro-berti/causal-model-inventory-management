#!/usr/bin/env python3
"""
ocel_metrics.py

Compute OCEL-derived inventory metrics per ocel:type:MAT_PLA
from the post-processed OCEL inventory CSV.
"""

from __future__ import annotations

import argparse
import ast
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACTIVITY_COL = "ocel:activity"
TIMESTAMP_COL = "ocel:timestamp"
MAT_PLA_COL = "ocel:type:MAT_PLA"
STOCK_BEFORE_COL = "Stock Before"
STOCK_AFTER_COL = "Stock After"

UNDER_LABEL = "understock"
OVER_LABEL = "overstock"
NORMAL_LABEL = "normal"
LOG_EPS = 1e-9


def _parse_mat_pla(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    return text.strip("'\"")


def _stats(series: Iterable[float]) -> tuple[float, float]:
    values = pd.Series(series, dtype="float64").dropna()
    if values.empty:
        return (np.nan, np.nan)
    return (float(values.mean()), float(values.std(ddof=0)))


def _lead_times_days(po_times: pd.Series, gr_times: pd.Series) -> list[float]:
    po_list = [t for t in po_times if not pd.isna(t)]
    gr_list = [t for t in gr_times if not pd.isna(t)]
    if not po_list or not gr_list:
        return []

    po_list.sort()
    gr_list.sort()

    lead_times = []
    for po_time in po_list:
        idx = bisect_left(gr_list, po_time)
        if idx >= len(gr_list):
            continue
        delta_days = (gr_list[idx] - po_time).total_seconds() / 86400.0
        if delta_days >= 0:
            lead_times.append(delta_days)
    return lead_times


def _safe_cv(mean_val: float, std_val: float) -> float:
    if np.isnan(mean_val) or np.isnan(std_val) or mean_val == 0:
        return np.nan
    return float(std_val / mean_val)


def _safe_log(series: pd.Series) -> pd.Series:
    series = series.astype("float64")
    return np.log(series.clip(lower=LOG_EPS))


def _activity_startswith(series: pd.Series, prefix: str) -> pd.Series:
    return series.fillna("").astype(str).str.startswith(prefix)


def _compute_status_time_shares(group: pd.DataFrame, status_col: str) -> tuple[float, float, float]:
    if status_col not in group.columns:
        return (np.nan, np.nan, np.nan)

    g = group[[TIMESTAMP_COL, status_col]].dropna(subset=[TIMESTAMP_COL]).copy()
    if g.empty:
        return (np.nan, np.nan, np.nan)

    g[status_col] = g[status_col].astype(str).str.strip().str.lower()
    g = g.sort_values(TIMESTAMP_COL)
    g["next_time"] = g[TIMESTAMP_COL].shift(-1)
    g["delta_sec"] = (g["next_time"] - g[TIMESTAMP_COL]).dt.total_seconds()
    g = g[g["delta_sec"] > 0]
    if g.empty:
        return (np.nan, np.nan, np.nan)

    total_time = g["delta_sec"].sum()
    if total_time <= 0:
        return (np.nan, np.nan, np.nan)

    under_time = g.loc[g[status_col] == UNDER_LABEL, "delta_sec"].sum()
    over_time = g.loc[g[status_col] == OVER_LABEL, "delta_sec"].sum()
    normal_time = g.loc[g[status_col] == NORMAL_LABEL, "delta_sec"].sum()

    return (
        float(100.0 * under_time / total_time),
        float(100.0 * over_time / total_time),
        float(100.0 * normal_time / total_time),
    )


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[MAT_PLA_COL] = data[MAT_PLA_COL].map(_parse_mat_pla)
    data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL], errors="coerce")
    data[STOCK_BEFORE_COL] = pd.to_numeric(data[STOCK_BEFORE_COL], errors="coerce")
    data[STOCK_AFTER_COL] = pd.to_numeric(data[STOCK_AFTER_COL], errors="coerce")
    data = data.dropna(subset=[MAT_PLA_COL])

    status_col = "Current Status" if "Current Status" in data.columns else "Status"

    rows = []
    for mat_pla, group in data.groupby(MAT_PLA_COL):
        gi = group[_activity_startswith(group[ACTIVITY_COL], "Goods Issue")]
        demand_delta = (gi[STOCK_BEFORE_COL] - gi[STOCK_AFTER_COL]).dropna()
        demand_mean, demand_std = _stats(demand_delta)
        demand_total = demand_mean  # average per-GI change, used as demand_total output
        demand_cv = _safe_cv(demand_mean, demand_std)

        gr = group[_activity_startswith(group[ACTIVITY_COL], "Goods Receipt")]
        eoq_delta = (gr[STOCK_AFTER_COL] - gr[STOCK_BEFORE_COL]).dropna()
        eoq_mean, eoq_std = _stats(eoq_delta)
        eoq_cv = _safe_cv(eoq_mean, eoq_std)

        po = group[_activity_startswith(group[ACTIVITY_COL], "Create Purchase Order Item")]
        rop_values = po[STOCK_BEFORE_COL].dropna()
        rop_mean, rop_std = _stats(rop_values)
        rop_cv = _safe_cv(rop_mean, rop_std)

        lead_times = _lead_times_days(
            po[TIMESTAMP_COL].dropna(),
            gr[TIMESTAMP_COL].dropna(),
        )
        lead_time_mean, lead_time_std = _stats(lead_times)
        lead_time_cv = _safe_cv(lead_time_mean, lead_time_std)

        po_times = po[TIMESTAMP_COL].dropna().sort_values()
        if len(po_times) >= 2:
            interorder_deltas = po_times.diff().dropna().dt.total_seconds() / 86400.0
            interorder_time_mean_days = float(interorder_deltas.mean())
        else:
            interorder_time_mean_days = np.nan

        number_poi = int(po.shape[0])

        under_time_pct, over_time_pct, normal_time_pct = _compute_status_time_shares(
            group,
            status_col,
        )
        eps = LOG_EPS
        under_prop = under_time_pct / 100.0 if not np.isnan(under_time_pct) else np.nan
        over_prop = over_time_pct / 100.0 if not np.isnan(over_time_pct) else np.nan
        normal_prop = normal_time_pct / 100.0 if not np.isnan(normal_time_pct) else np.nan
        under_logodds_vs_normal = (
            np.log((under_prop + eps) / (normal_prop + eps))
            if not np.isnan(under_prop) and not np.isnan(normal_prop)
            else np.nan
        )
        over_logodds_vs_normal = (
            np.log((over_prop + eps) / (normal_prop + eps))
            if not np.isnan(over_prop) and not np.isnan(normal_prop)
            else np.nan
        )

        rows.append(
            {
                MAT_PLA_COL: mat_pla,
                "demand_total": demand_total,
                "demand_cv": demand_cv,
                "lead_time_mean_days": lead_time_mean,
                "lead_time_cv": lead_time_cv,
                "eoq_mean": eoq_mean,
                "eoq_cv": eoq_cv,
                "interorder_time_mean_days": interorder_time_mean_days,
                "rop_mean": rop_mean,
                "rop_cv": rop_cv,
                "number_poi": number_poi,
                "under_time_pct": under_time_pct,
                "over_time_pct": over_time_pct,
                "normal_time_pct": normal_time_pct,
                "under_logodds_vs_normal": under_logodds_vs_normal,
                "over_logodds_vs_normal": over_logodds_vs_normal,
            }
        )

    out = pd.DataFrame(rows).sort_values(MAT_PLA_COL)
    ordered_cols = [
        "demand_total",
        "demand_cv",
        "lead_time_mean_days",
        "lead_time_cv",
        "eoq_mean",
        "eoq_cv",
        "interorder_time_mean_days",
        "rop_mean",
        "rop_cv",
        "number_poi",
        "under_time_pct",
        "over_time_pct",
        "normal_time_pct",
        "under_logodds_vs_normal",
        "over_logodds_vs_normal",
        "log_demand_total",
        "log_demand_cv",
        "log_lead_time_mean",
        "log_lead_time_cv",
        "log_eoq_mean",
        "log_eoq_cv",
        "log_interorder_time_days",
        "log_rop_mean",
        "log_rop_cv",
    ]
    out[ordered_cols[:15]] = out[ordered_cols[:15]].fillna(0.0)
    out["log_demand_total"] = _safe_log(out["demand_total"])
    out["log_demand_cv"] = _safe_log(out["demand_cv"])
    out["log_lead_time_mean"] = _safe_log(out["lead_time_mean_days"])
    out["log_lead_time_cv"] = _safe_log(out["lead_time_cv"])
    out["log_eoq_mean"] = _safe_log(out["eoq_mean"])
    out["log_eoq_cv"] = _safe_log(out["eoq_cv"])
    out["log_interorder_time_days"] = _safe_log(out["interorder_time_mean_days"])
    out["log_rop_mean"] = _safe_log(out["rop_mean"])
    out["log_rop_cv"] = _safe_log(out["rop_cv"])
    out[ordered_cols[15:]] = out[ordered_cols[15:]].fillna(0.0)
    return out[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input OCEL CSV.")
    parser.add_argument("--out", type=str, default="output_data/ocel_metrics.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    metrics = compute_metrics(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
