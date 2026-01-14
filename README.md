# causal-model-inventory-management

Repository for simulating or extracting OCEL-derived inventory metrics, fitting a
structural equation model (SEM) for undesirable inventory states (under/overstock),
and running simple what-if analyses on the fitted model.

The flow is:
1) Simulate observables or extract them from an OCEL export.
2) Fit the SEM and save parameters + a visualization.
3) Run what-if analysis using the fitted parameters.


## 1) Simulate observables (optional)

Use this when you need a synthetic dataset with OCEL-like metrics.

```bash
python observables_simulation.py --n 2000 --out output_data/simulated_ocel_sem_inventory.csv
```

This creates a dataset with event-log computable observables plus log-transformed
indicators for SEM estimation.


## 2) Extract metrics from an OCEL export

The script `ocel_metrics.py` reads a CSV like `input_data/post_ocel_inventory_management.csv`
and produces the exact columns expected by the SEM. Activity matching uses prefix
checks (e.g., "Goods Issue (Understock)" still matches "Goods Issue").

```bash
python ocel_metrics.py --data input_data/post_ocel_inventory_management.csv --out output_data/ocel_metrics.csv
```

### How metrics are computed (per `ocel:type:MAT_PLA`)

- `demand_total`, `demand_cv`
  - From "Goods Issue" events: `Stock Before - Stock After` per event.
  - `demand_total` is the mean of those deltas; `demand_cv = std / mean`.
- `eoq_mean`, `eoq_cv`
  - From "Goods Receipt" events: `Stock After - Stock Before` per event.
- `rop_mean`, `rop_cv`
  - From "Create Purchase Order Item" events: `Stock Before` values.
- `lead_time_mean_days`, `lead_time_cv`
  - For each PO creation time, the time to the next "Goods Receipt".
- `interorder_time_mean_days`
  - Mean time between consecutive PO creation events.
- `number_poi`
  - Count of PO creation events.
- `under_time_pct`, `over_time_pct`, `normal_time_pct`
  - Time-weighted shares from the status column (`Current Status` or `Status`)
    using timestamp deltas between consecutive events.
- `under_logodds_vs_normal`, `over_logodds_vs_normal`
  - Log-odds vs normal from the three time shares.
- Log indicators:
  - `log_*` fields are `log(metric)` with a small epsilon for zeros.

Missing metrics are filled with 0; log values become `log(eps)` accordingly.

Output columns (ordered):
```
demand_total,demand_cv,lead_time_mean_days,lead_time_cv,eoq_mean,eoq_cv,
interorder_time_mean_days,rop_mean,rop_cv,number_poi,under_time_pct,over_time_pct,
normal_time_pct,under_logodds_vs_normal,over_logodds_vs_normal,log_demand_total,
log_demand_cv,log_lead_time_mean,log_lead_time_cv,log_eoq_mean,log_eoq_cv,
log_interorder_time_days,log_rop_mean,log_rop_cv
```


## 3) Estimate the SEM and save a visualization

`estimation.py` fits the SEM using `semopy`, saves a PDF diagram, and exports
parameter estimates.

```bash
python estimation.py --data output_data/ocel_metrics.csv \
  --out output_data/sem_inventory_model_fitted.pdf \
  --params-out output_data/sem_inventory_model_params.csv
```

Outputs:
- `sem_inventory_model_fitted.pdf`: SEM diagram with standardized coefficients.
- `sem_inventory_model_params.csv`: parameter estimates (`model.inspect()`).


## 4) What-if analysis (doubling observables)

`what_if_analysis.py` reads the parameters CSV and computes the change in
`over_logodds_vs_normal` when each observable is doubled. It uses the fitted
loadings plus the structural paths in the SEM to propagate effects.

```bash
python what_if_analysis.py --params output_data/sem_inventory_model_params.csv \
  --out output_data/what_if_overstock.csv
```

Output columns:
```
observable,log_indicator,latent_factor,loading,delta_log_indicator,
total_effect_on_over,delta_over_logodds
```

Interpretation:
- `delta_log_indicator` is `log(2)` for doubling.
- `total_effect_on_over` is the total path effect from the latent factor to
  `over_logodds_vs_normal`.
- `delta_over_logodds` is the implied change in overstock log-odds.


## Requirements

Core scripts use:
- Python 3.10+
- pandas, numpy
- semopy (only for estimation and plotting)
