# data_pipeline

dbt on DuckDB. Sources read the raw extracts in place (Parquet for the per-shot
tables, CSV for the rest); the warehouse file `molding.duckdb` holds staging
views and the intermediate, SPC and mart tables.

```bash
cd data_pipeline
python -m dbt.cli.main build --profiles-dir .
```

Layers:

- `staging/`: one view per source table, typed and renamed; free-text codes normalized.
- `intermediate/`: shot alignment to jobs, loads, lots, dryers, changes and maintenance; summary values normalized to the template (deviation as a fraction of the alarm band) and machine values to the process window; the unit's alarm logic recomputed.
- `spc/`: individuals and moving-range charts with Western Electric rules 1, 2, 4 and 5; EWMA and CUSUM drift detection with resets; lot-change step tests.
- `marts/`: shot, audit piece, job-hour and job grain tables for the models, screens, dashboard and reports.
