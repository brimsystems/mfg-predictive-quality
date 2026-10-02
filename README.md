# Cavity Pressure Monitoring and Predictive Quality

**A data platform, SPC layer and three model components for an injection molder's instrumented cell: every shot retained, charted, measured and screened for the unexpected.**

An injection molder runs two presses, IM-11 and IM-12, with cavity pressure sensors in six molds: automotive connector housings, medical device housings, a trim clip and a fluid-handling manifold. Each press has a monitoring unit that compares every shot's curve to a template and sorts shots that leave their bands. The shop's quality program ran on top of that: first-shot approvals, hourly audits with control charts, a reject bin reviewed each shift and scrap tallied by code at packing.

What the shop lacked was the connection. The units kept per-shot values for about 60 days and curves for about 10, nothing joined them to the MES job, the resin lot, the setpoint log or the audits, and the control-chart rules ran on one audit sample an hour.

[![Monitoring unit screen with the predicted weight, dimension and anomaly state](docs/screenshots/unit_screen.png)](docs/reports/unit_screen.html)

> **[All deliverables &rarr;](docs/index.html)**

---

## What was built

1. **Data pipeline.** Fifteen months of the units' per-shot summaries (about 2.9 million sensor rows), a retained sample of curves at 100 readings per second (about 600 million rows), machine-side shot data from the presses, and the MES, ERP, QMS, materials, dryer and tool-room records, landed as Parquet and CSV and modeled in dbt on DuckDB. Every shot is aligned as of its timestamp to its job, template, resin lot, dryer, setpoints and maintenance counters.
2. **SPC layer.** The units' template alarm logic recomputed and tested against their recorded state; individuals and moving-range charts on every shot with the QMS's Western Electric rules; EWMA and CUSUM drift detection with resets at approvals, corrections, vent cleaning, PM, ring replacement and lot changes; lot-change step tests.
3. **Virtual metrology.** Audit part weight and critical dimension predicted from each shot's features, so every shot carries a measurement.
4. **Supervised defect prediction**, attempted against the template limits and reported as it landed.
5. **Anomaly detection** for shots unlike anything validated: an isolation forest on the summary values, with a curve autoencoder as a comparison.
6. **Deliverables.** The unit screen and shift chart, the quality dashboard, an ML overview for the quality manager, the technical report and the monitoring report.

## Results

Test period December 2025 to March 2026; models trained through September 2025 and validated on October and November; five seeds.

**Layer comparison.** Confirmed defective pieces (sort review, audits, packing tallies) credited at the job-hour grain to the first layer that alarmed on the job:

| Layer | Share of confirmed defects | Alarms per shift |
|---|---|---|
| Template alarms and sort | 57.3% | 2.21 alarm hours (sort is automatic) |
| Control-chart rules 1 and 2 on every shot | +8.3 points | 0.96 |
| Drift detection | +5.6 points | 0.60 |
| Virtual metrology (dimensional codes) | +0.0 points | 0.01 |
| Anomaly model | +3.0 points | 1.71 |
| Caught by none | 25.8% | |

The deployed rules, drift signals and anomaly model together raise 3.3 technician alarms per shift, inside the three to six one technician can investigate. Rules 4 and 5 are not deployed on every shot: cavity pressure values are autocorrelated from shot to shot, and rule 5 fires on 75% of points. Linkage coverage: 28% of confirmed defective pieces can be tied to a shot.

**Virtual metrology** is the model that earns its place, because its labels are abundant (186,769 audited pieces, 88% linked to their shot). Pooled model on the test period: part weight RMSE 1.36 times the gauge R&R (R² 0.63), critical dimension 1.76 times (R² 0.85). Without the cavity pressure values the dimension error rises to 3.4 times gauge.

**Supervised defect prediction** does not beat the template limits on the common codes. At the template's own alarm rate it catches 83.0% of confirmed defective shots against 80.3%. It ranks sink (average precision 0.79 against 0.11) and dimensional defects (0.30 against 0.07) better than the band, and the rare codes have too few labels for anything usable.

**Anomaly detection** flagged all six novel events in the fifteen months (a failing check ring, a failing heater zone, nozzle drool, a wrong material), at 0.5% of validated test shots; 81% of its flags fall on shots the template passed. The curve autoencoder did not separate the events from other retained curves.

The technical report lists every realism check on the extracts, including the 8 of 35 that fall outside their targets on this run, with their values.

---

## Repository

```
data_source/
  generate/            generators, configuration, realism checks (checks.py)
  raw/                 extracts, gitignored (Parquet for per-shot tables, CSV for the rest)
  reference/           quality engineering's root-cause register and shot state, gitignored
  samples/             200-row samples of every extract
data_pipeline/         dbt project: staging, intermediate, spc, marts; tests
ml/
  src/                 features, virtual metrology, supervised defect, anomaly, layers, run_all
  reports/             report generators, shared style and results loader, build script
  tests/               as-of tests on the feature pipeline
analytics/reports/     quality dashboard generator
docs/                  published pages and screenshots
```

## Running it

```bash
python -m data_source.generate.run_generator
```

```bash
python -m data_source.generate.checks
```

```bash
cd data_pipeline && python -m dbt.cli.main build --profiles-dir .
```

```bash
python -m ml.src.run_all
```

```bash
python -m ml.reports.build
```

Generation takes about 25 minutes and writes about 1.4 GB of Parquet; the dbt build takes under two minutes; the models about 20 minutes. MLflow runs are logged to `ml/mlruns.db`.
