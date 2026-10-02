"""
Rolling-origin evaluation for virtual metrology and anomaly detection.

Each month from May 2025 to March 2026 is a test month. The models are refitted at the start of the month on every
earlier month (virtual metrology holds out the last earlier month for early stopping) and applied to that month only,
so every prediction is out of sample and the test months include the summer of 2025.

  virtual metrology   pooled model, weight and dimension, with the ablation without cavity pressure values;
                      per-piece metrics by month, and the predicted dimension of every shot and cavity (the advisory)
  anomaly             isolation forest per mold x press, refitted monthly on validated earlier shots, its threshold
                      calibrated on those shots so the alarm state covers anomaly.FLAG_RATE of them

Usage: python -m ml.src.rolling [vm] [anomaly]
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import anomaly, vm
from .features import ALL, DATA_DIR, MACHINE, HISTORY, load_audit_pieces, load_shots

RESULTS = DATA_DIR / "results"
MONTHS = pd.date_range("2025-05-01", "2026-03-01", freq="MS")
SEED = 11
VARIANTS = {"full": ALL, "no_cavity_signal": MACHINE + HISTORY}


def _months_of(ts):
    return ts.dt.to_period("M").dt.to_timestamp()


def run_vm(shots):
    df = vm.prepare(load_audit_pieces())
    cats = list(df["cell"].astype("category").cat.categories)
    codes = dict(zip(cats, range(len(cats))))
    metrics, preds, shot_pred = [], [], []
    for m0 in MONTHS:
        m1 = m0 + pd.DateOffset(months=1)
        stop = m0 - pd.DateOffset(months=1)
        tr, va = df[df["shot_ts"] < stop], df[(df["shot_ts"] >= stop) & (df["shot_ts"] < m0)]
        te = df[(df["shot_ts"] >= m0) & (df["shot_ts"] < m1)]
        sh = shots[(shots["shot_ts"] >= m0) & (shots["shot_ts"] < m1) & shots["after_approval"]]
        for kind in vm.TARGETS:
            for var, feats in VARIANTS.items():
                f = feats + ["cell_code", "cavity_key"]
                model = vm._fit(tr, va, f, f"y_{kind}", SEED)
                p = vm._unscale(te, pd.Series(model.predict(te[f]), index=te.index), kind)
                metrics.append(vm.metrics(te, p, kind).assign(month=m0, target=kind, variant=var))
                if var == "full":
                    preds.append(pd.DataFrame(dict(audit_id=te["audit_id"], shot_id=te["shot_id"], cavity_id=te["cavity_id"],
                                                   cell=te["cell"], shot_ts=te["shot_ts"], month=m0, target=kind,
                                                   measured=te[vm.TARGETS[kind][0]], predicted=p,
                                                   nominal=te[vm.TARGETS[kind][1]], tolerance=te["dimension_tolerance_mm"])))
                    if kind == "dimension":
                        shot_pred.append(vm.predict_shots(sh, kind, model, dict(features=f, cell_codes=codes)))
        print(f"  vm {m0:%Y-%m}: {len(te):,} audit pieces, {len(sh):,} shots", flush=True)
    pd.concat(metrics).to_csv(RESULTS / "rolling_vm_metrics.csv", index=False)
    pd.concat(preds).to_parquet(RESULTS / "rolling_vm_predictions.parquet", index=False)
    pd.concat(shot_pred).rename(columns={"y": "dim_pred"}).to_parquet(RESULTS / "rolling_vm_shot_dimension.parquet", index=False)


def run_anomaly(df):
    X = anomaly.change_features(df)
    feats = list(X.columns)
    d = df.join(X[[c for c in feats if c not in df.columns]])          # match score and spread are already shot columns
    mon = d["after_approval"] & (d["shots_since_approval"] >= anomaly.MONITOR_FROM)
    d = d[mon].sort_values("shot_ts")
    d["month"] = _months_of(d["shot_ts"])
    d["valid_fit"] = anomaly.validated(d)
    score = pd.Series(np.nan, index=d.index)
    flag = pd.Series(False, index=d.index)
    thr_rows = []
    for cell, g in d.groupby("cell"):
        for m0 in MONTHS:
            past = g[(g["shot_ts"] < m0) & g["valid_fit"]]
            cur = g[g["month"] == m0]
            if len(past) < 1000 or cur.empty:
                continue
            fs = past.sample(min(anomaly.FIT_CAP, len(past)), random_state=SEED)
            m = IsolationForest(n_estimators=200, max_samples=4096, random_state=SEED, n_jobs=2).fit(fs[feats].fillna(0).to_numpy())
            sc_past = pd.Series(-m.score_samples(past[feats].fillna(0).to_numpy()), index=past.index)
            lo, hi = float(sc_past.quantile(0.90)), float(sc_past.max())
            for _ in range(30):                                  # bisection on the alarm-state rate over the fit shots
                t = (lo + hi) / 2
                rate = anomaly.alarm_state(sc_past > t, past["job_id"]).mean()
                lo, hi = (t, hi) if rate > anomaly.FLAG_RATE else (lo, t)
            sc = pd.Series(-m.score_samples(cur[feats].fillna(0).to_numpy()), index=cur.index)
            score[cur.index] = sc
            flag[cur.index] = sc > hi
            thr_rows.append(dict(cell=cell, month=m0, threshold=hi, n_fit=len(past)))
        print(f"  anomaly {cell}", flush=True)
    scored = score.notna()
    alarm = pd.Series(False, index=d.index)
    alarm[scored] = anomaly.alarm_state(flag[scored], d.loc[scored, "job_id"])
    out = pd.DataFrame(dict(shot_id=d["shot_id"], month=d["month"], if_score=score, if_raw=flag, if_flag=alarm))[scored]
    out.to_parquet(RESULTS / "rolling_anomaly.parquet", index=False)
    pd.DataFrame(thr_rows).to_csv(RESULTS / "rolling_anomaly_thresholds.csv", index=False)


def main(stages=("vm", "anomaly")):
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_shots()
    if "vm" in stages:
        print("rolling virtual metrology", flush=True)
        run_vm(df)
    if "anomaly" in stages:
        print("rolling anomaly", flush=True)
        run_anomaly(df)
    print(f"done in {time.time() - t0:,.0f} s", flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("vm", "anomaly"))
