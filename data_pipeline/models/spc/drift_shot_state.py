"""
Drift detection on every shot after approval, per job.

  EWMA (lambda 0.2) on the post-gate pack and fill integral deviations
  CUSUM on end-of-fill pressure (vents, lower side), on pack-integral variance over
  rolling 200-shot windows (check ring, upper side), and on gate seal time (mold
  temperature, both sides); tuned to 0.75 sigma shifts on 25-shot block means,
  which absorbs the shot-to-shot autocorrelation.

The baseline for each job is its first 500 shots after approval. Statistics reset
at approval, at documented drift-correction changes, at vent cleaning, PM and ring
replacement, and restart at a resin lot change.
"""
import numpy as np
import pandas as pd

LAMBDA = 0.2
BLOCK = 25
K = 0.375
H = 5.0
BASE = 500


def _cusum(x, k, h, side, resets):
    s, out, sig = 0.0, np.zeros(len(x)), np.zeros(len(x), bool)
    for i, v in enumerate(x):
        if resets[i]:
            s = 0.0
        if np.isnan(v):
            out[i] = s
            continue
        s = max(0.0, s + (v - k if side > 0 else -v - k))
        out[i] = s
        if s > h:
            sig[i] = True
            s = 0.0
    return out, sig


def _job(g):
    g = g.sort_values("shot_ts").reset_index(drop=True)
    n = len(g)
    res = pd.DataFrame({"shot_id": g["shot_id"]})
    base = g.iloc[:BASE]
    # reset points: drift corrections, vent cleaning, PM, ring replacement, lot changes
    keys = g[["last_drift_correction_ts", "shots_since_vent_cleaning", "shots_since_pm",
              "days_since_ring_replace", "lot_change_ts"]]
    reset = np.zeros(n, bool)
    reset[1:] |= (g["last_drift_correction_ts"].astype("int64").diff().fillna(0).to_numpy()[1:] != 0)
    for c in ("shots_since_vent_cleaning", "shots_since_pm"):
        v = g[c].to_numpy(float)
        reset[1:] |= np.nan_to_num(np.diff(v), nan=1) < 0
    reset[1:] |= np.nan_to_num(np.diff(g["days_since_ring_replace"].to_numpy(float)), nan=0) < -0.5
    reset[1:] |= (g["lot_change_ts"].astype("int64").diff().fillna(0).to_numpy()[1:] != 0)
    res["drift_reset"] = reset

    for col, name in (("pg_pack_dev", "pack"), ("pg_fill_dev", "fill")):
        x = g[col].ffill().fillna(0).to_numpy()
        e = np.zeros(n)
        prev = np.nanmean(base[col]) if len(base) else 0.0
        for i in range(n):
            if reset[i]:
                prev = x[i]
            prev = LAMBDA * x[i] + (1 - LAMBDA) * prev
            e[i] = prev
        mu = np.nanmean(e[:BASE])
        sd = np.nanstd(e[:BASE]) or 1.0
        res[f"ewma_{name}"] = e
        res[f"ewma_{name}_signal"] = np.abs(e - mu) > 3.5 * sd

    def blocks(col):
        v = g[col].to_numpy(float)
        bm = pd.Series(v).rolling(BLOCK, min_periods=BLOCK // 2).mean().to_numpy()
        ref = bm[BLOCK:BASE:BLOCK]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 4:
            return np.full(n, np.nan)
        z = (bm - ref.mean()) / (ref.std() or 1.0)
        z[np.arange(n) % BLOCK != BLOCK - 1] = np.nan       # one reading per block
        return z

    ze = blocks("eof_pressure_dev") if g["eof_pressure_dev"].notna().any() else np.full(n, np.nan)
    c, s = _cusum(ze, K, H, -1, reset)
    res["cusum_eof"], res["cusum_eof_signal"] = c, s
    zg = blocks("pg_gate_seal_dev")
    cu, su = _cusum(zg, K, H, 1, reset)
    cl, sl = _cusum(zg, K, H, -1, reset)
    res["cusum_gate_seal"] = np.maximum(cu, cl)
    res["cusum_gate_seal_signal"] = su | sl
    # pack-integral variance: rolling 200-shot sd against the baseline sd, log scale
    v = pd.Series(g["pg_pack_dev"].to_numpy(float)).rolling(200, min_periods=100).std().to_numpy()
    b = np.nanmedian(v[100:BASE]) if np.isfinite(v[100:BASE]).any() else np.nan
    lr = np.log(v / b)
    lr[np.arange(n) % 50 != 49] = np.nan
    cv, sv = _cusum(lr, 0.10, 1.2, 1, reset)
    res["cusum_pack_var"], res["cusum_pack_var_signal"] = cv, sv
    res["drift_any_signal"] = (res["ewma_pack_signal"] | res["ewma_fill_signal"] | res["cusum_eof_signal"]
                               | res["cusum_gate_seal_signal"] | res["cusum_pack_var_signal"])
    return res


def model(dbt, session):
    dbt.config(materialized="table")
    s = dbt.ref("int_shot_summary").df()[["shot_id", "pg_pack_dev", "pg_fill_dev", "eof_pressure_dev", "pg_gate_seal_dev"]]
    c = dbt.ref("int_shot_context").df()[["shot_id", "job_id", "shot_ts", "after_approval", "last_drift_correction_ts",
                                          "shots_since_vent_cleaning", "shots_since_pm", "days_since_ring_replace",
                                          "lot_change_ts"]]
    d = c[c["after_approval"]].merge(s, on="shot_id")
    for col in ("last_drift_correction_ts", "lot_change_ts"):
        d[col] = pd.to_datetime(d[col]).fillna(pd.Timestamp("2000-01-01"))
    out = [_job(g) for _, g in d.groupby("job_id", sort=False)]
    return pd.concat(out, ignore_index=True)
