"""
Anomaly detection: shots unlike any validated shot.

Primary: an isolation forest per mold x press on the normalized summary values
taken against the job's own recent shots (each value's change from the mean of the
300 shots before it), plus the match score and the spread between sensed cavities.
A job running steadily off its template is the template's and the charts' business;
the anomaly layer looks for a shot that suddenly stops looking like its run. Fitted
on validated training shots (after approval, no alarm, no linked defect) with the
threshold set so the alarm state covers 0.30% of the fit window.

Comparison: an autoencoder per mold x press on the retained curves (first
post-gate sensor, resampled to 120 points over the cycle and scaled to the job's
median peak after removing the sensor's baseline), with the same flag rate on its fit window.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

from .features import CAVITY, DATA_DIR, connect

FLAG_RATE = 0.003                # deployed: the alarm state covers 0.30% of validated shots, set for the alarm budget
FIT_CAP = 60000
CURVE_POINTS = 120


def validated(df):
    return (df["after_approval"] & (df["shots_since_approval"] >= 500) & (df["unit_alarm_state"] == "none")
            & (df["pieces_confirmed_defective"].fillna(0) == 0) & (df["audit_rejects"].fillna(0) == 0))


LEVEL = [c for c in CAVITY if c not in ("match_score_min", "pg_pack_dev_spread")]
RECENT = 300


def change_features(df):
    """Each cavity value's change from the mean of the job's previous 300 shots (as of the shot)."""
    d = df.sort_values("shot_ts")
    out = pd.DataFrame(index=d.index)
    for f in LEVEL:
        base = d.groupby("job_id")[f].transform(lambda x: x.shift(1).rolling(RECENT, min_periods=50).mean())
        out[f"{f}_chg"] = d[f] - base
    out["match_score_min"] = d["match_score_min"]
    out["pg_pack_dev_spread"] = d["pg_pack_dev_spread"]
    return out.reindex(df.index)


MONITOR_FROM = 500                # like the control charts, monitoring starts once setup has converged
PERSIST = (2, 10)                 # an alarm: two flags in the last ten shots


def alarm_state(flags, jobs):
    """Two flags in the last ten shots of the same job."""
    k, n = PERSIST
    return flags.groupby(jobs).transform(lambda x: x.rolling(n, min_periods=1).sum() >= k).astype(bool)


def isolation_forest(df, feats=None, seed=11, X_all=None):
    """Returns the score per shot and the alarm state per shot. The threshold is set so the alarm state (two flags in
    ten shots) covers FLAG_RATE of validated steady-state shots in the training window."""
    X_all = change_features(df) if X_all is None else X_all
    feats = list(X_all.columns)
    df = df.join(X_all, rsuffix="_x") if feats[0] not in df.columns else df
    score = pd.Series(np.nan, index=df.index)
    alarm = pd.Series(False, index=df.index)
    mon = df["after_approval"] & (df["shots_since_approval"] >= MONITOR_FROM)
    for cell, g in df[mon].sort_values("shot_ts").groupby("cell"):
        fit = g[(g["split"] == "train") & validated(g)]
        if len(fit) < 1000:
            continue
        fs = fit.sample(min(FIT_CAP, len(fit)), random_state=seed)
        m = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=2).fit(fs[feats].fillna(0).to_numpy())
        sc = pd.Series(-m.score_samples(g[feats].fillna(0).to_numpy()), index=g.index)
        score[g.index] = sc
        calib = (g["split"] == "train") & validated(g)
        lo, hi = float(sc[calib].quantile(0.90)), float(sc.max())
        for _ in range(30):                                  # bisection on the alarm-state rate
            t = (lo + hi) / 2
            rate = alarm_state(sc > t, g["job_id"])[calib].mean()
            lo, hi = (t, hi) if rate > FLAG_RATE else (lo, t)
        alarm[g.index] = alarm_state(sc > hi, g["job_id"])
    return score, alarm


def load_curves(shot_ids, mold_id):
    """Resampled post-gate curves (first post-gate sensor) for the given shots."""
    con = connect()
    ids = pd.DataFrame({"shot_id": np.asarray(shot_ids, dtype=np.int64)})
    con.register("ids", ids)
    q = f"""
        with s as (select min(sensor_id) as sensor_id from stg_monitoring__shot_summary
                   where mold_id = '{mold_id}' and sensor_position = 'post_gate')
        select c.shot_id, c.t_ms, c.pressure_bar
        from stg_monitoring__cavity_curves c join ids using (shot_id)
        where c.mold_id = '{mold_id}' and c.sensor_id = (select sensor_id from s)
    """
    raw = con.execute(q).df()
    con.close()
    if raw.empty:
        return pd.DataFrame()
    raw["bin"] = (raw["t_ms"] / (raw.groupby("shot_id")["t_ms"].transform("max") + 10) * CURVE_POINTS).astype(int)
    wide = raw.pivot_table(index="shot_id", columns="bin", values="pressure_bar", aggfunc="mean")
    return wide.reindex(columns=range(CURVE_POINTS)).ffill(axis=1).fillna(0)


def autoencoder(df, seed=11):
    """Reconstruction error on retained curves, per mold x press. Returns a frame of retained shots."""
    rows = []
    for cell, g in df[df["after_approval"] & df["curve_retained"]].groupby("cell"):
        fit_ids = g[(g["split"] == "train") & validated(g)]["shot_id"]
        if len(fit_ids) < 500:
            continue
        fit_ids = fit_ids.sample(min(6000, len(fit_ids)), random_state=seed)
        mold = g["mold_id"].iloc[0]
        C = load_curves(g["shot_id"], mold)
        if C.empty:
            continue
        # remove the sensor's baseline (its reading once the part has released), which drifts and resets at recalibration
        C = C.sub(C.iloc[:, -12:].median(axis=1), axis=0)
        peak = C.max(axis=1)
        job_peak = g.set_index("shot_id")["job_id"].reindex(C.index).map(peak.groupby(g.set_index("shot_id")["job_id"].reindex(C.index)).median())
        X = C.div(job_peak, axis=0).fillna(0)
        Xf = X.loc[X.index.intersection(fit_ids)]
        # curves are on one scale already (fraction of the job's median peak); standardizing each point
        # would magnify the steep rise and decay, where a small timing shift between jobs is normal
        ae = MLPRegressor(hidden_layer_sizes=(48, 12, 48), activation="relu", alpha=1e-4, max_iter=400,
                          early_stopping=True, random_state=seed).fit(Xf, Xf)
        err = np.mean((ae.predict(X) - X.to_numpy()) ** 2, axis=1)
        err_fit = pd.Series(err, index=X.index).loc[Xf.index]
        thr = float(np.quantile(err_fit, 1 - FLAG_RATE))
        rows.append(pd.DataFrame(dict(shot_id=X.index, cell=cell, ae_error=err, ae_flag=err > thr)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["shot_id", "cell", "ae_error", "ae_flag"])


def run(df, seed=11):
    score, flag = isolation_forest(df, seed=seed)
    out = pd.DataFrame(dict(shot_id=df["shot_id"], if_score=score, if_flag=flag))
    ae = autoencoder(df, seed=seed)
    out = out.merge(ae[["shot_id", "ae_error", "ae_flag"]], on="shot_id", how="left")
    d = DATA_DIR / "results"
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"anomaly_scores_seed{seed}.parquet", index=False)
    return out
