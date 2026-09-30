"""
Runs every model and evaluation, logs each to MLflow, and writes ml/data/results/.

  virtual metrology    per mold x press and pooled, five seeds
  ablations            one sensor missing (post-gate only), no machine side, no cavity
                       signal, no history
  supervised defect    any code and per code, five seeds, against the template limits
                       and the control-chart rules at the template's alarm rate
  anomaly              isolation forest (five seeds) and the curve autoencoder
  layer comparison     job-hour grain on the test period, and the alarm budget

Usage: python -m ml.src.run_all
"""
import json
import time

import mlflow
import numpy as np
import pandas as pd

from . import anomaly, anomaly_eval, defect, layers, vm
from .features import ALL, CAVITY, DATA_DIR, HISTORY, MACHINE, POST_GATE_ONLY, ROOT, SEEDS, STATE, TRAIN_END, VALID_END, load_shots

RESULTS = DATA_DIR / "results"
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'ml' / 'mlruns.db').as_posix()}")
mlflow.set_experiment("molding-cell")

ABLATIONS = {
    "post_gate_only": [c for c in ALL if c not in CAVITY] + POST_GATE_ONLY,
    "no_machine_side": [c for c in ALL if c not in MACHINE],
    "no_cavity_signal": MACHINE + HISTORY,
    "no_history": CAVITY + MACHINE + STATE,
}


def log(name, params, metrics):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})


def main(stages=("vm", "defect", "anomaly", "layers")):
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    if "vm" in stages:
        run_vm()
    df = load_shots()
    if "defect" in stages:
        run_defect(df)
    if "anomaly" in stages:
        run_anomaly(df)
    if "layers" in stages:
        an = pd.read_parquet(RESULTS / "anomaly_if_scores.parquet")
        lay = layer_comparison(df, an[an["seed"] == SEEDS[0]])
        out["layers"] = lay
        (RESULTS / "layers.json").write_text(json.dumps(out, indent=1, default=float))
        log("layer_comparison", dict(grain="job_hour", period="test"), {f"share_{k}": v for k, v in lay["share"].items()})
    print(f"done in {time.time() - t0:,.0f} s", flush=True)


def run_vm():
    # ── Virtual metrology ──
    print("virtual metrology", flush=True)
    r = vm.run(tag="full")
    s = vm.summarize(r)
    for (target, design), g in s.groupby(["target", "design"]):
        log(f"vm_{target}_{design}", dict(target=target, design=design, seeds=len(SEEDS)),
            dict(rmse_over_gauge_median=g["rmse_over_gauge"].median(), r2_median=g["r2"].median()))
    abl = []
    for name, feats in ABLATIONS.items():
        print("  ablation", name)
        a = vm.run(feature_set=feats, designs=("pooled",), seeds=SEEDS[:3], tag=name, save_predictions=False)
        abl.append(a)
        for target, g in a.groupby("target"):
            log(f"vm_{target}_{name}", dict(target=target, ablation=name), dict(rmse_over_gauge_median=g["rmse_over_gauge"].median(),
                                                                                  r2_median=g["r2"].median()))
    pd.concat(abl).to_csv(RESULTS / "vm_ablations.csv", index=False)


def run_defect(df):
    # ── Supervised defect model ──
    print("supervised defect", flush=True)
    d = defect.run(df, tag="full")
    for t, g in d[d["usable"] == True].groupby("target"):
        log(f"defect_{t}", dict(target=t), dict(recall_model=g["recall_model"].mean(), recall_template=g["recall_template"].mean(),
                                                ap_model=g["ap_model"].mean(), ap_template=g["ap_template"].mean()))
    dab = []
    for name in ("no_machine_side", "no_cavity_signal"):
        dab.append(defect.run(df, feats=ABLATIONS[name], tag=name))
    pd.concat(dab).to_csv(RESULTS / "defect_ablations.csv", index=False)


def run_anomaly(df):
    # ── Anomaly ──
    print("anomaly", flush=True)
    an = []
    X_chg = anomaly.change_features(df)
    for s_ in SEEDS:
        sc, fl = anomaly.isolation_forest(df, seed=s_, X_all=X_chg)
        an.append(pd.DataFrame(dict(shot_id=df["shot_id"], seed=s_, if_score=sc, if_flag=fl)))
    an = pd.concat(an)
    an.to_parquet(RESULTS / "anomaly_if_scores.parquet", index=False)
    ae = anomaly.autoencoder(df, seed=SEEDS[0])
    ae.to_parquet(RESULTS / "anomaly_ae_scores.parquet", index=False)
    evr, onset, summ = anomaly_eval.evaluate(df, an[an["seed"] == SEEDS[0]].set_index(df.index)["if_flag"], ae)
    evr.to_csv(RESULTS / "anomaly_events.csv", index=False)
    onset.rename("flag_rate").to_frame().to_csv(RESULTS / "anomaly_im12_by_month.csv")
    (RESULTS / "anomaly_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log("anomaly_isolation_forest", dict(flag_rate=anomaly.FLAG_RATE), summ)


# ── Deployed thresholds for the layer comparison ──────────────────────────
RULES_DEPLOYED = ("WE1", "WE2")                  # beyond 3 sigma, and two of three beyond 2 sigma
ANOMALY_PERSIST = (2, 10)                        # two flags in ten consecutive shots


def anomaly_alarms(te, flags):
    f = te[["shot_id", "job_id", "shot_ts"]].merge(flags, on="shot_id", how="left").sort_values("shot_ts")
    f["if_flag"] = f["if_flag"].fillna(False).astype(bool)
    k, n = ANOMALY_PERSIST
    f["alarm"] = f.groupby("job_id")["if_flag"].transform(lambda x: x.rolling(n, min_periods=1).sum() >= k)
    return f[f["alarm"]]


def layer_comparison(df, an):
    from .features import connect
    te = df[(df["shot_ts"] >= VALID_END) & df["after_approval"]].copy()
    n_hours = te["shot_ts"].dt.floor("h").nunique()
    con = connect()
    rules = con.execute(f"""select job_id, shot_ts from spc_alarms where shot_ts >= '{VALID_END.date()}'
                            and rule in {RULES_DEPLOYED}""").df()
    drift = con.execute(f"select job_id, shot_ts from drift_signals where shot_ts >= '{VALID_END.date()}'").df()
    con.close()
    p = pd.read_parquet(RESULTS / "vm_test_predictions_full.parquet")
    p = p[(p["target"] == "dimension") & (p["design"] == "pooled")]
    vm_flag = p[(p["predicted"] - p["nominal"]).abs() > 0.8 * p["tolerance"]]
    vm_flag = vm_flag.merge(te[["shot_id", "job_id"]], on="shot_id")
    signals = [("template_sort", layers.hours(te[te["unit_alarm_state"] == "alarm"])),
               ("rules", layers.hours(rules)),
               ("drift", layers.hours(drift)),
               ("virtual_metrology", layers.hours(vm_flag)),
               ("anomaly", layers.hours(anomaly_alarms(te, an[["shot_id", "if_flag"]])))]
    defects = layers.confirmed(VALID_END.date())
    d, share, dim_share = layers.compare(defects, signals)
    budget = {name: layers.alarms_per_shift(sig, n_hours) for name, sig in signals}
    budget["technician_total"] = sum(budget[k] for k in ("rules", "drift", "anomaly"))
    by_mold = d.groupby(["mold_id", "layer"])["qty"].sum().unstack(fill_value=0)
    by_mold = by_mold.div(by_mold.sum(axis=1), axis=0)
    by_code = d.groupby(["defect_code", "layer"])["qty"].sum().unstack(fill_value=0)
    return dict(share=share.to_dict(), dimensional_share=dim_share.to_dict(), alarms_per_shift=budget,
                by_mold=by_mold.round(4).to_dict(orient="index"), by_code=by_code.to_dict(orient="index"),
                n_defects=int(d["qty"].sum()), n_dimensional=int(d.loc[d["defect_code"] == "dimensional", "qty"].sum()),
                n_test_hours=int(n_hours))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("vm", "defect", "anomaly", "layers"))
