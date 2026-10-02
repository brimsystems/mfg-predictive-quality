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


def main(stages=("vm", "defect", "anomaly", "layers", "extras")):
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
    if "layers" in stages or "extras" in stages:
        an = pd.read_parquet(RESULTS / "anomaly_if_scores.parquet")
        an = an[an["seed"] == SEEDS[0]]
    if "extras" in stages:
        run_extras(df, an)
    if "layers" in stages:
        lay = layer_comparison(df, an)
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
    """Shots in the anomaly model's alarm state (the stored flag is already the two-in-ten alarm state)."""
    f = te[["shot_id", "job_id", "shot_ts"]].merge(flags, on="shot_id", how="left")
    return f[f["if_flag"].fillna(False).astype(bool)]


VM_FLAG = 0.8                                    # predicted dimension beyond 80% of tolerance: a layer alarm
VM_AUDIT = 0.75                                  # beyond 75%: an advisory extra audit, never a sort


def vm_shot_dimension(shots, model=None, meta=None):
    """Largest predicted |dimension deviation| over a shot's cavities, in tolerance units."""
    p = vm.predict_shots(shots, "dimension", model, meta)
    return p.assign(a=p["y"].abs()).groupby("shot_id")["a"].max()


def window_signals(df, an, start, end, vm_dim):
    """Each layer's flagged shots, one definition used by both the shot-level table and the alarm budget:
    template alarm on the shot; rule 1 or 2 on the shot (AR(1)-residual charts); a drift detector signaling on the
    shot; the anomaly model's alarm state; the virtual metrology advisory (predicted dimension past 75% of tolerance)."""
    te = df[(df["shot_ts"] >= start) & (df["shot_ts"] < end) & df["after_approval"]].copy()
    te["vm_dim"] = te["shot_id"].map(vm_dim)
    an_al = anomaly_alarms(te, an[["shot_id", "if_flag"]])
    sig = {"template_sort": te[te["unit_alarm_state"] == "alarm"], "rules": te[te["spc_we1"] | te["spc_we2"]],
           "drift": te[te["drift_any_signal"].astype(bool)], "virtual_metrology": te[te["vm_dim"] > VM_AUDIT], "anomaly": an_al}
    return te, sig


def layer_comparison(df, an, start=None, end=None, vm_dim=None):
    start, end = start or VALID_END, end or pd.Timestamp("2026-04-01")
    te = df[(df["shot_ts"] >= start) & (df["shot_ts"] < end) & df["after_approval"]]
    if vm_dim is None:
        vm_dim = vm_shot_dimension(te)
    te, sig = window_signals(df, an, start, end, vm_dim)
    n_hours = te["shot_ts"].dt.floor("h").nunique()
    signals = [(k, layers.hours(v)) for k, v in sig.items()]
    from .features import connect
    con = connect()
    defects = con.execute(f"""select source, job_id, press_id, mold_id, hour_ts, defect_code, qty from fct_confirmed_defects
                              where source in ('sort', 'audit', 'tally') and job_id is not null
                              and hour_ts >= '{start}' and hour_ts < '{end}'""").df()
    con.close()
    d, share, dim_share = layers.compare(defects, signals)
    budget = {name: layers.alarms_per_shift(s_, n_hours) for name, s_ in signals}
    budget["added_layers"] = sum(budget[k] for k in ("rules", "drift", "virtual_metrology", "anomaly"))
    budget["technician_total"] = budget["added_layers"]
    budget["total_with_template"] = budget["added_layers"] + budget["template_sort"]
    audit_trig = te[te["vm_dim"] > VM_AUDIT]                     # the virtual metrology layer is its advisory
    budget["vm_advisory_audits"] = layers.alarms_per_shift(layers.hours(audit_trig), n_hours)
    by_mold = d.groupby(["mold_id", "layer"])["qty"].sum().unstack(fill_value=0)
    by_mold = by_mold.div(by_mold.sum(axis=1), axis=0)
    by_code = d.groupby(["defect_code", "layer"])["qty"].sum().unstack(fill_value=0)
    # shot-level catch: audit-found defects (pulled on a clock, linked to their shot), alarm on that shot
    con = connect()
    linked = con.execute(f"""select shot_id, defect_code, qty from fct_confirmed_defects
                             where source = 'audit' and shot_id is not null and hour_ts >= '{start}' and hour_ts < '{end}'""").df()
    con.close()
    lk = linked.merge(te[["shot_id", "unit_alarm_state", "spc_we1", "spc_we2", "drift_any_signal", "vm_dim"]], on="shot_id")
    an_ids = set(sig["anomaly"]["shot_id"])
    flags = {"template_sort": lk["unit_alarm_state"] == "alarm", "rules": lk["spc_we1"] | lk["spc_we2"],
             "drift": lk["drift_any_signal"].astype(bool), "anomaly": lk["shot_id"].isin(an_ids),
             "virtual_metrology": (lk["vm_dim"] > VM_AUDIT) & (lk["defect_code"] == "dimensional")}
    shot_catch = {k: float((v * lk["qty"]).sum() / lk["qty"].sum()) for k, v in flags.items()}
    return dict(share=share.to_dict(), dimensional_share=dim_share.to_dict(), alarms_per_shift=budget,
                by_mold=by_mold.round(4).to_dict(orient="index"), by_code=by_code.to_dict(orient="index"),
                n_defects=int(d["qty"].sum()), n_dimensional=int(d.loc[d["defect_code"] == "dimensional", "qty"].sum()),
                n_test_hours=int(n_hours), shot_level=shot_catch, n_shot_linked=int(lk["qty"].sum()),
                period=[str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())])


def run_extras(df, an):
    """Summer backtest of virtual metrology and its layer comparison; the audit-only supervised comparison."""
    out = {}
    summer = (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01"))
    bt = {}
    for kind in ("weight", "dimension"):
        rows = []
        for s_ in SEEDS[:3]:
            m_, model, meta = vm.backtest(kind, summer[0], summer[0], summer[1], seed=s_)
            rows.append(m_.assign(seed=s_))
            if kind == "dimension" and s_ == SEEDS[0]:
                sh = df[(df["shot_ts"] >= summer[0]) & (df["shot_ts"] < summer[1]) & df["after_approval"]]
                vm_dim = vm_shot_dimension(sh, model, meta)
        r = pd.concat(rows)
        bt[kind] = dict(ratio=float(r.groupby("seed")["rmse_over_gauge"].median().mean()),
                        r2=float(r.groupby("seed")["r2"].median().mean()))
    out["summer_backtest"] = bt
    out["summer_layers"] = layer_comparison(df, an, summer[0], summer[1], vm_dim)
    lab = defect.labeled(df)
    ao = defect.evaluate_audit_only(lab)
    out["audit_only"] = ao.mean(numeric_only=True).to_dict()
    out["audit_only_sd"] = ao.std(numeric_only=True).to_dict()
    (RESULTS / "overview_extras.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("vm", "defect", "anomaly", "layers", "extras"))
