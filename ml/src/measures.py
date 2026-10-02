"""
Per-layer measures at the deployed thresholds, May 2025 to March 2026 (the rolling-origin evaluation period).

  budget        each layer's flagged shots and alarm episodes per shift (a layer's flagged shots on a job, with gaps
                under 30 shots, are one investigation)
  shot_level    defective against good audited pieces flagged, by layer and defect group, with the incremental
                column (template, rules, drift, anomaly, virtual metrology in that order); warm months separately
  job_hour      the earlier job-hour coverage and its chance level, for the technical report's note on it
  anomaly       novel events in the period: flagged, flagged before the template, lead to the first response
  vm            error against gauge R&R, R2 and calibration by month and mold, the ablation without cavity pressure

Drift detectors are tested against their own mechanisms in drift_mechanisms.py.

Usage: python -m ml.src.measures
"""
import json

import numpy as np
import pandas as pd

from . import anomaly_eval
from .features import DATA_DIR, ROOT, connect, load_shots


RESULTS = DATA_DIR / "results"
REF = ROOT / "data_source" / "reference"
START, END = pd.Timestamp("2025-05-01"), pd.Timestamp("2026-04-01")
WARM = (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01"))
ORDER = ["template_sort", "rules", "drift", "anomaly", "virtual_metrology"]
GAP = 30
B = 400
GROUPS = {"Fill volume": ["short_shot", "flash"], "Packing and shrinkage": ["sink", "void", "dimensional", "warp"],
          "Flow front": ["weld_line", "burn"], "Gate and other": ["gate_vestige", "other"],
          "Material": ["splay", "black_specks", "contamination"]}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
VM_DEPLOYED = 0.75               # advisory audit when a cavity's predicted dimension passes 75% of tolerance (deployed)


def boot(flag, defect, audit, rng, key_mask):
    """Bootstrap over audits: rate among defective pieces, rate among good pieces, and the difference."""
    f = np.nan_to_num(flag.astype(float))
    dk = defect.astype(bool) & key_mask.astype(bool)
    gd = ~defect.astype(bool)
    d = pd.DataFrame({"a": audit, "fd": f * dk, "nd": dk.astype(float), "fg": f * gd, "ng": gd.astype(float)})
    A = d.groupby("a")[["fd", "nd", "fg", "ng"]].sum().to_numpy()
    tot = A.sum(0)
    rd, rg = tot[0] / max(tot[1], 1), tot[2] / max(tot[3], 1)
    idx = rng.integers(0, len(A), size=(B, len(A)))
    S = A[idx].sum(1)
    bd = S[:, 0] / np.maximum(S[:, 1], 1)
    bg = S[:, 2] / np.maximum(S[:, 3], 1)
    diff = bd - bg
    return dict(n_defective=int(tot[1]), n_good=int(tot[3]), rate_defective=float(rd), rate_good=float(rg), above_chance=float(rd - rg),
                ci_defective=[float(np.quantile(bd, .025)), float(np.quantile(bd, .975))],
                ci_good=[float(np.quantile(bg, .025)), float(np.quantile(bg, .975))],
                ci_above_chance=[float(np.quantile(diff, .025)), float(np.quantile(diff, .975))])


def episodes(shots, flag):
    """Alarm episodes: flagged shots on a job, a gap of 30 shots or more starting a new one."""
    p = shots.loc[flag, ["job_id", "pos"]]
    return int((p.groupby("job_id")["pos"].diff().fillna(1e9) >= GAP).sum())


def shot_flags(df):
    an = pd.read_parquet(RESULTS / "rolling_anomaly.parquet")
    vd = pd.read_parquet(RESULTS / "rolling_vm_shot.parquet", columns=["shot_id", "cavity_id", "dim_pred"])
    s = df[(df["shot_ts"] >= START) & (df["shot_ts"] < END) & df["after_approval"]].sort_values(["job_id", "shot_ts"]).copy()
    s["pos"] = s.groupby("job_id").cumcount()
    s["template_sort"] = s["unit_alarm_state"] == "alarm"
    s["rules"] = s["spc_we1"] | s["spc_we2"]
    s["drift"] = s["drift_any_signal"].astype(bool)
    s["anomaly"] = s["shot_id"].isin(set(an.loc[an["if_flag"], "shot_id"]))
    s["vm_dim"] = s["shot_id"].map(vd.assign(a=vd["dim_pred"].abs()).groupby("shot_id")["a"].max())
    s["virtual_metrology"] = s["vm_dim"] > VM_DEPLOYED
    return s, vd


def budget(s):
    shifts = s["shot_ts"].dt.floor("h").nunique() / 8
    out = {}
    for l in ORDER:
        out[l] = dict(shots_flagged=float(s[l].mean()), episodes_per_shift=episodes(s, s[l]) / shifts)
    out["added_layers_total"] = sum(out[l]["episodes_per_shift"] for l in ORDER[1:])
    out["shifts"] = shifts
    return out


def rules_at(s, c=1.0):
    """Episodes per shift rules 1 and 2 would raise at c times the textbook limits (c = 1: 3 and 2 sigma)."""
    con = connect()
    pts = con.execute(f"""select shot_id, job_id, sensor_id, metric, z from spc_chart_points where is_monitoring
                          and shot_ts >= '{START}' and shot_ts < '{END}' order by job_id, sensor_id, metric, shot_ts""").df()
    con.close()
    key = (pts["job_id"] + "|" + pts["sensor_id"] + "|" + pts["metric"]).to_numpy()
    z = pts["z"].fillna(0).to_numpy()
    w2 = np.zeros(len(z), bool)
    for side in (1, -1):
        r = pd.Series((side * z > 2 * c).astype(int)).groupby(key).transform(lambda x: x.rolling(3, min_periods=1).sum())
        w2 |= r.to_numpy() >= 2
    flagged = set(pts.loc[(np.abs(z) > 3 * c) | w2, "shot_id"])
    shifts = s["shot_ts"].dt.floor("h").nunique() / 8
    f = s["shot_id"].isin(flagged)
    return dict(limits=[3 * c, 2 * c], shots_flagged=float(f.mean()), episodes_per_shift=episodes(s, f) / shifts)


def shot_level(s, vd, rng):
    con = connect()
    ap = con.execute(f"""select audit_id, shot_id, cavity_id, defect_code, shot_ts from fct_audit_piece
                         where linked_to_shot and after_approval and shot_ts >= '{START}' and shot_ts < '{END}'""").df()
    con.close()
    ap = ap.merge(s[["shot_id", "template_sort", "rules", "drift", "anomaly"]], on="shot_id")
    pv = vd.set_index(["shot_id", "cavity_id"])["dim_pred"]
    ap["vm_dim"] = pv.reindex(pd.MultiIndex.from_arrays([ap["shot_id"], ap["cavity_id"]])).to_numpy()
    ap["virtual_metrology"] = ap["vm_dim"].abs() > VM_DEPLOYED          # the advisory on the piece's own cavity
    ap["defective"] = ap["defect_code"].notna()
    ap["group"] = ap["defect_code"].map(GROUP_OF)
    ap["ml_layers"] = ap["anomaly"] | ap["virtual_metrology"]
    ap["warm"] = (ap["shot_ts"] >= WARM[0]) & (ap["shot_ts"] < WARM[1])
    targets = [("All defects", None)] + [(g, g) for g in GROUPS] + [("Dimensional and warp", ["dimensional", "warp"])]

    def table(sub, flag):
        res = {}
        for name, g in targets:
            if g is None:
                mask = np.ones(len(sub), bool)
            elif isinstance(g, list):
                mask = sub["defect_code"].isin(g).to_numpy()
            else:
                mask = (sub["group"] == g).to_numpy()
            res[name] = boot(flag, sub["defective"].to_numpy(), sub["audit_id"].to_numpy(), rng, mask)
        return res

    out = {"n_audit_pieces": int(len(ap)), "n_defective": int(ap["defective"].sum()), "layer": {}, "incremental": {},
           "incremental_warm": {}}
    for l in ORDER + ["ml_layers"]:
        out["layer"][l] = table(ap, ap[l].to_numpy())
    earlier = np.zeros(len(ap), bool)
    for l in ORDER:
        inc = ap[l].to_numpy(bool) & ~earlier
        out["incremental"][l] = table(ap, inc)
        w = ap["warm"].to_numpy()
        out["incremental_warm"][l] = table(ap[w], inc[w])
        earlier |= ap[l].to_numpy(bool)
    # by individual code: the template, and any layer
    any_layer = np.zeros(len(ap), bool)
    for l in ORDER:
        any_layer |= ap[l].to_numpy(bool)
    out["by_code"] = {}
    for code in GROUP_OF:
        mask = (ap["defect_code"] == code).to_numpy()
        if mask.sum() == 0:
            continue
        t = boot(ap["template_sort"].to_numpy(), ap["defective"].to_numpy(), ap["audit_id"].to_numpy(), rng, mask)
        a_ = boot(any_layer, ap["defective"].to_numpy(), ap["audit_id"].to_numpy(), rng, mask)
        out["by_code"][code] = dict(group=GROUP_OF[code], n_defective=t["n_defective"], template=t, any_layer=a_)
    spc = (ap["template_sort"] | ap["rules"] | ap["drift"]).to_numpy()
    ml = ap["ml_layers"].to_numpy() & ~spc
    out["incremental"]["ml_layers"] = table(ap, ml)
    w = ap["warm"].to_numpy()
    out["incremental_warm"]["ml_layers"] = table(ap[w], ml[w])
    return out


def context_15_months(df, rng):
    """Template, rules and drift over all fifteen months of audits (January 2025 to March 2026), as context for the
    evaluation-period table; and the share of defective audited pieces that fall in January to April 2025."""
    con = connect()
    ap = con.execute("""select a.audit_id, a.defect_code, a.shot_ts, s.unit_alarm_state = 'alarm' as template_sort,
                               s.spc_we1 or s.spc_we2 as rules, s.drift_any_signal as drift
                        from fct_audit_piece a join fct_shot s using (shot_id) where a.linked_to_shot and a.after_approval""").df()
    con.close()
    ap["defective"] = ap["defect_code"].notna()
    ap["group"] = ap["defect_code"].map(GROUP_OF)
    out = {"n_audit_pieces": int(len(ap)), "n_defective": int(ap["defective"].sum()),
           "early_share_of_defective": float((ap.loc[ap["defective"], "shot_ts"] < START).mean()), "layer": {}}
    for l in ("template_sort", "rules", "drift"):
        out["layer"][l] = {}
        for name, g in [("All defects", None)] + [(g, g) for g in GROUPS]:
            mask = np.ones(len(ap), bool) if g is None else (ap["group"] == g).to_numpy()
            out["layer"][l][name] = boot(ap[l].fillna(False).to_numpy(), ap["defective"].to_numpy(), ap["audit_id"].to_numpy(), rng, mask)
    return out


def job_hour(s):
    """Job-hour coverage (the earlier measure) and its chance level at the deployed thresholds: the share of confirmed
    defects credited to a layer that alarmed on the job in the hour or shortly before, against the share of good audited
    pieces whose job-hour would be credited under the same rules."""
    from . import layers
    sig = [(l, layers.hours(s[s[l]])) for l in ("template_sort", "rules", "drift", "virtual_metrology", "anomaly")]
    con = connect()
    defects = con.execute(f"""select source, job_id, press_id, mold_id, hour_ts, defect_code, qty from fct_confirmed_defects
                              where source in ('sort', 'audit', 'tally') and job_id is not null
                              and hour_ts >= '{START}' and hour_ts < '{END}'""").df()
    good = con.execute(f"""select job_id, shot_ts from fct_audit_piece where linked_to_shot and after_approval and defect_code is null
                           and shot_ts >= '{START}' and shot_ts < '{END}'""").df()
    con.close()
    _, share, _ = layers.compare(defects, sig)
    good["hour"] = pd.to_datetime(good["shot_ts"]).dt.floor("h")
    hs = dict(sig)
    keys = list(zip(good["job_id"], good["hour"]))
    credited = [any((j, h - pd.Timedelta(hours=k)) in hs[l] for l in ("template_sort", "rules", "drift", "anomaly")
                    for k in range(layers.LOOKBACK[l] + 1)) for j, h in keys]
    return dict(detected=float(1 - share.get("none", 0)), chance=float(np.mean(credited)), share=share.to_dict(),
                n_defects=int(defects["qty"].sum()), n_good=int(len(good)))


def anomaly_events(df, s, bud):
    an = pd.read_parquet(RESULTS / "rolling_anomaly.parquet")
    flags = df["shot_id"].isin(set(an.loc[an["if_flag"], "shot_id"]))
    evr, _, _ = anomaly_eval.evaluate(df, flags)
    evr = evr[(evr["start"] >= START) & (evr["start"] < END)].copy()
    evr["before_template"] = evr["detected"] & (evr["minutes_to_template_alarm"].isna()
                                                | (evr["minutes_to_flag"] < evr["minutes_to_template_alarm"]))
    evr.to_csv(RESULTS / "measures_anomaly_events.csv", index=False)
    return dict(events=int(len(evr)), flagged=int(evr["detected"].sum()), before_template=int(evr["before_template"].sum()),
                median_lead_over_response_min=float(evr["lead_over_response_min"].median()),
                episodes_per_shift=bud["anomaly"]["episodes_per_shift"],
                rows=evr[["job_id", "kind", "mold_id", "press_id", "start", "shots", "detected", "minutes_to_flag",
                          "minutes_to_template_alarm", "lead_over_response_min", "before_template"]].to_dict(orient="records"))


def vm_summary():
    m = pd.read_csv(RESULTS / "rolling_vm_metrics.csv", parse_dates=["month"])
    m["mold_id"] = m["cell"].str.split(" / ").str[0]
    m["warm"] = (m["month"] >= WARM[0]) & (m["month"] < WARM[1])
    cols = ["rmse_over_gauge", "r2", "calibration_slope"]
    by_month = m.groupby(["target", "variant", "month"])[cols].median().reset_index()
    overall = m.groupby(["target", "variant"])[cols].median()
    warm = m.groupby(["target", "variant", "warm"])[cols].median()
    p = pd.read_parquet(RESULTS / "rolling_vm_predictions.parquet")
    p["mold_id"] = p["cell"].str.split(" / ").str[0]
    r2 = {}
    for (t, mold), g in p.groupby(["target", "mold_id"]):
        ss = ((g["measured"] - g["measured"].mean()) ** 2).sum()
        r2[f"{t}|{mold}"] = float(1 - ((g["measured"] - g["predicted"]) ** 2).sum() / ss) if ss > 0 else np.nan
    return dict(by_month=by_month.assign(month=by_month["month"].dt.strftime("%Y-%m")).to_dict(orient="records"),
                overall={f"{t}|{v}": r.to_dict() for (t, v), r in overall.iterrows()},
                warm_vs_other={f"{t}|{v}|{'warm' if w else 'other'}": r.to_dict() for (t, v, w), r in warm.iterrows()},
                r2_by_mold=r2)


def main():
    rng = np.random.default_rng(7)
    df = load_shots()
    s, vd = shot_flags(df)
    out = {"period": [str(START.date()), str(END.date())], "warm": [str(WARM[0].date()), str(WARM[1].date())]}
    out["budget"] = budget(s)
    out["budget"]["rules_at_textbook_limits"] = rules_at(s, 1.0)
    print("budget", json.dumps(out["budget"], default=float), flush=True)
    out["shot_level"] = shot_level(s, vd, rng)
    out["job_hour"] = job_hour(s)
    out["context_15_months"] = context_15_months(df, rng)
    out["anomaly"] = anomaly_events(df, s, out["budget"])
    out["vm"] = vm_summary()
    (RESULTS / "measures.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", RESULTS / "measures.json")


if __name__ == "__main__":
    main()
