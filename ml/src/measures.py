"""
Per-layer measures at the deployed thresholds, May 2025 to March 2026 (the rolling-origin evaluation period).

  budget        each layer's flagged shots and alarm episodes per shift (a layer's flagged shots on a job, with gaps
                under 30 shots, are one investigation)
  shot_level    defective against good audited pieces flagged, by layer and defect group, with the incremental
                column (template, rules, drift, anomaly, virtual metrology in that order); warm months separately
  drift_lead    for flow-front, fill-volume and packing-variability episodes: a firing of the relevant drift detector
                on that mold in the preceding window, and the lead in shots, against random production shots
  vent_check    the end-of-fill CUSUM (deployed) against the burn rise, per vent-cleaning interval
  anomaly       novel events in the period: flagged, flagged before the template, lead to the first response
  vm            error against gauge R&R, R2 and calibration by month and mold, the ablation without cavity pressure

Usage: python -m ml.src.measures
"""
import json

import numpy as np
import pandas as pd

from . import anomaly_eval
from .detection import GROUP_OF, GROUPS, boot
from .features import DATA_DIR, ROOT, connect, load_shots


RESULTS = DATA_DIR / "results"
REF = ROOT / "data_source" / "reference"
START, END = pd.Timestamp("2025-05-01"), pd.Timestamp("2026-04-01")
WARM = (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01"))
ORDER = ["template_sort", "rules", "drift", "anomaly", "virtual_metrology"]
GAP = 30
VM_DEPLOYED = 0.75               # advisory audit when a cavity's predicted dimension passes 75% of tolerance (deployed)
LEAD_WINDOW = 6000
DRIFT_FOR = {"Flow front": (["burn", "weld_line"], "cusum_eof"),
             "Fill volume": (["short_shot", "flash"], "ewma_fill"),
             "Packing variability": (["sink", "void"], "cusum_pack_var")}


def episodes(shots, flag):
    """Alarm episodes: flagged shots on a job, a gap of 30 shots or more starting a new one."""
    p = shots.loc[flag, ["job_id", "pos"]]
    return int((p.groupby("job_id")["pos"].diff().fillna(1e9) >= GAP).sum())


def shot_flags(df):
    an = pd.read_parquet(RESULTS / "rolling_anomaly.parquet")
    vd = pd.read_parquet(RESULTS / "rolling_vm_shot_dimension.parquet")
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
    spc = (ap["template_sort"] | ap["rules"] | ap["drift"]).to_numpy()
    ml = ap["ml_layers"].to_numpy() & ~spc
    out["incremental"]["ml_layers"] = table(ap, ml)
    w = ap["warm"].to_numpy()
    out["incremental_warm"]["ml_layers"] = table(ap[w], ml[w])
    return out


def mold_positions(df):
    s = df[df["after_approval"]].sort_values(["mold_id", "shot_ts"])[["shot_id", "mold_id", "job_id", "shot_ts"]].copy()
    s["mpos"] = s.groupby("mold_id").cumcount()
    return s


def drift_lead(df, rng):
    con = connect()
    st = con.execute("select * from drift_shot_state").df()
    cd = con.execute(f"""select job_id, mold_id, hour_ts, defect_code from fct_confirmed_defects
                         where source in ('sort', 'audit', 'tally') and job_id is not null""").df()
    con.close()
    pos = mold_positions(df).merge(st, on="shot_id", how="left")
    out = {}
    for name, (codes, det) in DRIFT_FOR.items():
        ev = cd[cd["defect_code"].isin(codes)].sort_values(["mold_id", "hour_ts"])
        ev = ev[(ev["hour_ts"] >= START) & (ev["hour_ts"] < END)]
        # an episode: confirmed defects of the group on a mold, a gap of more than one shift starting a new one
        ev["new"] = ev.groupby("mold_id")["hour_ts"].diff().fillna(pd.Timedelta(days=999)) > pd.Timedelta(hours=8)
        starts = ev[ev["new"]]
        rows = []
        for m, g in starts.groupby("mold_id"):
            pm = pos[pos["mold_id"] == m]
            if pm[f"{det}_fire"].isna().all():
                continue                                    # this mold does not carry the detector (no end-of-fill sensor)
            ts = pm["shot_ts"].to_numpy()
            fires = pm["mpos"].to_numpy()[pm[f"{det}_fire"].fillna(False).to_numpy(bool)]
            in_period = pm[(pm["shot_ts"] >= START) & (pm["shot_ts"] < END) & (pm["mpos"] >= LEAD_WINDOW)]["mpos"].to_numpy()

            def look(p0):
                f = fires[(fires < p0) & (fires >= p0 - LEAD_WINDOW)]
                return (True, int(p0 - f.max()), int(p0 - f.min())) if len(f) else (False, np.nan, np.nan)
            for h in g["hour_ts"]:
                i = np.searchsorted(ts, np.datetime64(h))
                if i >= len(ts) or i < LEAD_WINDOW:
                    continue
                hit, last, first = look(int(pm["mpos"].iloc[i]))
                rows.append(dict(mold_id=m, kind="episode", fired=hit, lead_last=last, lead_first=first))
            for p0 in rng.choice(in_period, size=min(400, len(in_period)), replace=False) if len(in_period) else []:
                hit, last, first = look(int(p0))
                rows.append(dict(mold_id=m, kind="random", fired=hit, lead_last=last, lead_first=first))
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        e, c = r[r["kind"] == "episode"], r[r["kind"] == "random"]
        out[name] = dict(detector=det, window_shots=LEAD_WINDOW, episodes=int(len(e)), molds=sorted(e["mold_id"].unique()),
                         preceded=float(e["fired"].mean()), chance=float(c["fired"].mean()),
                         median_lead_last=float(e["lead_last"].median()), median_lead_first=float(e["lead_first"].median()),
                         by_mold={m: dict(episodes=int((e["mold_id"] == m).sum()), preceded=float(e.loc[e["mold_id"] == m, "fired"].mean()),
                                          chance=float(c.loc[c["mold_id"] == m, "fired"].mean()))
                                  for m in e["mold_id"].unique()})
    return out


def vent_check(df):
    """End-of-fill CUSUM (deployed limit) against the burn rise, per vent-cleaning interval: the realism requirement."""
    con = connect()
    st = con.execute("select shot_id, cusum_eof_fire from drift_shot_state").df()
    maint = con.execute("select mold_id, event_ts from stg_toolroom__mold_maintenance where event_type = 'vent_cleaning'").df()
    con.close()
    reg = pd.read_parquet(REF / "root_cause_register.parquet")
    burns = reg[reg["defect_code"] == "burn"].groupby("shot_id").size()
    pos = df[df["after_approval"]][["shot_id", "mold_id", "shot_ts"]].merge(st, on="shot_id")
    pos["burn"] = pos["shot_id"].map(burns).fillna(0)
    leads, n_int = [], 0
    for m, g in pos.groupby("mold_id"):
        if not g["cusum_eof_fire"].notna().any() or g["cusum_eof_fire"].fillna(False).sum() == 0 and g["burn"].sum() == 0:
            continue
        g = g.sort_values("shot_ts")
        cl = maint[maint["mold_id"] == m]["event_ts"].sort_values()
        edges = [pd.Timestamp.min] + list(cl) + [pd.Timestamp.max]
        for a, b in zip(edges[:-1], edges[1:]):
            iv = g[(g["shot_ts"] > a) & (g["shot_ts"] <= b)].reset_index(drop=True)
            if len(iv) < 5000:
                continue
            n_int += 1
            f = np.where(iv["cusum_eof_fire"].fillna(False).to_numpy(bool))[0]
            roll = iv["burn"].rolling(1000, min_periods=1000).sum().to_numpy()
            b0 = np.nanmean(roll[:3000]) if np.isfinite(roll[:3000]).any() else 0
            rise = np.where(roll > max(3.0, 3 * b0))[0]
            leads.append(dict(mold_id=m, fired=len(f) > 0, rise=len(rise) > 0,
                              lead=int(rise[0] - f[0]) if len(f) and len(rise) else np.nan))
    r = pd.DataFrame(leads)
    lead = r["lead"]
    return dict(intervals=int(n_int), with_rise=int(r["rise"].sum()), with_signal=int(r["fired"].sum()),
                median_lead=float(lead.median()), positive_lead_share=float((lead > 0).sum() / max(n_int, 1)),
                lead_2000_6000_share=float(((lead >= 2000) & (lead <= 6000)).sum() / max(n_int, 1)))


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
    print("budget", json.dumps(out["budget"], default=float), flush=True)
    out["shot_level"] = shot_level(s, vd, rng)
    out["drift_lead"] = drift_lead(df, rng)
    out["vent_check"] = vent_check(df)
    out["anomaly"] = anomaly_events(df, s, out["budget"])
    out["vm"] = vm_summary()
    (RESULTS / "measures.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", RESULTS / "measures.json")


if __name__ == "__main__":
    main()
