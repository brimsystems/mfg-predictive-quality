"""
ML technical report: the data and how it was generated, the realism checks, the SPC layer, the two model components,
the supervised attempt, and detection at the deployed thresholds, May 2025 to March 2026.

Usage: python ml/reports/generate_ml_technical.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_source.generate.config import CELL, DEFECTS, LABELS, MOLDS  # noqa: E402
from ml.reports import results as RS  # noqa: E402
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREEN, GREY, RED, fig, img, pct, shell, table  # noqa: E402
from ml.src.features import CAVITY, HISTORY, MACHINE, STATE  # noqa: E402

OUT = Path(__file__).resolve().parent / "ml_technical.html"


def curve_figure():
    s = RS.q("""
        with pick as (select shot_id, mold_id from fct_shot where mold_id = 'M-2118' and curve_retained and after_approval
                      and unit_alarm_state = 'none' and shots_since_approval > 2000 limit 1)
        select c.shot_id, c.sensor_id, c.t_ms, c.pressure_bar, n.sensor_position
        from stg_monitoring__cavity_curves c join pick using (shot_id)
        join (select distinct sensor_id, sensor_position from stg_monitoring__shot_summary where mold_id = 'M-2118') n using (sensor_id)
        where c.mold_id = 'M-2118' order by c.t_ms""")
    f, ax = fig(3.6)
    for (sid, pos), g in s.groupby(["sensor_id", "sensor_position"]):
        ax.plot(g["t_ms"] / 1000, g["pressure_bar"], lw=1.2, color=BRAND_BLUE if pos == "post_gate" else AMBER,
                label=f"{sid} ({pos.replace('_', ' ')})")
    ax.set_xlim(0, 16)
    ax.set_xlabel("time in cycle (s)")
    ax.set_ylabel("cavity pressure (bar)")
    ax.legend(frameon=False, fontsize=9)
    for x, t in [(0.35, "arrival"), (1.0, "fill rise"), (1.5, "transfer"), (4.5, "pack plateau"), (8.6, "gate seal"), (11.5, "decay")]:
        ax.annotate(t, (x, ax.get_ylim()[1] * 0.93), fontsize=8.5, color="#555", ha="center")
    return img(f, "retained curves")


def mechanism_curves():
    """Mean post-gate curve of M-2041 shots early and late in a vent interval (end-of-fill sensor)."""
    s = RS.q("""
        with sh as (select shot_id, shots_since_vent_cleaning as v from fct_shot
                    where mold_id = 'M-2041' and curve_retained and after_approval and unit_alarm_state <> 'alarm'),
        tagged as (select shot_id, case when v < 5000 then 'just after cleaning' when v > 22000 then 'near the cleaning count' end as phase
                   from sh where v < 5000 or v > 22000),
        eof as (select distinct sensor_id from stg_monitoring__shot_summary where mold_id = 'M-2041' and sensor_position = 'end_of_fill')
        select t.phase, c.t_ms, avg(c.pressure_bar) as p
        from stg_monitoring__cavity_curves c join tagged t using (shot_id)
        where c.mold_id = 'M-2041' and c.sensor_id in (select sensor_id from eof) and c.t_ms < 3000
        group by 1, 2 order by 2""")
    f, ax = fig(3.0, 6.5)
    for (ph, g), c in zip(s.groupby("phase"), [RED, BRAND_BLUE]):
        ax.plot(g["t_ms"] / 1000, g["p"], color=c, lw=1.4, label=ph)
    ax.set_xlabel("time in cycle (s)")
    ax.set_ylabel("end-of-fill pressure (bar)")
    ax.legend(frameon=False)
    return img(f, "vent restriction")



REF = RS.REF
ORDER = ["template_sort", "rules", "drift", "anomaly", "virtual_metrology"]
NAMES = {"template_sort": "Template sort", "rules": "Rules 1 and 2", "drift": "Drift detection", "anomaly": "Anomaly detection",
         "virtual_metrology": "Virtual metrology advisory", "ml_layers": "Both models"}
GROUPS = ["Fill volume", "Packing and shrinkage", "Flow front", "Gate and other", "Material"]


def sgn(x):
    v = round(x * 100, 1)
    return "0.0" if v == 0 else f"{v:+.1f}"


def ac(r, ci=True):
    s = sgn(r["above_chance"])
    return s + (f" [{sgn(r['ci_above_chance'][0])}, {sgn(r['ci_above_chance'][1])}]" if ci else "")


def in_band_shares():
    """Share of dimensional and warp defects on shots where every template value sat inside its alarm band: among
    confirmed defects (the target) and among all defective pieces (reference: the root-cause register)."""
    shots = pd.read_parquet(REF / "shot_truth.parquet", columns=["shot_id", "in_production", "alarm_state"])
    reg = pd.read_parquet(REF / "root_cause_register.parquet", columns=["shot_id", "defect_code", "root_cause_code", "caught_by"])
    prod = shots[shots["in_production"]]
    inband = set(prod.loc[prod["alarm_state"] != "alarm", "shot_id"])
    rp = reg[reg["shot_id"].isin(set(prod["shot_id"]))]
    dw = rp[rp["defect_code"].isin(["dimensional", "warp"])]
    conf = dw[dw["caught_by"].notna()]
    allc = rp[rp["caught_by"].notna()]
    return dict(confirmed=float(conf["shot_id"].isin(inband).mean()), all=float(dw["shot_id"].isin(inband).mean()),
                n_conf=len(conf), n_all=len(dw), g10=float((allc["root_cause_code"] == "G10").mean()))


def early_period():
    """Why January to April 2025 holds so many of the defective audited pieces: the jobs, their technicians and the
    mechanisms behind them (root-cause register)."""
    ap = RS.q("""select a.shot_id, a.cavity_id, a.defect_code, a.shot_ts, a.job_id, a.mold_id, a.press_id
                 from fct_audit_piece a where a.linked_to_shot and a.after_approval""")
    reg = pd.read_parquet(REF / "root_cause_register.parquet", columns=["shot_id", "cavity", "root_cause_code"])
    reg = reg.rename(columns={"cavity": "cavity_id"}).drop_duplicates(["shot_id", "cavity_id"])
    ap = ap.merge(reg, on=["shot_id", "cavity_id"], how="left")
    ap["defective"] = ap["defect_code"].notna()
    ap["month"] = ap["shot_ts"].dt.to_period("M").astype(str)
    monthly = ap.groupby("month").agg(pieces=("defective", "size"), defective=("defective", "sum"))
    monthly["rate"] = monthly["defective"] / monthly["pieces"]
    e = ap[ap["defective"] & (ap["shot_ts"] < "2025-05-01")]
    top = e.groupby(["job_id", "mold_id", "press_id"]).agg(n=("defective", "size"),
                                                          cause=("root_cause_code", lambda x: ", ".join(f"{k} {v}" for k, v in x.value_counts().head(2).items())))
    top = top.sort_values("n", ascending=False).head(3).reset_index()
    runs = pd.read_parquet(REF / "run_log.parquet", columns=["job_id", "technician_id", "tenure_days"])
    top = top.merge(runs, on="job_id", how="left")
    lots = RS.q(f"""select job_id, string_agg(distinct resin_lot_id, ', ') as lots, min(cert_mfi_band_position) as mfi_pos
                    from int_shot_context where job_id in ({", ".join(f"'{j}'" for j in top["job_id"])}) and after_approval group by 1""")
    top = top.merge(lots, on="job_id", how="left")
    return monthly, top, int(len(e)), int(ap["defective"].sum())


def capability_comparison(R):
    """Critical-dimension capability per run in the evaluation period: from the hourly audits, from the predicted dimension
    on every shot (the rolling-origin model in force that month), and from the prediction corrected for model error.
    Uses Ppk (overall spread) throughout so the three compare."""
    parts = R["parts"].set_index("mold_id")
    jobs = RS.q("select job_id, mold_id, press_id from fct_job where first_shot_ts >= '2025-05-01' and first_shot_ts < '2026-04-01'")
    vbc = RS.vm_by_cell(R)
    vmm = vbc[vbc["target"] == "dimension"].set_index("cell")[["rmse", "gauge_sd"]]
    rows = []
    for j in jobs.itertuples():
        p = parts.loc[j.mold_id]
        nom, tol = p["critical_dimension_mm"], p["dimension_tolerance_mm"]
        a = RS.q(f"""select p.dimension_1 as d from stg_qms__qc_audit_pieces p join stg_qms__qc_audits a using (audit_id)
                     where a.job_id = '{j.job_id}'""")["d"]
        sh = RS.q(f"select shot_id, active_cavities from fct_shot where job_id = '{j.job_id}' and after_approval")
        cell = f"{j.mold_id} / {j.press_id}"
        if len(a) < 30 or sh.empty or cell not in vmm.index:
            continue
        pr = RS.vm_shots(sh["shot_id"])
        pr = pr[pr["cavity_id"] <= pr["shot_id"].map(sh.set_index("shot_id")["active_cavities"])]
        if pr.empty:
            continue
        d = nom + pr["dim_pred"] * tol
        err_var = max(vmm.loc[cell, "rmse"] ** 2 - vmm.loc[cell, "gauge_sd"] ** 2, 0.0)
        ppk = lambda mu, sd: min(nom + tol - mu, mu - nom + tol) / (3 * sd)
        s_corr = float(np.sqrt(d.var() + err_var))
        rows.append(dict(job=j.job_id, cell=cell, audit_n=len(a), ppk_audit=ppk(a.mean(), a.std()), pieces=len(d),
                         ppk_pred=ppk(d.mean(), d.std()), ppk_pred_corr=ppk(d.mean(), s_corr),
                         sd_audit=a.std(), sd_pred=d.std(), sd_corr=s_corr))
    return pd.DataFrame(rows)


def main():
    R = RS.load()
    S = R["shots"]
    M, DM = R["M"], R["DM"]
    SL = M["shot_level"]
    lay, inc, incw = SL["layer"], SL["incremental"], SL["incremental_warm"]
    bud = M["budget"]
    checks = pd.DataFrame(R["checks"])
    ranged = checks[checks["passed"].notna()]
    n_pass = int(ranged["passed"].sum())
    fails = ranged[ranged["passed"] == False]
    ib = in_band_shares()

    # ── realism checks table ──
    ct = ranged.assign(value=ranged.apply(lambda r: f"{r['value']:.3f}", axis=1),
                       range=ranged.apply(lambda r: f"{r['lo']:.3g} to {r['hi']:.3g}", axis=1),
                       result=ranged["passed"].map({True: "in range", False: "out of range"}))[["group", "name", "value", "range", "result"]]
    ct.columns = ["Check", "Measure", "Value", "Target", "Result"]
    ft = fails.assign(value=fails["value"].map(lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"),
                      range=fails.apply(lambda r: f"{r['lo']:,.3g} to {r['hi']:,.3g}", axis=1))[["name", "value", "range"]]
    ft.columns = ["Check", "Value", "Target"]

    # ── mechanism table ──
    share = R["register_share"]
    mech = pd.DataFrame([
        ("G1", "Resin lot viscosity", f"lot MFI spread {pct(0.065, 1)} of nominal; supplier RS-3 lots sit low-flow"),
        ("G2", "Vent restriction", f"rises with shots since cleaning (exponent {CELL['vent_exp']:.0f}); end-of-fill loss {CELL['vent_eof_coef']} per unit"),
        ("G3", "Setup convergence", f"initial offset {CELL['g3_offset']}; time constant {CELL['g3_tau_shots'][0]:.0f} to {CELL['g3_tau_shots'][1]:.0f} shots by tenure on the mold"),
        ("G4", "Check ring leakage", f"IM-12 drifts {CELL['ring_leak_rate_per_day']} per day until the ring replacement; pack variance x(1 + {CELL['ring_var_coef']:.0f} x leak)"),
        ("G5", "Moisture", f"residence below minimum after a fresh lot; faulty dew-point sensor on {CELL['faulty_dryer']} {CELL['faulty_from']} to {CELL['faulty_to']}"),
        ("G6", "Mold temperature", f"controller drift {CELL['tcu_drift_c_per_day']} C per day between services; summer +{CELL['summer_peak_c']} C; moves the dimension and warp through cooling"),
        ("G7", "Hot-runner tip wear", f"M-2041, M-2043; tip change every {CELL['tip_change_shots']:,} shots"),
        ("G8", "Technician adjustments", f"{CELL['changes_per_run'][0]} to {CELL['changes_per_run'][1]} checks per run; tampering share {CELL['tamper_share']}; melt temperature moves the dimension"),
        ("G9", "Cavity imbalance and blocking", f"blocking on {pct(CELL['block_prob_per_run'], 0)} of runs; steps values, template re-established"),
        ("G10", "Novel signatures", f"{CELL['g10_events_per_year']} events per year: failing ring, heater zone (both gradual), nozzle drool, wrong material"),
        ("G0", "Residual", "handling, contamination and specks with no curve signal"),
    ], columns=["Code", "Mechanism", "Parameters"])
    mech["Share of confirmed defects"] = mech["Code"].map(lambda c: pct(share.get(c, 0.0), 1))

    molds = pd.DataFrame([dict(Mold=k, Part=v["desc"], Cavities=v["cavities"], Resin=v["resin"], Cycle=f"{v['cycle_s']:.0f} s",
                               Sensors=", ".join(f"{p.replace('_', ' ')} c{c}" for p, c in v["sensors"]),
                               Transfer=f"{v['transfer_bar']} bar", Pack=f"{v['pack_bar']} bar", Seal=f"{v['seal_s']} s")
                          for k, v in MOLDS.items()])

    # ── labels ──
    codes = R["codes"].pivot_table(index="defect_code", columns="source", values="qty", aggfunc="sum", fill_value=0)
    codes["linked to a shot"] = codes.get("sort", 0) + codes.get("audit", 0)
    codes = codes.sort_values("linked to a shot", ascending=False).reset_index().rename(columns={"defect_code": "code"})
    codes["code"] = codes["code"].str.replace("_", " ")
    monthly, top, n_early, n_def15 = early_period()
    mt = monthly.reset_index().assign(rate=lambda d: d["rate"].map(lambda v: pct(v, 1)))
    mt.columns = ["Month", "Audited pieces", "Defective", "Defect rate"]
    top_txt = "; ".join(f"{r.job_id} ({r.mold_id} on {r.press_id}, {r.n} pieces; {r.cause}; technician {r.technician_id}, "
                        f"{r.tenure_days:.0f} days' tenure; lots {r.lots})" for r in top.itertuples())

    # ── SPC ──
    pts_ = R["spc_points"]
    rs = R["lot_steps"]
    rs3 = rs.assign(rs3=rs["supplier_id"] == "RS-3").groupby("rs3")["fill_step"].mean()
    lv, ls_, rg = DM["vent"], DM["lot_steps"], DM["ring"]
    dm_t = pd.DataFrame([
        ("End-of-fill CUSUM, vent cleaning intervals", f"first firing before the burn rise: {pct(lv['preceded']['share'], 0)} of {lv['preceded']['n']} "
         f"[{pct(lv['preceded']['ci'][0], 0)}, {pct(lv['preceded']['ci'][1], 0)}]; median lead {lv['lead']['median']:,.0f} shots "
         f"[{lv['lead']['ci'][0]:,.0f}, {lv['lead']['ci'][1]:,.0f}]",
         f"random burn-rise time: {pct(lv['chance_preceded']['mean'], 0)} [{pct(lv['chance_preceded']['range95'][0], 0)}, "
         f"{pct(lv['chance_preceded']['range95'][1], 0)}]; median lead {lv['chance_lead_median']:,.0f}"),
        ("Fill EWMA, resin lot changes mid-run", f"steps of at least {ls_['step_threshold_band']} band flagged within {ls_['window_shots']} shots: "
         f"{pct(ls_['with_step']['share'], 0)} of {ls_['with_step']['n']} [{pct(ls_['with_step']['ci'][0], 0)}, {pct(ls_['with_step']['ci'][1], 0)}]; "
         f"median {ls_['median_shots_to_flag']:.0f} shots to flag",
         f"changes under {ls_['no_step_threshold_band']} band flagged: {pct(ls_['without_step']['share'], 0)} of {ls_['without_step']['n']} "
         f"[{pct(ls_['without_step']['ci'][0], 0)}, {pct(ls_['without_step']['ci'][1], 0)}]"),
        ("Variance CUSUM, check-ring wear (jobs at leakage over 0.02)", f"fired before the first sink or void: {rg['fired_before_defect']['k']} of "
         f"{rg['fired_before_defect']['n']} [{pct(rg['fired_before_defect']['ci'][0], 0)}, {pct(rg['fired_before_defect']['ci'][1], 0)}]; "
         f"{rg['fires_per_10k_worn']:.2f} firings per 10,000 shots",
         f"matched unworn windows (leakage under 0.01) flagged: {pct(rg['matched_unworn']['share'], 1)} of {rg['matched_unworn']['n']} "
         f"[{pct(rg['matched_unworn']['ci'][0], 1)}, {pct(rg['matched_unworn']['ci'][1], 1)}]; {rg['fires_per_10k_unworn']:.2f} per 10,000 shots"),
    ], columns=["Detector and mechanism", "Detector", "Comparison"])

    # ── virtual metrology ──
    vm = R["vm_month"]
    vmo = RS.vm_overall(R)
    vmo_abl = RS.vm_overall(R, "no_cavity_signal")
    by_m = vm.groupby(["month", "target", "variant"])[["rmse_over_gauge", "r2"]].median().unstack(["target", "variant"])
    vmt = pd.DataFrame({"Month": by_m.index.strftime("%b %Y")})
    for t in ("dimension", "weight"):
        vmt[f"{t.capitalize()} error vs gauge"] = by_m[("rmse_over_gauge", t, "full")].map(lambda v: f"{v:.2f}x").values
        vmt[f"{t.capitalize()} R2"] = by_m[("r2", t, "full")].map(lambda v: f"{v:.2f}").values
        vmt[f"{t.capitalize()} R2 without cavity pressure"] = by_m[("r2", t, "no_cavity_signal")].map(lambda v: f"{v:.2f}").values
    vsum = pd.DataFrame([(t.capitalize(), per, f"{o[per]['rmse_over_gauge']:.2f}x", f"{o[per]['r2']:.2f}", f"{o[per]['calibration_slope']:.2f}",
                          f"{ob[per]['rmse_over_gauge']:.2f}x", f"{ob[per]['r2']:.2f}")
                         for t in ("dimension", "weight") for o, ob in [(vmo[t], vmo_abl[t])] for per in ("all", "warm", "other")],
                        columns=["Target", "Months", "Error vs gauge", "R2", "Calibration slope", "Error vs gauge, no cavity pressure", "R2, no cavity pressure"])
    vsum["Months"] = vsum["Months"].map({"all": "May 2025 to March 2026", "warm": "warm (June to September)", "other": "other months"})
    vbc = RS.vm_by_cell(R)
    vbt = vbc.assign(rmse=vbc["rmse"].map(lambda v: f"{v:.4f}"), gauge_sd=vbc["gauge_sd"].map(lambda v: f"{v:.4f}"),
                     rmse_over_gauge=vbc["rmse_over_gauge"].map(lambda v: f"{v:.2f}x"), r2=vbc["r2"].map(lambda v: f"{v:.2f}"),
                     calibration_slope=vbc["calibration_slope"].map(lambda v: f"{v:.2f}"), n=vbc["n"].map(lambda v: f"{v:,}"))
    vbt = vbt[["target", "cell", "n", "rmse", "gauge_sd", "rmse_over_gauge", "r2", "calibration_slope"]]
    vbt.columns = ["Target", "Mold / press", "Pieces", "RMSE", "Gauge sd", "RMSE over gauge", "R2", "Calibration slope"]

    # ── supervised ──
    dt = RS.defect_table(R)
    dtt = dt[["code", "n_train_pos", "n_test_pos", "recall_model", "recall_model_sd", "recall_template", "recall_spc", "ap_model", "ap_template"]].copy()
    for c in ("recall_model", "recall_template", "recall_spc", "ap_model", "ap_template", "recall_model_sd"):
        dtt[c] = dtt[c].map(lambda v: f"{v:.3f}")
    dtt.columns = ["Code", "Train positives", "Test positives", "Recall, model", "sd over seeds", "Recall, template", "Recall, rules",
                   "AP, model", "AP, template"]
    unusable = R["defect"][R["defect"]["usable"] == False]["target"].unique()
    da = R["defect_abl"][R["defect_abl"]["target"] == "y_any"].groupby("tag")[["recall_model", "ap_model"]].mean()
    ao = R["audit_only"]

    # ── anomaly ──
    A = M["anomaly"]
    ev = pd.DataFrame(A["rows"])
    evt = ev[["kind", "mold_id", "press_id", "start", "shots", "detected", "minutes_to_flag", "minutes_to_template_alarm",
              "lead_over_response_min", "before_template"]].copy()
    evt["kind"] = evt["kind"].str.replace("_", " ")
    evt["start"] = pd.to_datetime(evt["start"]).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("minutes_to_flag", "minutes_to_template_alarm", "lead_over_response_min"):
        evt[c] = evt[c].map(lambda v: "" if v != v else f"{v:.0f}")
    evt.columns = ["Event", "Mold", "Press", "Start", "Shots", "Flagged", "Minutes to first flag", "Minutes to first template alarm",
                   "Flag lead over the shop's response (min)", "Before the template"]
    ae = R["ae"]
    thr = R["anomaly_thresholds"]

    # ── detection ──
    bt = pd.DataFrame([
        ("Template sort", "the template's alarm bands", pct(bud["template_sort"]["shots_flagged"], 2), f"{bud['template_sort']['episodes_per_shift']:.2f}"),
        ("Rules 1 and 2", "4.05 and 2.70 sigma on the AR(1) residual charts", pct(bud["rules"]["shots_flagged"], 2), f"{bud['rules']['episodes_per_shift']:.2f}"),
        ("Rules 1 and 2 at the standard limits (not deployed)", "3 and 2 sigma", pct(bud["rules_at_textbook_limits"]["shots_flagged"], 2),
         f"{bud['rules_at_textbook_limits']['episodes_per_shift']:.2f}"),
        ("Drift detection", "standard limits; re-armed after 3,000 shots unless reset", pct(bud["drift"]["shots_flagged"], 2), f"{bud['drift']['episodes_per_shift']:.2f}"),
        ("Anomaly detection", "two of ten above a threshold for 0.30% of validated shots", pct(bud["anomaly"]["shots_flagged"], 2), f"{bud['anomaly']['episodes_per_shift']:.2f}"),
        ("Virtual metrology advisory", "predicted dimension past 75% of tolerance", pct(bud["virtual_metrology"]["shots_flagged"], 2),
         f"{bud['virtual_metrology']['episodes_per_shift']:.2f}"),
        ("Added layers (rules, drift, anomaly, advisory)", "", "", f"{bud['added_layers_total']:.2f}"),
    ], columns=["Layer", "Deployed threshold", "Shots flagged", "Episodes per shift"])
    rows = []
    for k in ORDER:
        r = lay[k]["All defects"]
        rows.append(dict(Layer=NAMES[k], **{"Defective flagged": pct(r["rate_defective"]), "Good flagged": pct(r["rate_good"]),
                                             "Above chance": ac(r), "Incremental": ac(inc[k]["All defects"]),
                                             "Incremental, warm months": ac(incw[k]["All defects"], ci=False)}))
    rows.append(dict(Layer=NAMES["ml_layers"], **{"Defective flagged": pct(lay["ml_layers"]["All defects"]["rate_defective"]),
                                                   "Good flagged": pct(lay["ml_layers"]["All defects"]["rate_good"]),
                                                   "Above chance": ac(lay["ml_layers"]["All defects"]),
                                                   "Incremental": ac(inc["ml_layers"]["All defects"]),
                                                   "Incremental, warm months": ac(incw["ml_layers"]["All defects"], ci=False)}))
    shot_t = pd.DataFrame(rows)
    grp_cols = GROUPS + ["Dimensional and warp"]
    gtab = pd.DataFrame([dict(Layer=NAMES[k], **{g: ac(lay[k][g], ci=False) for g in grp_cols}) for k in ORDER])
    itab = pd.DataFrame([dict(Layer=NAMES[k], **{g: ac(inc[k][g], ci=False) for g in grp_cols}) for k in ORDER + ["ml_layers"]])
    iwtab = pd.DataFrame([dict(Layer=NAMES[k], **{g: ac(incw[k][g], ci=False) for g in grp_cols}) for k in ORDER + ["ml_layers"]])
    ntab = pd.DataFrame([dict(Group=g, **{"Defective pieces": f"{lay['template_sort'][g]['n_defective']:,}"}) for g in grp_cols])
    bc = SL["by_code"]
    ctab = pd.DataFrame([dict(Code=c.replace("_", " "), Group=v["group"], **{"Defective pieces": f"{v['n_defective']:,}",
                                                                           "Template flagged": pct(v["template"]["rate_defective"]),
                                                                           "Template above chance": ac(v["template"], ci=False),
                                                                           "Any layer flagged": pct(v["any_layer"]["rate_defective"]),
                                                                           "Any layer above chance": ac(v["any_layer"], ci=False)})
                         for c, v in sorted(bc.items(), key=lambda kv: (GROUPS.index(kv[1]["group"]), -kv[1]["n_defective"]))])
    C15 = M["context_15_months"]
    c15 = pd.DataFrame([dict(Layer=NAMES[k], **{"All defects": ac(C15["layer"][k]["All defects"]),
                                                 **{g: ac(C15["layer"][k][g], ci=False) for g in GROUPS}}) for k in ("template_sort", "rules", "drift")])
    JH = M["job_hour"]

    # ── capability ──
    cc = capability_comparison(R)
    infl = float((cc["ppk_pred"] > cc["ppk_audit"]).mean()) if len(cc) else float("nan")
    ratio_sd = float((cc["sd_pred"] / cc["sd_audit"]).median()) if len(cc) else float("nan")
    gap_raw = float((cc["ppk_pred"] - cc["ppk_audit"]).median()) if len(cc) else float("nan")
    gap_corr = float((cc["ppk_pred_corr"] - cc["ppk_audit"]).median()) if len(cc) else float("nan")
    cct = cc.assign(**{c: cc[c].map(lambda v: f"{v:.2f}") for c in ("ppk_audit", "ppk_pred", "ppk_pred_corr")},
                    pieces=cc["pieces"].map(lambda v: f"{v:,.0f}"))[["job", "cell", "audit_n", "ppk_audit", "pieces", "ppk_pred", "ppk_pred_corr"]]
    cct.columns = ["Job", "Mold / press", "Audit pieces", "Ppk, audits", "Predicted pieces", "Ppk, prediction", "Ppk, prediction corrected"]
    f, ax = fig(3.4, 5.4)
    ax.scatter(cc["ppk_audit"], cc["ppk_pred"], color=GREY, s=18, label="prediction, uncorrected")
    ax.scatter(cc["ppk_audit"], cc["ppk_pred_corr"], color=BRAND_BLUE, s=18, label="prediction, corrected for model error")
    lim = [0, max(4.0, float(cc[["ppk_audit", "ppk_pred"]].max().max()) * 1.05)] if len(cc) else [0, 4]
    ax.plot(lim, lim, color="#999", lw=0.8)
    ax.set_xlabel("Ppk from the hourly audits")
    ax.set_ylabel("Ppk from the predicted dimension")
    ax.legend(frameon=False, fontsize=8.5)
    cap_png = img(f, "capability comparison")

    body = f"""
<div class="note">The data behind this report is generated by a physical model of the cell (section 1). It validates the pipeline, the label
design and the evaluation; it does not show how the models would transfer to another press, mold or plant.</div>

<h2 id="model">1. The curve model</h2>
<p>Every shot's cavity pressure curve is built from the shot's effective state, and every summary value, alarm, sort decision and defect follows
from the curve and the state; no defect is written first and a curve fitted to it. Curves are built at 500 samples per second over the active part of
the cycle in six phases, summarized the way the monitoring unit summarizes them, and retained at 100 readings per second on one shot in
{CELL['curve_retain_every']} and every alarmed or sorted shot.</p>
{curve_figure()}
<p class="caption">A retained M-2118 shot: post-gate sensors arrive first and hold a higher plateau; the end-of-fill sensors arrive later by the
flow-length ratio and sit lower by the gate-to-end pressure drop.</p>
{table(molds)}
<h3>Mechanisms</h3>
<p>Eleven mechanisms move the effective state over the fifteen months. Their shares of confirmed defects come from quality engineering's
root-cause register, kept outside the load path and used only here and in the realism checks.</p>
{table(mech)}
{mechanism_curves()}
<p class="caption">G2 on M-2041: the end-of-fill pressure in the fill phase, averaged over retained shots just after a vent cleaning and near the
cleaning count.</p>
<h3 id="changes">1.1 Two failure modes the template bands do not see</h3>
<p>Two mechanisms were built so the data carries failure modes that univariate template bands miss, as they do on a real press.</p>
<p><strong>Critical dimension and warp follow the part's thermal history.</strong> A part shrinks as it cools. How much it shrinks depends on how much
material was packed into the cavity (the pack integral and the gate seal time, which the template watches) and on how hot the part is when it leaves the
mold, which the template does not watch: a hotter mold, a hotter melt or a shorter cooling time ejects a hotter part that keeps shrinking outside the mold,
so the dimension comes out smaller. A temperature difference between the two mold halves cools the faces unevenly and bows the part. The critical
dimension therefore moves jointly with the mold temperature actuals on both halves (0.085 of tolerance per degree above setpoint, and 0.03 per degree of
difference between the halves), the melt temperature (0.012 per degree), the cooling time (0.12 per second; 40% of runs have their cooling shortened by
0.3 to 1.0 s for throughput, which shows in the recorded cycle time) and the cavity's own offset, as well as the pack integral and gate seal time. Warp
comes from the same uneven cooling, plus residual stress from packing well off its level. A mold-temperature controller drifting between services (G6)
or summer chiller load can now push parts out of tolerance, or warp them, while the pack integral stays inside its band: {pct(ib['confirmed'], 0)} of the
{ib['n_conf']:,} confirmed dimensional and warp defects fall on shots where every template value is inside its alarm band (target 30 to 50%), and
{pct(ib['all'], 0)} of all {ib['n_all']:,} defective dimensional and warp pieces, most of which are never confirmed because nothing sorts them.</p>
<p><strong>Novel faults arrive as combinations of small shifts.</strong> A check ring that starts to fail lets melt slip back past the screw tip as it
wears through: fill takes longer, the cushion left in front of the screw varies more from shot to shot, and the pack integral eases down because less of
the hold pressure reaches the cavity. Leakage builds over the event as the 1.5th power of elapsed time, so each value sits inside its own band early and
crosses only as the ring wears through. A failing heater zone lets one barrel zone fall away from its setpoint, gradually, by up to 40 degrees; the
melt stiffens and the fill integral drifts with it. Novel events make up {pct(ib['g10'], 1)} of confirmed defects (target 3 to 6%).</p>

<h2 id="sensors">2. Sensor realism</h2>
<p>Each sensor carries a gain error of {CELL['sensor_gain_err'][0] * 100:.0f} to {CELL['sensor_gain_err'][1] * 100:.0f}%, an offset that drifts slowly,
AR(1) noise with autocorrelation {CELL['ar_integrals'][0]} to {CELL['ar_integrals'][1]} on integrals and {CELL['ar_timings'][0]} to
{CELL['ar_timings'][1]} on timings, dropouts on {pct(CELL['dropout_rate'], 1)} of shots and spikes on {pct(CELL['spike_rate'], 2)}.
Measured on the extract: {pct(RS.check(R, 'summary rows dropped')['value'], 2)} of summary rows dropped and a lag-1 autocorrelation of
{RS.check(R, 'lag-1 autocorrelation')['value']:.2f} on the pack integral's shot-to-shot residual.</p>

<h2 id="labels">3. Label design</h2>
<p>Labels come from the shop's own inspection processes over the true quality of each piece, with their coverage and noise: the reject-bin review
(per sorted shot through an indexed tray on the medical molds M-2118 and M-2119; once per shift by job, with no shot, on the other four), hourly audits of
{LABELS['audit_pieces'][0]} to {LABELS['audit_pieces'][1]} pieces ({pct(R['audits']['linked'], 0)} linked to a shot through the robot's recorded cycle,
on all six molds), packing tallies by hour and code, and customer returns weeks later. A job's quality is taken as known {LABELS['maturity_days']} days
after its last shot. Detection is measured on the audit pieces, which are pulled on a clock from every mold; the supervised model's labels also include
the sort reviews tied to a shot, which come only from the medical molds. Confirmed defective pieces by code and source:</p>
{table(codes, {c: (lambda v: f'{v:,.0f}') for c in codes.columns if c != 'code'})}
<h3 id="early">3.1 January to April 2025</h3>
<p>January to April 2025 holds {n_early:,} of the {n_def15:,} defective audited pieces ({pct(n_early / n_def15, 0)}), and the audit defect rate
falls from {pct(monthly['rate'].iloc[0], 1)} in January to {pct(monthly.loc['2025-04', 'rate'], 1)} in April. The cause is in the data, not the
generator's start-up: three jobs hold most of it. {top_txt}. Two technicians joined in January and February 2025, and a technician new to the
cell approves runs before setup has converged (G3); confirmed setup-convergence defects fall as they gain experience. The January job on M-2301 ran a
lot at the bottom of its certified melt-flow band (short shots, G1) on vents that had run long since their last cleaning (burns, G2); the vent
restriction matches that of later M-2301 runs, and the two conditions simply coincided. These are fill-volume defects the template detects well,
which is why the fifteen-month figures run higher than the evaluation period's.</p>
{table(mt)}

<h2 id="checks">4. Realism checks</h2>
<p>{n_pass} of {len(ranged)} ranged checks fall inside their targets on this run. The dimension R&sup2; on the pack integral and gate seal time has a
target of 0.35 to 0.55, lower than the earlier 0.65 to 0.85, because the dimension now also follows the mold temperatures, melt and cooling time. The
checks outside their targets are listed here with their values rather than tuned away:</p>
{table(ft)}
{table(ct)}

<h2 id="features">5. Feature pipeline and the as-of rule</h2>
<p>All three model components share one feature pipeline, built in dbt: summary values normalized to the template in force (deviation as a fraction
of the alarm band), machine-side values against the process window or relative to the job's first 500 shots (the cycle time among them, which carries
the cooling time), and context joined as of the shot. Every join takes the latest record at or before the shot, and unit tests recompute samples of the
loads, setpoints, maintenance counters and templates to confirm no shot sees a later record. Feature groups, used by the ablations: cavity
({len(CAVITY)}), machine side ({len(MACHINE)}), history and context ({len(HISTORY)}), and control-chart and drift state ({len(STATE)}).</p>

<h2 id="spc">6. The SPC layer</h2>
<p>The unit's alarm logic, recomputed from summary values and templates, agrees with the recorded state on {pct(S['unit_agreement'], 2)} of production
shots (a dbt test). The control charts plot each shot's residual from an AR(1) fit (the mean and coefficient from the run's shots 200 to 1,000, the
residual sigma pooled over earlier runs of the same mold, press and sensor), centred on the previous 500 shots, so the points are close to independent.
Rules 1 and 2 are deployed at <strong>4.05 and 2.70 sigma</strong>, wider than the standard 3 and 2 sigma, chosen to meet the alarm budget of about one
episode per shift: at the standard limits they would raise {bud['rules_at_textbook_limits']['episodes_per_shift']:.1f} episodes per shift. At the deployed
limits rule 1 fires on {pct(pts_['we1'], 2)} of chart points and rule 2 on {pct(pts_['we2'], 2)}; rules 4 and 5 (on {pct(pts_['we4'], 1)} and
{pct(pts_['we5'], 1)} of points) are not deployed.</p>
<p>Drift detection: EWMA (lambda 0.2) on the post-gate fill and pack integrals, limit 3.5 standard deviations of the run's first 500 shots; CUSUMs on
25-shot block means (k = 0.375, h = 5) of end-of-fill pressure and gate seal time; a CUSUM on the log of the 200-shot pack-integral standard deviation
(k = 0.10, h = 1.2). Each resets at approval, documented drift corrections, vent cleaning, PM, ring replacement and lot changes; after a detector fires,
a new excursion of the same detector within 3,000 shots is not signaled again unless a reset comes between. Each detector is tested against the
mechanism it is built for, with intervals:</p>
{table(dm_t)}
<p>Only the fill EWMA separates from its comparison. The end-of-fill CUSUM's first firing after a cleaning comes about as early as chance would put it;
the realism requirement that it lead the burn rise by 2,000 to 6,000 shots on at least 70% of vent-cleaning intervals is not met at the deployed limit
({pct(lv['lead_2000_6000']['share'], 0)} of intervals). The variance CUSUM cannot see check-ring wear: leakage changes by a median
{rg['leak_change_within_job']:.4f} within a job while the baseline is the job's own first 500 shots, and in {rg['first_defect_in_first_hour']} of
{rg['wear_episodes']} worn jobs the first sink or void comes within the first 500 shots, before monitoring starts. Lot-change step tests on the fill
integral: mean step {rs3.get(True, np.nan):+.3f} bands into RS-3 lots against {rs3.get(False, np.nan):+.3f} into others.</p>

<h2 id="vm">7. Virtual metrology</h2>
<p>XGBoost regression on audit pieces with shot linkage (pooled over mold and press, with the mold and press and the cavity as features), evaluated by
rolling origin: at the start of each month from May 2025 to March 2026 the model is retrained on every earlier month, the last of them held out for
early stopping, and applied to that month only. The same model scores every production shot and cavity of the month for the advisory, the unit screen
and the run reports. The ablation drops the cavity pressure values (machine side and context only).</p>
{table(vsum)}
{table(vmt)}
<p>Per mold and press, pooled over the eleven months:</p>
{table(vbt)}
<p>Weight predictions sit close to the gauge's own error and can support stretching audit intervals where the error stays within about 1.5 times
gauge. Dimension predictions run two to three times gauge, and higher in the warm months, when mold temperature moves the dimension: they flag drift
between audits, shot by shot, but not in place of the audits. Calibration slopes above 1 mean the predictions are slightly compressed toward the mean.</p>

<h2 id="supervised">8. Supervised defect prediction</h2>
<p>Gradient boosting with class weighting on the labeled shots (sort-reviewed and robot-linked audits), trained through September 2025 and compared
at the template's alarm rate on the labeled test set, December 2025 to March 2026, five seeds. Codes with fewer than 30 training or 10 test positives are
not scored: {', '.join(c.replace('y_', '') for c in unusable) or 'none'}.</p>
{table(dtt)}
<p>Ablations on any defect: without the machine side, recall {da.loc['no_machine_side', 'recall_model']:.3f}; without the cavity signal,
{da.loc['no_cavity_signal', 'recall_model']:.3f}. Audit-only re-test, on defects found independently of the sort: recall
{ao['recall_model']:.3f} against the template's {ao['recall_template']:.3f} at the template's alarm rate ({pct(ao['alarm_rate'], 2)}), average precision
{ao['ap_model']:.3f} against {ao['ap_template']:.3f}, {ao['n_test_pos']:.0f} defective of {ao['n_test']:,.0f} shots. No material gain over the template.</p>

<h2 id="anomaly">9. Anomaly detection</h2>
<p>Isolation forest per mold and press on each cavity value's change from the mean of the job's previous 300 shots, with the match score and the
spread between sensed cavities, from shot 500 after approval. Evaluated by rolling origin like virtual metrology: refitted at the start of each month on
validated earlier shots (after approval, no alarm, no linked defect; at most 60,000 per fit), its threshold set by bisection so the alarm state (two
flags in ten shots) covers 0.30% of those shots. Thresholds over the {len(thr)} monthly fits range from {thr['threshold'].min():.3f} to
{thr['threshold'].max():.3f}. Event windows are taken from quality engineering's event log (reference). Six novel events fall in the fifteen months;
the {A['events']} below fall in the evaluation period.</p>
{table(evt)}
<p>The heater-zone flag came {ev.loc[ev['kind'] == 'heater_zone', 'minutes_to_flag'].iloc[0] if (ev['kind'] == 'heater_zone').any() else float('nan'):.0f}
minutes into the event and before the template, but after the shop had already responded, so its lead over the response is negative. The curve
autoencoder, run as a comparison on retained curves, flags {pct(ae['event_flag_rate'], 1)} of retained event shots against
{pct(ae['other_flag_rate'], 1)} of the rest.</p>

<h2 id="detection">10. Detection at the deployed thresholds</h2>
<p>Every figure in this section uses the thresholds below and the episode definition: a layer's flagged shots on a job, with gaps under 30 shots,
count as one investigation. Evaluation period May 2025 to March 2026, {M['budget']['shifts']:.0f} shifts.</p>
{table(bt)}
<h3>Shot-level detection on audit pieces</h3>
<p>{SL['n_audit_pieces']:,} audited pieces linked to their shot, {SL['n_defective']:,} defective. For each layer, the share of defective and of good
pieces whose shot it flagged; above chance is the difference, with 95% intervals from resampling whole audits. Incremental: in the order template, rules,
drift, anomaly, virtual metrology, the pieces a layer flagged that no earlier layer flagged, minus the same share for good pieces. The virtual
metrology advisory is judged on the piece's own cavity.</p>
{table(shot_t)}
<p>Above chance by defect group (points):</p>
{table(gtab)}
<p>Incremental by defect group (points):</p>
{table(itab)}
<p>Incremental by defect group, warm months June to September 2025 (points):</p>
{table(iwtab)}
{table(ntab)}
<h3>By defect code</h3>
{table(ctab)}
<h3>Fifteen-month context</h3>
<p>Template, rules and drift over all audits, January 2025 to March 2026 ({C15['n_defective']:,} defective of {C15['n_audit_pieces']:,} pieces). Context
only: January to April 2025 holds {pct(C15['early_share_of_defective'], 0)} of the defective pieces (section 3.1).</p>
{table(c15)}
<h3>Why job-hour coverage was dropped</h3>
<p>Earlier versions credited a layer with a confirmed defect when the layer had alarmed on the same job in the defect's hour or shortly before. At the
deployed thresholds that measure credits {pct(JH['detected'])} of {JH['n_defects']:,} confirmed defects to some layer, but the same rules credit
{pct(JH['chance'])} of the job-hours of {JH['n_good']:,} good audited pieces. With some layer alarming in most job-hours, most of the credit is
coincidence: only the {(JH['detected'] - JH['chance']) * 100:.1f}-point gap can be read as detection, and the measure cannot say which defects make it up.
Shot-level detection against good pieces from the same audits replaces it.</p>

<h2 id="capability">10a. Capability from audits and from the prediction</h2>
<p>Two estimates of the critical dimension's capability for each run in the evaluation period, as a method comparison: the customer receives the
audit-based figure. The audit estimate rests on a small measured sample (gauge noise included). The prediction estimate covers every piece of every
shot, but it carries model error. The regression smooths, so predictions spread less than the parts do and Ppk comes out too high: predictions spread
{ratio_sd * 100:.0f}% as much as the audits (median), and the uncorrected prediction Ppk exceeds the audit Ppk on {pct(infl, 0)} of runs (median gap
{gap_raw:+.2f}). Because the regression is close to calibrated, its error is close to uncorrelated with its prediction, so the parts' true spread is the
prediction's spread plus the model's error variance (RMSE squared minus gauge sd squared, per mold and press). After the correction the median gap to the
audit Ppk is {gap_corr:+.2f}.</p>
{cap_png}
<p class="caption">Uncorrected predictions overstate capability; the correction brings them close to the audit figures.</p>
{table(cct)}

<h2 id="mechanisms">11. Mechanism recovery</h2>
<p>Each mechanism's signature, measured on the extract the way the platform sees it:</p>
{table(ct[ct['Check'].isin(['4', '10'])])}
"""
    toc = [("model", "Model"), ("changes", "Thermal and novel faults"), ("labels", "Labels"), ("early", "Jan to Apr 2025"), ("checks", "Checks"),
           ("features", "Features"), ("spc", "SPC"), ("vm", "Virtual metrology"), ("supervised", "Supervised"), ("anomaly", "Anomaly"),
           ("detection", "Detection"), ("capability", "Capability"), ("mechanisms", "Mechanisms")]
    OUT.write_text(shell("Cavity Pressure ML Technical Report", "Molding quality · IM-11 and IM-12",
                         f"{S['shots']:,.0f} shots, {S['jobs']} jobs · January 2025 to March 2026 · evaluation May 2025 to March 2026",
                         body, toc), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
