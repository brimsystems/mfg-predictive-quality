"""
ML technical report.

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
from ml.src.features import CAVITY, HISTORY, MACHINE, STATE, VALID_END  # noqa: E402

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


def main():
    R = RS.load()
    S = R["shots"]
    L = R["layers"]
    vmh = RS.vm_headline(R)
    checks = pd.DataFrame(R["checks"])
    ranged = checks[checks["passed"].notna()]
    n_pass = int(ranged["passed"].sum())

    # ── realism checks table ──
    ct = ranged.assign(value=ranged.apply(lambda r: f"{r['value']:.3f}", axis=1),
                       range=ranged.apply(lambda r: f"{r['lo']:.3g} to {r['hi']:.3g}", axis=1),
                       result=ranged["passed"].map({True: "in range", False: "out of range"}))[["group", "name", "value", "range", "result"]]
    ct.columns = ["Check", "Measure", "Value", "Target", "Result"]

    # ── mechanism table ──
    share = R["register_share"]
    mech = pd.DataFrame([
        ("G1", "Resin lot viscosity", f"lot MFI spread {pct(0.065, 1)} of nominal; supplier RS-3 lots sit low-flow"),
        ("G2", "Vent restriction", f"rises with shots since cleaning (exponent {CELL['vent_exp']:.0f}); end-of-fill loss {CELL['vent_eof_coef']} per unit"),
        ("G3", "Setup convergence", f"initial offset {CELL['g3_offset']}; time constant {CELL['g3_tau_shots'][0]:.0f} to {CELL['g3_tau_shots'][1]:.0f} shots by tenure on the mold"),
        ("G4", "Check ring leakage", f"IM-12 drifts {CELL['ring_leak_rate_per_day']} per day until the ring replacement; pack variance x(1 + {CELL['ring_var_coef']:.0f} x leak)"),
        ("G5", "Moisture", f"residence below minimum after a fresh lot; faulty dew-point sensor on {CELL['faulty_dryer']} {CELL['faulty_from']} to {CELL['faulty_to']}"),
        ("G6", "Mold temperature", f"controller drift {CELL['tcu_drift_c_per_day']} C per day between services; summer +{CELL['summer_peak_c']} C"),
        ("G7", "Hot-runner tip wear", f"M-2041, M-2043; tip change every {CELL['tip_change_shots']:,} shots"),
        ("G8", "Technician adjustments", f"{CELL['changes_per_run'][0]} to {CELL['changes_per_run'][1]} checks per run; tampering share {CELL['tamper_share']}"),
        ("G9", "Cavity imbalance and blocking", f"blocking on {pct(CELL['block_prob_per_run'], 0)} of runs; steps values, template re-established"),
        ("G10", "Novel signatures", f"{CELL['g10_events_per_year']} events per year: failing ring, heater zone, nozzle drool, wrong material"),
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

    # ── virtual metrology ──
    vs = RS.vm_summary(R)
    vs_t = vs.assign(rmse=vs["rmse"].map(lambda v: f"{v:.4f}"), gauge_sd=vs["gauge_sd"].map(lambda v: f"{v:.4f}"),
                     rmse_over_gauge=vs["rmse_over_gauge"].map(lambda v: f"{v:.2f}x"), r2=vs["r2"].map(lambda v: f"{v:.2f}"),
                     calibration_slope=vs["calibration_slope"].map(lambda v: f"{v:.2f}"),
                     residual_lag1=vs["residual_lag1"].map(lambda v: f"{v:.2f}"))
    ab = R["vm_abl"].groupby(["target", "tag"])[["rmse_over_gauge", "r2"]].median().reset_index()
    full = R["vm"][R["vm"]["design"] == "pooled"].groupby("target")[["rmse_over_gauge", "r2"]].median().reset_index().assign(tag="all features")
    ab = pd.concat([full, ab]).assign(rmse_over_gauge=lambda d: d["rmse_over_gauge"].map(lambda v: f"{v:.2f}x"),
                                      r2=lambda d: d["r2"].map(lambda v: f"{v:.2f}"))
    ab.columns = ["Target", "RMSE over gauge", "R2", "Features"]
    ab = ab[["Target", "Features", "RMSE over gauge", "R2"]]

    p = R["vm_pred"]
    p = p[p["design"] == "pooled"].copy()
    p["month"] = pd.to_datetime(p["shot_ts"]).dt.to_period("M").astype(str)
    p["err"] = p["measured"] - p["predicted"]
    gauge = R["parts"].set_index("part_id")
    drift_err = p.groupby(["target", "month"])["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack(0)
    f, ax = fig(3.0)
    for t, c in zip(drift_err.columns, [BRAND_BLUE, AMBER]):
        ax.plot(drift_err.index, drift_err[t] / drift_err[t].iloc[0], marker="o", color=c, label=t)
    ax.set_ylabel("RMSE relative to December")
    ax.legend(frameon=False)
    drift_png = img(f, "error over test months")

    # ── supervised ──
    dt = RS.defect_table(R)
    dtt = dt[["code", "n_train_pos", "n_test_pos", "recall_model", "recall_model_sd", "recall_template", "recall_spc", "ap_model", "ap_template"]].copy()
    for c in ("recall_model", "recall_template", "recall_spc", "ap_model", "ap_template"):
        dtt[c] = dtt[c].map(lambda v: f"{v:.3f}")
    dtt["recall_model_sd"] = dtt["recall_model_sd"].map(lambda v: f"{v:.3f}")
    dtt.columns = ["Code", "Train positives", "Test positives", "Recall, model", "sd over seeds", "Recall, template", "Recall, rules",
                   "AP, model", "AP, template"]
    unusable = R["defect"][R["defect"]["usable"] == False]["target"].unique()
    da = R["defect_abl"][R["defect_abl"]["target"] == "y_any"].groupby("tag")[["recall_model", "ap_model"]].mean()

    # ── anomaly ──
    ev = R["anomaly_events"].copy()
    evt = ev[["kind", "mold_id", "press_id", "split", "shots", "flag_rate", "template_alarm_rate", "minutes_to_flag",
              "minutes_to_template_alarm", "lead_over_response_min"]].copy()
    for c in ("flag_rate", "template_alarm_rate"):
        evt[c] = evt[c].map(pct)
    for c in ("minutes_to_flag", "minutes_to_template_alarm", "lead_over_response_min"):
        evt[c] = evt[c].map(lambda v: "" if v != v else f"{v:.0f}")
    evt.columns = ["Event", "Mold", "Press", "Period", "Shots", "Anomaly flag rate", "Template alarm rate", "Minutes to first flag",
                   "Minutes to first template alarm", "Flag lead over the shop's response (min)"]
    on = R["anomaly_im12"]
    f, ax = fig(2.8)
    ax.bar(on["shot_ts"], on["flag_rate"], color=[AMBER if m < "2025-10" else ACCENT for m in on["shot_ts"]])
    ax.set_ylabel("IM-12 anomaly flag rate")
    ax.tick_params(axis="x", rotation=60)
    im12_png = img(f, "IM-12 flag rate by month")
    an = R["anomaly"]

    lay = RS.layer_table(R)
    lay["share"] = lay["share"].map(lambda v: f"{v * 100:.1f}%")
    lay["alarms_per_shift"] = lay["alarms_per_shift"].map(lambda v: "" if v != v else f"{v:.2f}")
    lay.columns = ["Layer", "Share", "Alarms per shift"]
    bm = pd.DataFrame(L["by_mold"]).T.fillna(0)
    bm = bm.reindex(columns=[c for c in ["template_sort", "rules", "drift", "virtual_metrology", "anomaly", "none"] if c in bm.columns])
    bm = (bm * 100).round(1).reset_index().rename(columns={"index": "mold"})

    pts = R["spc_points"]
    drift = R["drift"]
    rs = R["lot_steps"]
    rs3 = rs.assign(rs3=rs["supplier_id"] == "RS-3").groupby("rs3")["fill_step"].mean()

    body = f"""
<div class="note">The data behind this report is generated by a physical model of the cell (section 1). It validates the pipeline, the label
design and the evaluation; it does not show how the models would transfer to another press, mold or plant.</div>

<h2 id="model">1. The curve model</h2>
<p>Every shot's cavity pressure curve is built from the shot's effective state, and every summary value, alarm, sort decision and defect follows
from the curve; no defect is written first and a curve fitted to it. Curves are built at 500 samples per second over the active part of the cycle
in six phases, summarized the way the monitoring unit summarizes them, and retained at 100 readings per second on one shot in
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

<h2 id="sensors">2. Sensor realism</h2>
<p>Each sensor carries a gain error of {CELL['sensor_gain_err'][0] * 100:.0f} to {CELL['sensor_gain_err'][1] * 100:.0f}%, an offset that drifts slowly,
AR(1) noise with autocorrelation {CELL['ar_integrals'][0]} to {CELL['ar_integrals'][1]} on integrals and {CELL['ar_timings'][0]} to
{CELL['ar_timings'][1]} on timings, dropouts on {pct(CELL['dropout_rate'], 1)} of shots and spikes on {pct(CELL['spike_rate'], 2)}.
Measured on the extract: {pct(RS.check(R, 'summary rows dropped')['value'], 2)} of summary rows dropped and a lag-1 autocorrelation of
{RS.check(R, 'lag-1 autocorrelation')['value']:.2f} on the pack integral's shot-to-shot residual.</p>

<h2 id="labels">3. Label design</h2>
<p>Labels come from the shop's own inspection processes over the true quality of each piece, with their coverage and noise: the reject-bin review
of every sorted shot, hourly audits of {LABELS['audit_pieces'][0]} to {LABELS['audit_pieces'][1]} pieces ({pct(R['audits']['linked'], 0)} linked
to a shot through the robot's recorded cycle), packing tallies by hour and code, and customer returns weeks later. A job's quality is taken as known
{LABELS['maturity_days']} days after its last shot. Confirmed defective pieces by code and source:</p>
{table(codes, {c: (lambda v: f'{v:,.0f}') for c in codes.columns if c != 'code'})}

<h2 id="checks">4. Realism checks</h2>
<p>{n_pass} of {len(ranged)} ranged checks fall inside their targets on this run; the others are listed with their values rather than tuned away.</p>
{table(ct)}

<h2 id="features">5. Feature pipeline and the as-of rule</h2>
<p>All three model components share one feature pipeline, built in dbt: summary values normalized to the template in force (deviation as a fraction
of the alarm band), machine-side values against the process window, and context joined as of the shot. Every join takes the latest record at or
before the shot, and unit tests recompute samples of the loads, setpoints, maintenance counters and templates to confirm no shot sees a later
record. Feature groups, used by the ablations: cavity ({len(CAVITY)}), machine side ({len(MACHINE)}), history and context ({len(HISTORY)}),
and control-chart and drift state ({len(STATE)}).</p>

<h2 id="spc">6. The SPC layer</h2>
<p>The unit's alarm logic, recomputed from summary values and templates, agrees with the recorded state on {pct(S['unit_agreement'], 2)} of production
shots (a dbt test). Individuals charts on five values per sensor take their limits from the first 500 shots after each approval. Sigma is the
baseline standard deviation: the moving-range estimate understates spread when shots are autocorrelated, and with it rule 1 fired on most points.
Even so, on every shot rule 1 fires on {pct(pts['we1'], 1)} of points, rule 2 on {pct(pts['we2'], 1)}, rule 4 on {pct(pts['we4'], 1)} and rule 5 on
{pct(pts['we5'], 1)}; only rules 1 and 2 are deployed. Drift detection (EWMA lambda 0.2; CUSUMs on 25-shot block means, k = 0.375, h = 5, reset at
approval, drift corrections, vent cleaning, PM, ring replacement and lot changes) raised these signal onsets:</p>
{table(drift)}
<p>Lot-change step tests on the fill integral: mean step {rs3.get(True, np.nan):+.3f} bands into RS-3 lots against {rs3.get(False, np.nan):+.3f} into others.</p>

<h2 id="vm">7. Virtual metrology</h2>
<p>XGBoost regression on audit pieces with shot linkage, trained through September 2025 with early stopping on October and November, evaluated
December 2025 to March 2026 over five seeds. Pooled model: weight RMSE {vmh[('pooled', 'weight')]['ratio']:.2f}x gauge
(sd {vmh[('pooled', 'weight')]['ratio_sd']:.3f} over seeds), dimension {vmh[('pooled', 'dimension')]['ratio']:.2f}x; per mold and press:
{vmh[('per_cell', 'weight')]['ratio']:.2f}x and {vmh[('per_cell', 'dimension')]['ratio']:.2f}x.</p>
{table(vs_t)}
<h3>Ablations</h3>
{table(ab)}
<p>Dropping the end-of-fill sensors (the post-gate-only set, as M-2260 runs) costs little; dropping the cavity signal costs most. Error by test month,
relative to December:</p>
{drift_png}

<h2 id="supervised">8. Supervised defect prediction</h2>
<p>Gradient boosting with class weighting on the labeled shots (sort-reviewed and robot-linked audits), compared at the template's alarm rate on the
labeled test set. Codes with fewer than 30 training or 10 test positives are not scored: {', '.join(c.replace('y_', '') for c in unusable) or 'none'}.</p>
{table(dtt)}
<p>Ablations on any defect: without the machine side, recall {da.loc['no_machine_side', 'recall_model']:.3f}; without the cavity signal,
{da.loc['no_cavity_signal', 'recall_model']:.3f}.</p>

<h2 id="anomaly">9. Anomaly detection</h2>
<p>Isolation forest per mold and press on each cavity value's change from the mean of the job's previous 300 shots, with the match
score and the spread between sensed cavities: a job running steadily off its template is left to the template and the charts, and the model
looks for a shot that stops looking like its run. Fitted on validated training shots, threshold at
{pct(0.0045, 2)} of the fit window. Test-period flag rate {pct(an['test_flag_rate'], 2)}; on validated test shots
{pct(an['false_positive_rate'], 2)}. {pct(an['share_on_template_pass'], 0)} of test flags fall on shots the template passed. The autoencoder on
retained curves flags {pct(an.get('ae_event_flag_rate', np.nan), 0)} of retained event shots against {pct(an.get('ae_nonevent_flag_rate', np.nan), 1)} of
the rest; the isolation forest flags {pct(an.get('if_event_flag_rate_retained', np.nan), 0)} of the same event shots. Event windows are taken
from quality engineering's event log.</p>
{table(evt)}
{im12_png}
<p class="caption">IM-12 flag rate by month; amber before the ring replacement in October 2025.</p>

<h2 id="layers">10. Layer comparison</h2>
{table(lay)}
<h3>By mold (percent of confirmed defects)</h3>
{table(bm)}
<p>{L['n_defects']:,} confirmed defective pieces in the test period, {L['n_dimensional']:,} of them dimensional.</p>

<h2 id="mechanisms">11. Mechanism recovery</h2>
<p>Each mechanism's signature, measured on the extract the way the platform sees it:</p>
{table(ct[ct['Check'].isin(['4', '10'])])}
"""
    toc = [("model", "Model"), ("sensors", "Sensors"), ("labels", "Labels"), ("checks", "Checks"), ("features", "Features"), ("spc", "SPC"),
           ("vm", "Virtual metrology"), ("supervised", "Supervised"), ("anomaly", "Anomaly"), ("layers", "Layers"), ("mechanisms", "Mechanisms")]
    OUT.write_text(shell("Cavity Pressure ML Technical Report", "Molding quality · IM-11 and IM-12",
                         f"{S['shots']:,.0f} shots, {S['jobs']} jobs · January 2025 to March 2026", body, toc), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
