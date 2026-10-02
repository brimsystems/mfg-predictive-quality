"""
The monitoring unit's screen for one shift: what the process technician sees at the press.

Shift: job J-250165 (M-2119 on IM-12), shift C, 22:00 on 12 September to 06:00 on 13 September 2025.
A hold-pressure change at 01:11 with no reason recorded steps the pack integral up by most of a band;
the template alarms on several shots, but the pack EWMA stays below its limit, which is set from the run's
first 500 shots and so includes setup. The change is put back at 03:42, also unrecorded. The pack-integral
variability CUSUM fires at 04:48 because its 200-shot window spans both hold-pressure levels, not because
the check ring changed, and nothing reset it because neither change was logged as a correction.

Panels are tagged by source: the monitoring unit (template), SPC, or a model.

Usage: python ml/reports/generate_unit_screen.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports.style import fig, img  # noqa: E402
from ml.reports import results as RS  # noqa: E402
from ml.src.features import DATA_DIR, connect  # noqa: E402

OUT = Path(__file__).resolve().parent / "unit_screen.html"
JOB, T0 = "J-250165", pd.Timestamp("2025-09-12 22:00")
T1 = T0 + pd.Timedelta(hours=8)
SHIFT = "C"
C1, C4 = "#2E6FA7", "#C2571A"                 # cavity 1 and cavity 4, used everywhere on the screen
DRIFT_TINT = "#DCE6F0"                          # neutral: never yellow, which means warning
METRICS = [("cycle_integral", "Cycle integral (bar s)", "post_gate"), ("fill_integral", "Fill integral (bar s)", "post_gate"),
           ("pack_integral", "Pack integral (bar s)", "post_gate"), ("end_of_fill_pressure_bar", "End-of-fill pressure (bar)", "end_of_fill")]
GATE_SEAL = ("gate_seal_time_s", "Gate seal time (s)", "post_gate")
LABEL = {"fill_integral": "Fill integral", "pack_integral": "Pack integral", "peak_pressure_bar": "Peak pressure",
         "gate_seal_time_s": "Gate seal time", "end_of_fill_pressure_bar": "End-of-fill pressure", "cycle_integral": "Cycle integral"}


def q(sql):
    con = connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def sensor_name(sid, cav, pos):
    return f"{sid.split('-')[-1]} · cavity {int(cav)} · {pos.replace('_', '-')}"


# ── data ──────────────────────────────────────────────────────────────────
def load():
    D = {}
    D["shots"] = q(f"select * from fct_shot where job_id = '{JOB}' and shot_ts >= '{T0}' and shot_ts < '{T1}' order by shot_ts")
    D["run"] = q(f"select * from fct_shot where job_id = '{JOB}' and after_approval order by shot_ts")
    ids = ",".join(str(int(x)) for x in D["shots"]["shot_id"])
    D["sens"] = q(f"""select n.* from int_sensor_shot_normalized n where n.shot_id in ({ids}) order by n.shot_ts""")
    D["sensors"] = q(f"""select distinct sensor_id, sensor_position, cavity_no from stg_monitoring__shot_summary
                         where mold_id = (select mold_id from fct_job where job_id = '{JOB}') order by sensor_id""")
    D["spc"] = q(f"""select shot_id, sensor_id, metric, value, center, sigma, we1, we2 from spc_chart_points
                     where job_id = '{JOB}' and shot_ts >= '{T0}' and shot_ts < '{T1}'""")
    D["drift"] = q(f"""select d.shot_id, d.detector, d.shot_ts, f.cycle_no from drift_signals d join fct_shot f using (shot_id)
                       where d.job_id = '{JOB}' and d.shot_ts >= '{T0}' and d.shot_ts < '{T1}' order by d.shot_ts""")
    D["changes"] = q(f"""select change_ts, parameter, old_value, new_value, reason_code, technician_id from stg_qms__setpoint_changes
                         where job_id = '{JOB}' and change_ts >= '{T0}' and change_ts < '{T1}' and old_value is distinct from new_value
                         order by change_ts""")
    D["loads"] = q(f"""select load_ts, resin_lot_id, lag(resin_lot_id) over (order by load_ts) as prev_lot from stg_materials__material_loads
                       where job_id = '{JOB}' order by load_ts""")
    D["maint"] = q(f"""select event_ts, event_type from stg_toolroom__mold_maintenance
                       where mold_id = (select mold_id from fct_job where job_id = '{JOB}') and event_ts >= '{T0}' and event_ts < '{T1}'""")
    D["job"] = q(f"select * from fct_job where job_id = '{JOB}'").iloc[0]
    D["part"] = q(f"select * from stg_erp__part_attributes where mold_id = '{D['job'].mold_id}'").iloc[0]
    D["sheet"] = q(f"select revision from stg_engineering__process_sheets where mold_id = '{D['job'].mold_id}' and press_id = '{D['job'].press_id}'").iloc[0, 0]
    D["templates"] = q(f"""select * from stg_monitoring__templates where mold_id = '{D['job'].mold_id}' and press_id = '{D['job'].press_id}'
                           order by effective_from""")
    D["audits"] = q(f"""select audit_id, audit_ts, cavity_id, part_weight_g, critical_dimension_mm, shot_id from fct_audit_piece
                        where job_id = '{JOB}' and audit_ts < '{T1}' order by audit_ts""")
    return D


def anomaly_alarm_shots(shots):
    an = RS.anomaly_shots(shots["shot_id"])[["shot_id", "if_flag"]]
    f = shots[["shot_id", "shot_ts"]].merge(an, on="shot_id", how="left").sort_values("shot_ts")
    f["unusual"] = f["if_flag"].fillna(False).astype(bool)          # the alarm state: two flags in the last ten shots
    return f.set_index("shot_id")


def template_ref(template_id):
    """A retained shot from the window the template was set in (200 to 1,500 shots after approval), the one whose
    post-gate pack integral sits closest to the template value; and the template-era median cycle integral per sensor."""
    r = q(f"""with t as (select avg(pack_integral_template_value) as tv from stg_monitoring__templates where template_id = '{template_id}')
              select f.shot_id, abs(f.pg_pack_integral / (select tv from t) - 1) as d from fct_shot f
              where f.template_id = '{template_id}' and f.curve_retained and f.shots_since_approval between 200 and 1500
              order by d limit 1""")
    ci = q(f"""select n.sensor_id, median(n.cycle_integral) as ci from stg_monitoring__shot_summary n join fct_shot f using (shot_id)
               where f.template_id = '{template_id}' and f.shots_since_approval between 200 and 700 group by 1""")
    return (int(r["shot_id"].iloc[0]) if len(r) else None), ci.set_index("sensor_id")["ci"]


# ── drift episodes, classified against the setpoint log ───────────────────
def drift_episodes(D, cyc_of_ts):
    """Each detection on this shift as an episode with a label and shading. A detection within ten minutes after
    a setpoint change with no reason code is a step from that change, shaded while the change was in effect; a
    variability detection whose window spans a setpoint reversal is labeled as such; anything else is a slow drift,
    shaded lightly from its estimated start to detection and solidly from detection to the end of the shift."""
    eps = []
    ch = D["changes"]
    cyc_s = float(D["shots"]["cycle_time_s"].median())
    unrecorded = ch[ch["reason_code"].isna()]
    detected_changes = set()
    for r in D["drift"].itertuples():
        before = ch[(ch["change_ts"] <= r.shot_ts) & (ch["change_ts"] >= r.shot_ts - pd.Timedelta(minutes=10))]
        if r.detector in ("ewma_pack", "ewma_fill") and len(before):
            c = before.iloc[-1]
            detected_changes.add(c.change_ts)
            back = ch[(ch["change_ts"] > c.change_ts) & (ch["parameter"] == c.parameter) & (ch["new_value"] == c.old_value)]
            end_ts = back["change_ts"].min() if len(back) else T1
            what = "pack integral" if r.detector == "ewma_pack" else "fill integral"
            eps.append(dict(kind="step", detector=r.detector, det_cycle=r.cycle_no, det_ts=r.shot_ts,
                            start=cyc_of_ts(c.change_ts), end=cyc_of_ts(end_ts), metric=r.detector.split("_")[1] + "_integral",
                            label=f"Shift: {what} up after {c.parameter.replace('_bar', '').replace('_', ' ')} change",
                            log=f"Step change in the {what} after the {c.change_ts:%H:%M} {c.parameter.replace('_bar', '').replace('_', ' ')} change "
                                f"({c.old_value:.0f} to {c.new_value:.0f}); in effect until {end_ts:%H:%M}"))
        elif r.detector == "cusum_pack_var" and len(unrecorded[(unrecorded["change_ts"] <= r.shot_ts)
                                                                 & (unrecorded["change_ts"] >= r.shot_ts - pd.Timedelta(seconds=200 * cyc_s))]):
            eps.append(dict(kind="variability", detector=r.detector, det_cycle=r.cycle_no, det_ts=r.shot_ts,
                            start=r.cycle_no, end=r.cycle_no + 200, metric="pack_integral",
                            label="Variability signal following setpoint change; detector not reset because the change was unrecorded",
                            log="Pack-integral variability signal: its 200-shot window spans both hold-pressure levels. "
                                "Cushion and within-level spread unchanged, so not a check-ring signal. "
                                "Detector not reset because the change was unrecorded"))
        else:
            metric = {"ewma_pack": "pack_integral", "ewma_fill": "fill_integral", "cusum_eof": "end_of_fill_pressure_bar",
                      "cusum_gate_seal": "gate_seal_time_s", "cusum_pack_var": "pack_integral", "lot_step": "fill_integral"}[r.detector]
            eps.append(dict(kind="drift", detector=r.detector, det_cycle=r.cycle_no, det_ts=r.shot_ts, start=r.cycle_no - 150,
                            end=int(D["shots"]["cycle_no"].max()), metric=metric, label=f"Drift: {LABEL[metric].lower()}",
                            log=f"Drift onset: {r.detector.replace('_', ' ')}"))
    # a pack-moving setpoint change no detector caught: shaded while in effect, so the chart shows what the EWMA missed
    for c in unrecorded[unrecorded["parameter"] == "hold_pressure_bar"].itertuples():
        if c.change_ts in detected_changes or c.old_value != c.old_value:
            continue
        back = ch[(ch["change_ts"] > c.change_ts) & (ch["parameter"] == c.parameter) & (ch["new_value"] == c.old_value)]
        if not len(back):
            continue
        end_ts = back["change_ts"].min()
        n_al = int(((D["shots"]["shot_ts"] >= c.change_ts) & (D["shots"]["shot_ts"] < end_ts) & (D["shots"]["unit_alarm_state"] == "alarm")).sum())
        eps.append(dict(kind="step", detected=False, detector=None, det_cycle=cyc_of_ts(c.change_ts), det_ts=c.change_ts,
                        start=cyc_of_ts(c.change_ts), end=cyc_of_ts(end_ts), metric="pack_integral",
                        label="Pack integral up after hold pressure change (unrecorded); EWMA below its limit",
                        log=f"Hold pressure {c.old_value:.0f} to {c.new_value:.0f} bar at {c.change_ts:%H:%M} with no reason recorded; "
                            f"in effect until {end_ts:%H:%M}. Template alarms on {n_al} shots; the pack EWMA stays below its limit, "
                            f"which is set from the run's first 500 shots and so includes setup"))
    return eps


def events(D, cyc_of_ts):
    ev = []
    for c in D["changes"].itertuples():
        ev.append((cyc_of_ts(c.change_ts), c.change_ts, f"setpoint: {c.parameter.replace('_bar', '').replace('_', ' ')} {c.old_value:.0f} to {c.new_value:.0f} bar"))
    real = D["loads"][(D["loads"]["resin_lot_id"] != D["loads"]["prev_lot"]) & D["loads"]["prev_lot"].notna()]
    for l in real[(real["load_ts"] >= T0) & (real["load_ts"] < T1)].itertuples():
        ev.append((cyc_of_ts(l.load_ts), l.load_ts, f"lot change to {l.resin_lot_id}"))
    for m in D["maint"].itertuples():
        ev.append((cyc_of_ts(m.event_ts), m.event_ts, m.event_type.replace("_", " ")))
    for t in D["templates"].drop_duplicates("template_id").itertuples():
        if T0 <= t.effective_from < T1:
            ev.append((cyc_of_ts(t.effective_from), t.effective_from, "template re-established"))
    return sorted(ev)


# ── figures ───────────────────────────────────────────────────────────────
def cycle_graph(D, current_id, tmpl_ref, compare_refs):
    sens = D["sensors"]
    ids = [current_id, tmpl_ref] + compare_refs
    c = q(f"""select shot_id, sensor_id, t_ms / 1000.0 as t, pressure_bar as p from stg_monitoring__cavity_curves
              where mold_id = '{D['job'].mold_id}' and shot_id in ({','.join(str(int(i)) for i in ids)}) order by t_ms""")
    summ = D["sens"][D["sens"]["shot_id"] == current_id].set_index("sensor_id")
    imgs = []
    for compare in (False, True):
        f, ax = fig(3.9, 8.0)
        f.patch.set_facecolor("#10161d")
        ax.set_facecolor("#10161d")
        for sp in ax.spines.values():
            sp.set_color("#3a4654")
        ax.tick_params(colors="#9fb0c2")
        ax.yaxis.grid(True, color="#1f2a36")
        if compare:
            for k, rid in enumerate(compare_refs):
                for sid, g in c[c["shot_id"] == rid].groupby("sensor_id"):
                    ax.plot(g["t"], g["p"], color="#8fa3b8", lw=0.8, alpha=0.5, ls=(0, (3, 2)),
                            label="templates, last approved runs" if (k == 0 and sid == sens.sensor_id.iloc[0]) else None)
        for s in sens.itertuples():
            col = C1 if s.cavity_no == 1 else C4
            g = c[(c["shot_id"] == tmpl_ref) & (c["sensor_id"] == s.sensor_id)]
            ax.plot(g["t"], g["p"], color="#56657a", lw=3.2, alpha=0.75, label="template" if s.Index == 0 else None)
        for s in sens.itertuples():
            col = C1 if s.cavity_no == 1 else C4
            g = c[(c["shot_id"] == current_id) & (c["sensor_id"] == s.sensor_id)]
            ax.plot(g["t"], g["p"], color=col, lw=1.3, ls="-" if s.sensor_position == "post_gate" else (0, (4, 1.5)),
                    label=sensor_name(s.sensor_id, s.cavity_no, s.sensor_position))
        # annotations on cavity 1's sensors
        pg = sens[(sens.sensor_position == "post_gate") & (sens.cavity_no == 1)].iloc[0]
        eo = sens[(sens.sensor_position == "end_of_fill") & (sens.cavity_no == 1)].iloc[0]
        r = summ.loc[pg.sensor_id]
        g = c[(c["shot_id"] == current_id) & (c["sensor_id"] == pg.sensor_id)]
        t_arr, t_tr, t_gs = r.fill_time_to_sensor_s, r.time_to_peak_s - 0.02, r.gate_seal_time_s
        m = (g["t"] >= t_arr) & (g["t"] <= t_tr)
        ax.fill_between(g["t"][m], g["p"][m], color="#4f7fae", alpha=0.25, lw=0)
        m = (g["t"] >= t_tr) & (g["t"] <= t_gs)
        ax.fill_between(g["t"][m], g["p"][m], color="#7fa6cc", alpha=0.15, lw=0)
        ymax = float(c["p"].max()) * 1.18
        ax.annotate("fill integral", (t_arr - 0.1, ymax * 0.2), color="#c9d6e3", fontsize=8, ha="right")
        ax.annotate("pack integral", ((t_tr + t_gs) / 2, 40), color="#c9d6e3", fontsize=8, ha="center")
        ax.plot([t_tr, t_tr], [0, ymax * 0.06], color="#c9d6e3", lw=1.2)
        ax.annotate("transfer", (t_tr + 0.08, ymax * 0.02), color="#c9d6e3", fontsize=8, ha="left")
        ax.scatter([r.time_to_peak_s], [r.peak_pressure_bar], color="#ff5c5c", s=22, zorder=5)
        ax.annotate("peak pressure", (r.time_to_peak_s, r.peak_pressure_bar), xytext=(r.time_to_peak_s + 0.6, r.peak_pressure_bar + ymax * 0.05),
                    color="#c9d6e3", fontsize=8, arrowprops=dict(arrowstyle="-", color="#8fa3b8"))
        gs_p = float(g.loc[(g["t"] - t_gs).abs().idxmin(), "p"])
        ax.scatter([t_gs], [gs_p], color="#ff5c5c", s=22, zorder=5)
        ax.annotate("gate seal time", (t_gs, gs_p), xytext=(t_gs + 0.5, gs_p + ymax * 0.12), color="#c9d6e3", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#8fa3b8"))
        re = summ.loc[eo.sensor_id]
        ax.scatter([re.time_to_peak_s - 0.02], [re.end_of_fill_pressure_bar], color="#ffb300", s=22, zorder=5)
        ax.annotate("end-of-fill pressure", (re.time_to_peak_s - 0.02, re.end_of_fill_pressure_bar),
                    xytext=(re.time_to_peak_s + 1.5, re.end_of_fill_pressure_bar - ymax * 0.18), color="#c9d6e3", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#8fa3b8"))
        ax.set_xlim(0, t_gs + 4.5)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("time in cycle (s)", color="#9fb0c2")
        ax.set_ylabel("cavity pressure (bar)", color="#9fb0c2")
        leg = ax.legend(loc="upper right", frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color("#c9d6e3")
        imgs.append(img(f, "cycle graph"))
    return imgs


def stack(D, eps, evs, unusual, gate_seal):
    shots = D["shots"]
    cyc = shots.set_index("shot_id")["cycle_no"]
    sens = D["sensors"]
    rows = METRICS + ([GATE_SEAL] if gate_seal else [])
    tpl = D["sens"].groupby("sensor_id").last()
    f, axes = fig(1.9 * len(rows) + 0.6, 11.2, nrows=len(rows), sharex=True)
    x_all = shots["cycle_no"]
    for ax, (metric, label, pos) in zip(axes, rows):
        for cav, col in ((1, C1), (4, C4)):
            s = sens[(sens.sensor_position == pos) & (sens.cavity_no == cav)]
            if s.empty:
                continue
            sid = s.sensor_id.iloc[0]
            d = D["sens"][D["sens"]["sensor_id"] == sid].copy()
            d["cyc"] = d["shot_id"].map(cyc)
            ax.plot(d["cyc"], d[metric], color=col, lw=0.5, alpha=0.55)
            ax.plot(d["cyc"], d[metric].ewm(alpha=0.2, adjust=False).mean(), color=col, lw=1.6,
                    label=f"cavity {cav}: shots and EWMA" if metric == rows[0][0] else None)
            sp = D["spc"][(D["spc"]["sensor_id"] == sid) & (D["spc"]["metric"] == metric)]
            if len(sp):
                # the rules act on each shot's residual from what the previous shot predicts, so no fixed line on the raw
                # values corresponds to them; the markers show where they fired
                sp = sp.assign(cyc=sp["shot_id"].map(cyc))
                w1, w2 = sp[sp["we1"]], sp[sp["we2"] & ~sp["we1"]]
                ax.scatter(w1["cyc"], w1["value"], marker="x", s=26, color="#B00020", zorder=5,
                           label="rule 1: residual beyond 4.05 sigma" if metric == rows[0][0] and cav == 1 else None)
                ax.scatter(w2["cyc"], w2["value"], marker="^", s=18, facecolor="none", edgecolor="#B00020", zorder=5,
                           label="rule 2: two of three residuals beyond 2.70 sigma" if metric == rows[0][0] and cav == 1 else None)
            lo, hi = f"{metric}_alarm_low", f"{metric}_alarm_high"
            if lo in tpl.columns and pd.notna(tpl.loc[sid, lo]):
                for v in (tpl.loc[sid, lo], tpl.loc[sid, hi]):
                    ax.axhline(v, color="#999999", lw=0.7, alpha=0.6)
        for e in eps:
            if e["kind"] == "step":
                ax.axvspan(e["start"], e["end"], color=DRIFT_TINT, alpha=0.9, zorder=0)
            elif e["kind"] == "variability":
                ax.axvspan(e["start"], e["end"], color=DRIFT_TINT, alpha=0.45, zorder=0, hatch="//", edgecolor="#b8c8d8", lw=0)
            else:
                ax.axvspan(e["start"], e["det_cycle"], color=DRIFT_TINT, alpha=0.4, zorder=0)
                ax.axvspan(e["det_cycle"], e["end"], color=DRIFT_TINT, alpha=0.9, zorder=0)
            if e["metric"] == metric:
                y = 0.97 if e["kind"] != "variability" else 0.86
                txt = e["label"] if e["kind"] != "variability" else "Variability signal following setpoint change"
                ax.annotate("▼ " + txt, (e["det_cycle"], y), xycoords=("data", "axes fraction"), fontsize=7.5, color="#2F4458",
                            va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        for k, (cy, ts, name) in enumerate(evs):
            ax.axvline(cy, color="#555555", lw=0.8)
            if metric == rows[0][0]:
                ax.annotate(f"{ts:%H:%M} {name}", (cy, 1.04 + 0.13 * (k % 2)), xycoords=("data", "axes fraction"), fontsize=7.5,
                            ha="right" if k % 2 == 0 else "left", color="#333")
        if metric == "cycle_integral":
            u = shots[shots["shot_id"].map(unusual["unusual"]).fillna(False).astype(bool)]
            s1 = sens[(sens.sensor_position == "post_gate") & (sens.cavity_no == 1)].sensor_id.iloc[0]
            vals = D["sens"][D["sens"]["sensor_id"] == s1].set_index("shot_id")["cycle_integral"]
            ax.scatter(u["cycle_no"], u["shot_id"].map(vals), marker="o", s=22, facecolor="none", edgecolor="#7B3FA0", lw=1.2, zorder=6,
                       label="anomaly model: unusual")
        ax.set_ylabel(label + ("\nend-of-fill sensors" if pos == "end_of_fill" else "\npost-gate sensors"), fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("cycle (monitoring unit counter)")
    axes[-1].xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    h, l = [], []
    for a in axes:
        hh, ll = a.get_legend_handles_labels()
        h += hh
        l += ll
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h += [Line2D([], [], color="#999", lw=0.7), Patch(color=DRIFT_TINT)]
    l += ["alarm band edges (template)", "drift or shift interval"]
    axes[-1].legend(h, l, ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    return img(f, "shift chart")


def vm_trace(D, pred_dim):
    shots = D["shots"]
    pa = D["part"]
    tol = pa["dimension_tolerance_mm"]
    nom = pa["critical_dimension_mm"]
    p = shots[["shot_id", "cycle_no"]].merge(pred_dim.rename("pred"), left_on="shot_id", right_index=True)
    a = D["audits"][(D["audits"]["audit_ts"] >= T0) & D["audits"]["shot_id"].notna()]
    a = a.merge(shots[["shot_id", "cycle_no"]], on="shot_id")
    f, ax = fig(2.3, 11.2)
    ax.plot(p["cycle_no"], nom + p["pred"] * tol, color="#2F4458", lw=1.0, label="predicted critical dimension (mean of active cavities)")
    ax.scatter(a["cycle_no"], a["critical_dimension_mm"], s=16, color="#C2571A", zorder=5, label="hourly audit, gauged piece")
    for k, ls in ((1, "-"), (0.75, ":")):
        for sgn in (-1, 1):
            ax.axhline(nom + sgn * k * tol, color="#B00020" if k == 1 else "#999", lw=0.9, ls=ls)
    ax.set_ylabel("critical dimension (mm)", fontsize=8.5)
    ax.set_xlabel("cycle (monitoring unit counter)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    return img(f, "predicted dimension")


def balance(D):
    sens = D["sensors"]
    run = q(f"""select n.shot_id, n.sensor_id, n.shot_ts, n.pack_integral, n.end_of_fill_pressure_bar, f.cycle_no
                from int_sensor_shot_normalized n join fct_shot f using (shot_id) where f.job_id = '{JOB}' and f.after_approval""")
    out, caps = [], []
    f, axes = fig(2.6, 11.2, ncols=2)
    for ax, (metric, pos, label) in zip(axes, [("pack_integral", "post_gate", "Pack integral, cavity 1 minus cavity 4 (bar s)"),
                                               ("end_of_fill_pressure_bar", "end_of_fill", "End-of-fill pressure, cavity 1 minus cavity 4 (bar)")]):
        s1 = sens[(sens.sensor_position == pos) & (sens.cavity_no == 1)].sensor_id.iloc[0]
        s4 = sens[(sens.sensor_position == pos) & (sens.cavity_no == 4)].sensor_id.iloc[0]
        w = run.pivot_table(index=["shot_id", "cycle_no", "shot_ts"], columns="sensor_id", values=metric).reset_index()
        w["diff"] = w[s1] - w[s4]
        before = w[w["shot_ts"] < T0]["diff"].dropna()
        lo, hi = np.nanpercentile(before, [2.5, 97.5])
        sh = w[(w["shot_ts"] >= T0) & (w["shot_ts"] < T1)]
        ax.axhspan(lo, hi, color="#E8EEF4")
        ax.plot(sh["cycle_no"], sh["diff"], color="#2F4458", lw=0.6)
        ax.set_title(label, fontsize=9, fontweight="normal")
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        inside = float(((sh["diff"] >= lo) & (sh["diff"] <= hi)).mean())
        caps.append(f"{LABEL[metric].lower()}: mean difference {sh['diff'].mean():+.1f} this shift against {before.mean():+.1f} earlier in the run, "
                    f"{inside * 100:.0f}% of shots inside the run's normal range")
    return img(f, "cavity balance"), caps


# ── page ──────────────────────────────────────────────────────────────────
def main():
    D = load()
    shots, job, pa = D["shots"], D["job"], D["part"]
    cyc_of_ts = lambda ts: int(shots.loc[(shots["shot_ts"] - pd.Timestamp(ts)).abs().idxmin(), "cycle_no"])
    last = shots[shots["curve_retained"]].iloc[-1]          # the screen shows the latest shot whose curve the unit kept
    cur = int(last["shot_id"])
    unusual = anomaly_alarm_shots(D["run"])
    eps = drift_episodes(D, cyc_of_ts)
    evs = events(D, cyc_of_ts)

    tmpl = D["templates"][D["templates"]["template_id"] == last["template_id"]].iloc[0]
    approved = D["templates"].drop_duplicates("template_id")
    prev = approved[approved["effective_from"] < tmpl["effective_from"]].tail(4)
    compare_refs = [r for r in (template_ref(t)[0] for t in prev["template_id"]) if r is not None]
    ref_now, ci_tmpl = template_ref(tmpl["template_id"])
    appr = q(f"select cycle_no from fct_shot where job_id = '{JOB}' and shot_ts >= '{tmpl['effective_from']}' order by shot_ts limit 1").iloc[0, 0]
    lot = q(f"select resin_lot_id from stg_materials__material_loads where job_id = '{JOB}' and load_ts <= '{last['shot_ts']}' order by load_ts desc limit 1").iloc[0, 0]

    # status of the current shot
    state = last["unit_alarm_state"]
    status = "ALARM" if state == "alarm" else "WARNING" if state == "warning" else "OK"
    if bool(unusual["unusual"].get(cur, False)):
        status = "UNUSUAL" if status == "OK" else status
    open_eps = [e for e in eps if e["det_cycle"] <= last["cycle_no"] <= e["end"]]
    drift_txt = ""
    if open_eps:
        e = open_eps[-1]
        what = "pack-integral variability" if e["kind"] == "variability" else LABEL[e["metric"]].lower()
        drift_txt = f"Drift: {what}, post-gate" + (" (following setpoint change)" if e["kind"] == "variability" else "")

    # virtual metrology
    vp = RS.vm_shots(shots["shot_id"])                              # the rolling-origin model in force that month
    pw, pdm = vp.rename(columns={"weight_pred": "y"}), vp.rename(columns={"dim_pred": "y"})
    act = shots.set_index("shot_id")["active_cavities"]
    pw, pdm = pw[pw["cavity_id"] <= pw["shot_id"].map(act)], pdm[pdm["cavity_id"] <= pdm["shot_id"].map(act)]
    w_now = (1 + pw[pw["shot_id"] == cur]["y"].mean()) * pa["nominal_weight_g"]
    d_now_frac = pdm[pdm["shot_id"] == cur]["y"].mean()
    d_now = pa["critical_dimension_mm"] + d_now_frac * pa["dimension_tolerance_mm"]
    pred_dim_shot = pdm.groupby("shot_id")["y"].mean()
    pred_dim_max = pdm.assign(a=pdm["y"].abs()).groupby("shot_id")["a"].max()
    last_audit = D["audits"][D["audits"]["audit_ts"] <= last["shot_ts"]]
    la_ts = last_audit["audit_ts"].max()
    la = last_audit[last_audit["audit_ts"] == la_ts]
    audit_req = shots[shots["shot_id"].map(pred_dim_max) > 0.75]
    req_now = bool(pred_dim_max.get(cur, 0) > 0.75)

    # counters
    n_warn = int((shots["unit_alarm_state"] == "warning").sum())
    n_alarm = int((shots["unit_alarm_state"] == "alarm").sum())
    n_sort = int(shots["unit_sorted"].sum())

    # cycle graph and charts
    cg_off, cg_on = cycle_graph(D, cur, ref_now, compare_refs)
    st4 = stack(D, eps, evs, unusual, gate_seal=False)
    st5 = stack(D, eps, evs, unusual, gate_seal=True)
    vmt = vm_trace(D, pred_dim_shot)
    bal_png, bal_caps = balance(D)

    # summary table for the current shot
    srow = (D["sens"][D["sens"]["shot_id"] == cur].merge(D["sensors"], on=["sensor_id"], suffixes=("", "_s"))
            .merge(D["templates"].drop(columns=["mold_id", "press_id"]), on=["template_id", "sensor_id"], how="left", suffixes=("", "_t")))
    tpl_curve = ci_tmpl
    trs = ""
    for r in srow.sort_values(["cavity_no", "sensor_position"], ascending=[True, False]).itertuples():
        name = sensor_name(r.sensor_id, r.cavity_no, r.sensor_position)
        keys = ["fill_integral", "pack_integral", "peak_pressure_bar"] + (["gate_seal_time_s"] if r.sensor_position == "post_gate" else ["end_of_fill_pressure_bar"])
        trs += f'<tr class="grp"><td colspan="8">{name}</td></tr>'
        name = ""
        for key in keys:
            tv = getattr(r, f"{key}_template_value", None)
            if tv is None or tv != tv:
                continue
            v = getattr(r, key)
            lo, hi = getattr(r, f"{key}_alarm_low"), getattr(r, f"{key}_alarm_high")
            wl, wh = getattr(r, f"{key}_warning_low"), getattr(r, f"{key}_warning_high")
            pos = "ALARM" if (v < lo or v > hi) else ("WARN" if (v < wl or v > wh) else "OK")
            cls = {"ALARM": "alarm", "WARN": "warn", "OK": "ok"}[pos]
            x = (v - lo) / (hi - lo) * 100
            trs += (f'<tr><td>{name}</td><td>{LABEL[key]}</td><td class="n">{v:,.2f}</td><td class="n">{tv:,.2f}</td>'
                    f'<td class="n">{wl:,.2f} to {wh:,.2f}</td><td class="n">{lo:,.2f} to {hi:,.2f}</td>'
                    f'<td><div class="bar"><div class="wz" style="left:{(wl - lo) / (hi - lo) * 100:.0f}%;width:{(wh - wl) / (hi - lo) * 100:.0f}%"></div>'
                    f'<div class="mk {cls}" style="left:{min(max(x, 0), 100):.0f}%"></div></div></td><td><span class="st {cls}">{pos}</span></td></tr>')
        ci = r.cycle_integral
        ct = tpl_curve.get(r.sensor_id, np.nan)
        trs += (f'<tr><td>{name}</td><td>Cycle integral</td><td class="n">{ci:,.2f}</td><td class="n">{ct:,.2f}</td>'
                f'<td class="n muted">not banded</td><td class="n muted">not banded</td><td></td><td><span class="st muted">charted</span></td></tr>')

    # event log
    log = []
    sa = q(f"""select f.cycle_no, f.shot_ts, f.unit_alarm_state, f.unit_sorted, f.alarm_values, f.match_score_min from fct_shot f
               where f.job_id = '{JOB}' and f.shot_ts >= '{T0}' and f.shot_ts < '{T1}' and f.unit_alarm_state <> 'none' order by f.shot_ts""")
    for r in sa[sa["unit_alarm_state"] == "alarm"].itertuples():
        trig = r.alarm_values if isinstance(r.alarm_values, str) and r.alarm_values else (f"match score {r.match_score_min:.2f}" if r.match_score_min == r.match_score_min else "match score")
        trig = "; ".join(LABEL.get(t, t).lower() for t in trig.split(";")) if isinstance(r.alarm_values, str) and r.alarm_values else trig
        log.append((r.shot_ts, r.cycle_no, "Alarm, shot sorted" if r.unit_sorted else "Alarm", trig, "", ""))
    w = sa[sa["unit_alarm_state"] == "warning"]
    for h, g in w.groupby(w["shot_ts"].dt.floor("h")):
        log.append((h, int(g["cycle_no"].iloc[0]), f"Warnings this hour: {len(g)}", "values in warning band, not sorted", "", ""))
    for e in eps:
        log.append((e["det_ts"], e["det_cycle"], ("Shift detected (SPC)" if e.get("detected", True) else "Setpoint change, not detected (SPC)") if e["kind"] == "step" else "Variability signal (SPC)" if e["kind"] == "variability" else "Drift onset (SPC)",
                    e["log"], "", ""))
    for e in eps:
        if e["end"] <= int(shots["cycle_no"].max()):
            ts_end = shots.loc[(shots["cycle_no"] - e["end"]).abs().idxmin(), "shot_ts"]
            why = ("setpoint put back" if e["kind"] == "step" else "window no longer spans the two hold-pressure levels" if e["kind"] == "variability" else "reset")
            log.append((ts_end, e["end"], ("Shift cleared (SPC)" if e.get("detected", True) else "Setpoint put back") if e["kind"] == "step" else "Signal cleared (SPC)", why, "", ""))
    for c in D["changes"].itertuples():
        log.append((c.change_ts, cyc_of_ts(c.change_ts), "Setpoint change", f"{c.parameter.replace('_bar', '').replace('_', ' ')} {c.old_value:.0f} to {c.new_value:.0f} bar",
                    c.reason_code if isinstance(c.reason_code, str) else "", f"changed by {c.technician_id}"))
    u = shots[shots["shot_id"].map(unusual["unusual"]).fillna(False).astype(bool)]
    u_on = u[u["shot_id"].map(unusual["unusual"].shift(1).fillna(False).astype(bool).reindex(unusual.index)).fillna(False) == False]
    for r in u_on.itertuples():
        log.append((r.shot_ts, r.cycle_no, "Unusual (anomaly model)", "two of ten shots above threshold", "", ""))
    for r in audit_req.itertuples():
        log.append((r.shot_ts, r.cycle_no, "Audit requested (virtual metrology)", f"predicted dimension at {pred_dim_max[r.shot_id] * 100:.0f}% of tolerance", "", ""))
    for cy, ts, name in evs:
        if not name.startswith("setpoint"):
            log.append((ts, cy, name.capitalize(), "", "", ""))
    log.sort(key=lambda x: x[0])
    log_rows = "".join(f"<tr><td>{pd.Timestamp(t):%H:%M:%S}</td><td class='n'>{int(c):,}</td><td>{e}</td><td>{v}</td><td>{fd}</td><td>{ac}</td></tr>"
                       for t, c, e, v, fd, ac in log)

    stat_cls = {"OK": "none", "WARNING": "warning", "ALARM": "alarm", "UNUSUAL": "unusual"}[status]
    tol = pa["dimension_tolerance_mm"]
    d_in = abs(d_now - pa["critical_dimension_mm"]) <= tol
    w_dev = (w_now / pa["nominal_weight_g"] - 1) * 100
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unit Screen</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#0b1015;color:#dde6ee;font-family:"Segoe UI",Tahoma,sans-serif;font-size:13px}}
.top{{background:#16202b;border-bottom:2px solid #263444;padding:8px 16px}}
.row1{{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}
.id{{font-weight:700;font-size:15px}} .id span{{font-weight:400;color:#8fa3b8;margin-left:10px;font-size:12px}}
.ctx{{color:#8fa3b8;font-size:12px;margin-top:3px}}
.status{{display:flex;gap:8px;align-items:center}}
.lamp{{display:inline-block;padding:4px 12px;border-radius:3px;font-weight:700;letter-spacing:.6px}}
.lamp.none{{background:#1d6b3a}} .lamp.warning{{background:#9a6a12}} .lamp.alarm{{background:#a11b1b}} .lamp.unusual{{background:#5b2d7a}}
.drift{{border:1px solid #5d7a99;background:#1b2a3a;color:#cfe0f2;padding:3px 10px;border-radius:3px;font-size:12px}}
.grid{{display:grid;grid-template-columns:1.2fr 1fr;gap:12px;padding:12px 16px}} .grid1{{padding:0 16px 12px}}
tr.grp td{{color:#8fa3b8;font-weight:600;padding-top:8px;border-bottom:1px solid #2a3a4c}} .tw{{overflow-x:auto}}
.panel{{background:#121a23;border:1px solid #243140;padding:10px 12px;min-width:0}}
.panel h3{{margin:0 0 8px;font-size:12px;color:#8fa3b8;text-transform:uppercase;letter-spacing:1px;display:flex;justify-content:space-between;gap:8px}}
.src{{font-size:10px;letter-spacing:.5px;padding:1px 6px;border-radius:2px;text-transform:none;white-space:nowrap}}
.src.unit{{background:#2a3a4c;color:#c9d6e3}} .src.spc{{background:#2F4458;color:#dde6ee}} .src.model{{background:#7a4a0c;color:#fbe3c2}}
table{{border-collapse:collapse;width:100%}} td,th{{padding:4px 6px;border-bottom:1px solid #1f2b38;text-align:left}} th{{color:#8fa3b8;font-weight:600;font-size:11.5px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} td:first-child{{white-space:nowrap}} .muted{{color:#6f8296}}
.bar{{position:relative;height:10px;background:#3a1515;border-radius:2px;min-width:90px}} .wz{{position:absolute;top:0;bottom:0;background:#1d4d2e}}
.mk{{position:absolute;top:-3px;width:3px;height:16px;background:#fff}} .mk.warn{{background:#ffb300}} .mk.alarm{{background:#ff3b3b}}
.st{{font-weight:700;font-size:11px}} .st.ok{{color:#5fd38a}} .st.warn{{color:#ffb300}} .st.alarm{{color:#ff5c5c}}
.counters{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}} .ctr{{background:#0f161e;border:1px solid #243140;padding:8px}}
.ctr .v{{font-size:22px;font-weight:700}} .ctr .l{{font-size:11px;color:#8fa3b8}}
.vm{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}} .vm .ctr .v{{font-size:19px}}
.req{{margin-top:8px;padding:6px 10px;border:1px solid #9a6a12;color:#ffcf7a;border-radius:3px;font-size:12px}}
.light{{background:#F4F5F7;color:#222;padding:16px}} .light .panel{{background:white;border-color:#ddd;color:#222}}
.light h3{{color:#3D5166}} .light td,.light th{{border-color:#eee}} .light th{{color:#555}} img{{max-width:100%;display:block}}
.toggle{{font-size:12px;color:#8fa3b8;display:flex;gap:6px;align-items:center;margin:6px 0}} .light .toggle{{color:#555}}
.cap{{font-size:12px;color:#666;margin:6px 0 0}} .foot{{color:#8fa3b8;font-size:12px;padding:8px 16px 14px;line-height:1.5}}
[hidden]{{display:none!important}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}} .counters{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<div class="top">
 <div class="row1"><div class="id">{job['press_id']} · CPU-{job['press_id'][-2:]}<span>Mold {job['mold_id']} · {pa['description']} ({pa['part_id']}) · Job {JOB} · Shift {SHIFT} {T0:%d %b %Y} · Cycle {int(last['cycle_no']):,} (shot ID {cur})</span></div>
 <div class="status"><span class="lamp {stat_cls}">{status}</span>{f'<span class="drift">{drift_txt}</span>' if drift_txt else ''}</div></div>
 <div class="ctx">Resin lot {lot} · Template {tmpl['template_id']}, established at cycle {int(appr):,} on {pd.Timestamp(tmpl['effective_from']):%d %b %Y %H:%M} · Process sheet revision {D['sheet']}</div>
</div>
<div class="grid">
 <div class="panel"><h3>Cycle graph · cycle {int(last['cycle_no']):,} against its template (grey)<span class="src unit">Monitoring unit (template)</span></h3>
  <label class="toggle"><input type="checkbox" id="cmp"> Compare with last approved runs ({len(compare_refs)} templates)</label>
  <div id="cg-off">{cg_off}</div><div id="cg-on" hidden>{cg_on}</div></div>
 <div class="panel"><h3>Shift counters<span class="src unit">Monitoring unit (template)</span></h3><div class="counters">
   <div class="ctr"><div class="v">{len(shots):,}</div><div class="l">shots this shift</div></div>
   <div class="ctr"><div class="v">{n_warn}</div><div class="l">warning shots</div></div>
   <div class="ctr"><div class="v">{n_alarm}</div><div class="l">alarm shots</div></div>
   <div class="ctr"><div class="v">{n_sort}</div><div class="l">shots sorted</div></div></div>
  <h3 style="margin-top:14px">Predicted part quality, this shot<span class="src model">Model: virtual metrology</span></h3>
  <div class="vm">
   <div class="ctr"><div class="v">{w_now:,.3f} g</div><div class="l">predicted weight · nominal {pa['nominal_weight_g']:.2f} g ({w_dev:+.2f}%) · last audit {la['part_weight_g'].mean():,.3f} g at {pd.Timestamp(la_ts):%H:%M}</div></div>
   <div class="ctr"><div class="v" style="color:{'#5fd38a' if d_in else '#ff5c5c'}">{d_now:,.3f} mm</div><div class="l">predicted critical dimension · {pa['critical_dimension_mm']:.2f} ± {tol:.2f} mm, {'in tolerance' if d_in else 'out of tolerance'} ({d_now_frac * 100:+.0f}% of tolerance) · last audit {la['critical_dimension_mm'].mean():,.3f} mm at {pd.Timestamp(la_ts):%H:%M}</div></div>
  </div>
  {'<div class="req">Audit requested: predicted dimension past 75% of tolerance. Advisory only; never sorts a part.</div>' if req_now else f'<div class="cap" style="color:#8fa3b8">No audit request on this shot; {len(audit_req)} this shift (predicted dimension past 75% of tolerance, advisory).</div>'}
   </div>
</div>
<div class="grid1"><div class="panel"><h3>Summary values against template bands, cycle {int(last['cycle_no']):,}<span class="src unit">Monitoring unit (template)</span></h3>
  <div class="tw"><table><thead><tr><th></th><th>Summary value</th><th>This shot</th><th>Template</th><th>Warning band</th><th>Alarm band</th><th>Position in alarm band</th><th></th></tr></thead><tbody>{trs}</tbody></table></div>
</div></div>
<div class="light">
 <div class="panel"><h3>Shift chart · cavity 1 (blue) and cavity 4 (orange), shots and EWMA<span class="src spc">SPC</span></h3>
  <label class="toggle"><input type="checkbox" id="gs"> Add gate seal time</label>
  <div id="st4">{st4}</div><div id="st5" hidden>{st5}</div>
  <p class="cap">The pack integral steps up after the 01:11 hold-pressure change and comes back when it is put back at 03:42. The template alarms on the highest shots, but the EWMA stays below its limit, which includes the run's setup. The variability signal at 04:48 comes from a window that spans both levels; nothing reset it because neither change was logged.</p></div>
 <div class="panel" style="margin-top:12px"><h3>Predicted critical dimension and hourly audits<span class="src model">Model: virtual metrology</span></h3>{vmt}
  <p class="cap">The predicted dimension stays inside tolerance through the shift and tracks the gauged audit pieces; dotted lines mark 75% of tolerance, where an audit is requested.</p></div>
 <div class="panel" style="margin-top:12px"><h3>Cavity balance · cavity 1 minus cavity 4<span class="src unit">Monitoring unit (template)</span></h3>{bal_png}
  <p class="cap">Shaded: the run's normal range before this shift. {bal_caps[0].capitalize()}; {bal_caps[1]}.</p></div>
 <div class="panel" style="margin-top:12px"><h3>Event log (reaction log)<span class="src spc">Unit, SPC and models</span></h3>
  <table><thead><tr><th>Time</th><th>Cycle</th><th>Event</th><th>Value or rule</th><th>Technician finding</th><th>Action taken</th></tr></thead><tbody>{log_rows}</tbody></table></div>
</div>
<div class="foot">The process technician works from this screen, with the operator watching the status color. OK: nothing to do. WARNING: a value is in its
warning band; the shot is not sorted, and the technician watches the trend. ALARM: a value is outside its alarm band; the robot sorts the shot and the
technician investigates. UNUSUAL: the anomaly model sees a shot unlike the run's recent shots; the technician checks the press and the parts; nothing is
sorted. The drift indicator stays on while an SPC drift or shift signal is open. The model and SPC additions appear on the monitoring vendor's display where
it accepts additions, and on an MES companion view beside the press otherwise.</div>
<script>
(function(){{
  var cmp = document.getElementById('cmp'), gs = document.getElementById('gs');
  function apply(){{
    document.getElementById('cg-on').hidden = !cmp.checked; document.getElementById('cg-off').hidden = cmp.checked;
    document.getElementById('st5').hidden = !gs.checked; document.getElementById('st4').hidden = gs.checked;
  }}
  if (location.hash.indexOf('compare') >= 0) cmp.checked = true;
  if (location.hash.indexOf('gateseal') >= 0) gs.checked = true;
  cmp.addEventListener('change', apply); gs.addEventListener('change', apply); apply();
}})();
</script>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
