"""
Run report: one printable page per completed job, for the quality engineer at job close and the
customer record.

    from ml.reports.generate_run_report import run_report
    html = run_report("J-250165")

Usage: python ml/reports/generate_run_report.py [job_id ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREY, RED, fig, img, pct  # noqa: E402
from ml.reports import results as RS  # noqa: E402
from ml.reports.generate_unit_screen import template_ref  # noqa: E402
from ml.src.features import DATA_DIR, connect  # noqa: E402

HERE = Path(__file__).resolve().parent
PARAM = {"hold_pressure_bar": ("hold pressure", "bar"), "melt_temp_c": ("melt temperature", "C"), "mold_temp_c": ("mold temperature", "C"),
         "injection_velocity_mm_s": ("injection velocity", "mm/s"), "switchover_position_mm": ("switchover position", "mm"),
         "cycle_time_s": ("cycle time", "s")}


def setpoint_text(c):
    name, unit = PARAM.get(c.parameter, (c.parameter.replace("_", " "), ""))
    if c.old_value is None or c.old_value != c.old_value:
        return f"{name} set to {c.new_value:g} {unit} at changeover"
    return f"{name} {c.old_value:g} to {c.new_value:g} {unit}"


def codes_text(x):
    if not isinstance(x, str):
        return ""
    return ", ".join(f"{c.split(':')[0].replace('_', ' ')} {c.split(':')[1]}" for c in x.split(";"))
SNAPSHOT = pd.Timestamp("2026-04-30")
MEDICAL = ("M-2118", "M-2119")
C1, C4 = "#2E6FA7", "#C2571A"
CAV_COLORS = ["#2E6FA7", "#C2571A", "#5B8C3A", "#7B3FA0"]
REQ_MIN, REQ_TARGET = 1.33, 1.67
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 12: 3.258, 15: 3.472, 20: 3.735}
LABEL = {"cycle_integral": "Cycle integral (bar s)", "fill_integral": "Fill integral (bar s)", "pack_integral": "Pack integral (bar s)",
         "end_of_fill_pressure_bar": "End-of-fill pressure (bar)", "peak_pressure_bar": "Peak pressure (bar)", "gate_seal_time_s": "Gate seal time (s)"}


def q(sql):
    con = connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def d2(n):
    keys = sorted(D2)
    return D2[min(keys, key=lambda k: abs(k - n))]


def capability(x, lsl, usl, subgroup):
    """Cp and Cpk from within-subgroup spread (mean range over d2), Ppk from overall spread."""
    x = pd.Series(x).dropna()
    g = pd.DataFrame({"x": x.values, "g": pd.Series(subgroup).loc[x.index].values})
    r = g.groupby("g")["x"].agg(lambda v: v.max() - v.min() if len(v) > 1 else np.nan).dropna()
    n = g.groupby("g").size()
    sw = (r / n.loc[r.index].map(d2)).mean()
    so = x.std()
    mu = x.mean()
    return dict(n=len(x), subgroups=int(g["g"].nunique()), mean=mu, s_within=sw, s_overall=so,
                cp=(usl - lsl) / (6 * sw), cpk=min(usl - mu, mu - lsl) / (3 * sw), ppk=min(usl - mu, mu - lsl) / (3 * so))


def chart(title, image, caption, src):
    return (f'<div class="chart"><div class="ct">{title}<span class="src {src[0]}">{src[1]}</span></div>{image}'
            f'<p class="cap">{caption}</p></div>')


def tbl(df, num=()):
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    b = "".join("<tr>" + "".join(f'<td class="{"n" if c in num else ""}">{r[c]}</td>' for c in df.columns) + "</tr>" for _, r in df.iterrows())
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def run_report(job_id):
    job = q(f"select * from fct_job where job_id = '{job_id}'").iloc[0]
    mold, press = job["mold_id"], job["press_id"]
    pa = q(f"select * from stg_erp__part_attributes where mold_id = '{mold}'").iloc[0]
    shots = q(f"select * from fct_shot where job_id = '{job_id}' order by shot_ts")
    prod = shots[shots["after_approval"]]
    sheet = q(f"select revision from stg_engineering__process_sheets where mold_id = '{mold}' and press_id = '{press}'").iloc[0, 0]
    loads = q(f"select load_ts, resin_lot_id from stg_materials__material_loads where job_id = '{job_id}' order by load_ts")
    lots = list(dict.fromkeys(loads["resin_lot_id"]))
    sensors = q(f"select distinct sensor_id, sensor_position, cavity_no from stg_monitoring__shot_summary where mold_id = '{mold}' order by sensor_id")
    sens = q(f"""select n.* from int_sensor_shot_normalized n join fct_shot f using (shot_id)
                 where f.job_id = '{job_id}' and f.after_approval""")
    templates = q(f"select * from stg_monitoring__templates where mold_id = '{mold}' and press_id = '{press}' order by effective_from")
    run_tmpl = templates[templates["source_run_job_id"] == job_id]
    tids = list(dict.fromkeys(run_tmpl["template_id"]))
    confirmed = q(f"select source, defect_code, sum(qty) as qty from fct_confirmed_defects where job_id = '{job_id}' group by 1, 2")
    reviews = q(f"select * from stg_qms__sort_dispositions where job_id = '{job_id}' order by reviewed_ts")
    audits = q(f"""select a.audit_id, a.audit_ts, a.inspector_id, a.sample_size, a.sampled_shot_cycle_no, a.audit_rule_violations, a.disposition
                   from stg_qms__qc_audits a where a.job_id = '{job_id}' order by a.audit_ts""")
    pieces = q(f"""select p.audit_id, p.tray_row, p.cavity_id, p.part_weight_g, p.dimension_1 as critical_dimension_mm, p.visual_result, a.audit_ts
                   from stg_qms__qc_audit_pieces p join stg_qms__qc_audits a using (audit_id) where a.job_id = '{job_id}' order by a.audit_ts""")
    fsa = q(f"select * from stg_qms__first_shot_approvals where job_id = '{job_id}'")
    drift = q(f"""select d.detector, d.shot_ts, f.shots_since_approval from drift_signals d join fct_shot f using (shot_id)
                  where d.job_id = '{job_id}' order by d.shot_ts""")
    changes = q(f"""select change_ts, parameter, old_value, new_value, reason_code, technician_id from stg_qms__setpoint_changes
                    where job_id = '{job_id}' and old_value is distinct from new_value order by change_ts""")
    maint = q(f"""select event_ts, event_type from stg_toolroom__mold_maintenance where mold_id = '{mold}'
                  and event_ts >= '{shots['shot_ts'].min()}' and event_ts <= '{shots['shot_ts'].max()}'""")
    an = RS.anomaly_shots(prod["shot_id"]).set_index("shot_id")["if_flag"]
    prod["unusual"] = prod["shot_id"].map(an).fillna(False).astype(bool)   # the alarm state: two flags in the last ten shots
    unusual_on = prod["unusual"] & ~prod["unusual"].shift(1, fill_value=False)
    pdm = RS.vm_shots(prod["shot_id"]).rename(columns={"dim_pred": "y"})
    pdm = pdm[pdm["cavity_id"] <= pdm["shot_id"].map(prod.set_index("shot_id")["active_cavities"])]
    pdm_shot = pdm.groupby("shot_id")["y"].mean()
    pdm_max = pdm.assign(a=pdm["y"].abs()).groupby("shot_id")["a"].max()
    req = prod[prod["shot_id"].map(pdm_max) > 0.75]
    req_events = int(req.assign(h=req["shot_ts"].dt.floor("h")).drop_duplicates("h").shape[0])

    # ── 1. header ──
    pieces_produced = int(prod["active_cavities"].sum())
    medical = mold in MEDICAL
    hdr = pd.DataFrame([
        ("Job", job_id), ("Part", f"{pa['part_id']} · {pa['description']}"), ("Mold / press", f"{mold} / {press}"),
        ("Customer program", f"{pa['customer_id']} · {pa['program']}"), ("Resin lots", ", ".join(lots)),
        ("Start / end", f"{shots['shot_ts'].min():%d %b %Y %H:%M} to {shots['shot_ts'].max():%d %b %Y %H:%M}"),
        ("Shots (after approval)", f"{len(shots):,} ({len(prod):,})"), ("Pieces produced", f"{pieces_produced:,}"),
        ("Template version", ", ".join(tids) + (f" (re-established {len(tids) - 1} time{'s' if len(tids) > 2 else ''})" if len(tids) > 1 else "")),
        ("Process sheet", f"revision {sheet}")], columns=["Field", "Value"])

    # ── 2. run summary ──
    conf = confirmed[confirmed["source"].isin(["sort", "audit", "tally"])]
    defective = int(round(conf["qty"].sum()))
    sorted_pieces = int(reviews["pieces_reviewed"].sum())
    found_good = int(reviews["pieces_good"].sum())
    shipped = pieces_produced - defective - (sorted_pieces - found_good - int(round(conf[conf.source == 'sort']['qty'].sum())))
    by_code = conf.pivot_table(index="defect_code", columns="source", values="qty", aggfunc="sum", fill_value=0)
    by_code["total"] = by_code.sum(axis=1)
    by_code = by_code.sort_values("total", ascending=False).reset_index()
    by_code["defect_code"] = by_code["defect_code"].str.replace("_", " ")
    for c in by_code.columns[1:]:
        by_code[c] = by_code[c].map(lambda v: f"{v:,.0f}")
    by_code = by_code.rename(columns={"defect_code": "Code", "sort": "Sort review", "audit": "Audit", "tally": "Packing tally", "total": "Total"})
    returns = confirmed[confirmed["source"] == "return"]["qty"].sum()
    tol, nom = pa["dimension_tolerance_mm"], pa["critical_dimension_mm"]
    linked = pieces.copy()
    cap_dim = capability(linked["critical_dimension_mm"], nom - tol, nom + tol, linked["audit_id"])
    matured = bool(job["is_matured"])
    summary = pd.DataFrame([
        ("Good pieces shipped", f"{shipped:,}"),
        ("Shots sorted / pieces sorted", f"{int(prod['unit_sorted'].sum()):,} / {sorted_pieces:,}"),
        ("Sorted pieces found good at review", pct(found_good / max(sorted_pieces, 1), 1)),
        ("Confirmed defective pieces (sort, audit, packing)", f"{defective:,}"),
        ("Customer returns attributed so far", f"{returns:,.0f}"),
        ("Alarm shots / warning shots", f"{int((prod['unit_alarm_state'] == 'alarm').sum()):,} / {int((prod['unit_alarm_state'] == 'warning').sum()):,}"),
        ("Drift signals / anomaly flags", f"{len(drift)} / {int(unusual_on.sum())}"),
        ("Audits taken / audit requests from virtual metrology", f"{len(audits)} / {req_events}"),
        ("Capability, critical dimension (audits)", f"Cpk {cap_dim['cpk']:.2f}, Ppk {cap_dim['ppk']:.2f} on {cap_dim['n']} pieces"),
        ("Weight", "no weight specification on the drawing; reported as a process measure"),
    ], columns=["Measure", "This run"])
    maturity = (f"Quality as of {SNAPSHOT:%d %B %Y}. The job's last shot was {shots['shot_ts'].max():%d %B %Y}; "
                + ("its figures are matured (21 days have passed), " if matured else "its figures are provisional until 21 days after the last shot, ")
                + "and customer returns can still revise them.")

    # ── 3. first-shot approval ──
    if len(fsa):
        a = fsa.iloc[0]
        fs = pd.DataFrame([("Shots to approval", int(a["shots_to_approval"])), ("Approved at", f"{pd.Timestamp(a['approval_ts']):%d %b %Y %H:%M}"),
                           ("Weights measured (g)", str(a["part_weight_g"]).replace(";", ", ")),
                           ("Critical dimensions measured (mm)", str(a["dimension_values"]).replace(";", ", ")),
                           ("Visual result", a["visual_result"]), ("Approved by", f"technician {a['technician_id']}, inspector {a['inspector_id']}"),
                           ("Template established", a["template_id_established"])], columns=["Field", "Value"])
    else:
        fs = pd.DataFrame([("First-shot approval", "not recorded")], columns=["Field", "Value"])

    # ── 4. run-to-run overlay ──
    this_ref = template_ref(tids[0])[0]
    prior = templates.drop_duplicates("template_id")
    prior = prior[prior["effective_from"] < run_tmpl["effective_from"].min()].tail(5)
    ref_jobs = {}
    refs = []
    for t, jb in zip(prior["template_id"], prior["source_run_job_id"]):
        r_ = template_ref(t)[0]
        if r_ is not None:
            refs.append(r_)
            ref_jobs[r_] = jb
    pos_sensor = {p: sensors[(sensors.sensor_position == p)].sort_values("cavity_no").sensor_id.iloc[0]
                  for p in sensors["sensor_position"].unique()}
    cur = q(f"""select shot_id, sensor_id, t_ms / 1000.0 as t, pressure_bar as p from stg_monitoring__cavity_curves where mold_id = '{mold}'
                and shot_id in ({','.join(str(i) for i in [this_ref] + refs)}) and sensor_id in ({','.join(repr(s) for s in pos_sensor.values())})""")
    f, axes = fig(3.0, 9.0, ncols=len(pos_sensor))
    axes = np.atleast_1d(axes)
    diffs = []
    for ax, (p, sid) in zip(axes, pos_sensor.items()):
        for k, rid in enumerate(refs):
            g = cur[(cur.shot_id == rid) & (cur.sensor_id == sid)].sort_values("t")
            ax.plot(g["t"], g["p"], color=GREY, lw=0.9, alpha=0.8, label="last approved runs" if k == 0 else None)
        g = cur[(cur.shot_id == this_ref) & (cur.sensor_id == sid)].sort_values("t")
        ax.plot(g["t"], g["p"], color=BRAND_BLUE, lw=1.6, label="this run's template")
        ax.set_title(f"{sid} ({p.replace('_', '-')})", fontsize=10, fontweight="normal")
        ax.set_xlabel("time in cycle (s)")
        ax.set_xlim(0, float(g.loc[g["p"] > 0.1 * g["p"].max(), "t"].max()) + 1 if len(g) else None)
        ref_peaks = [cur[(cur.shot_id == r) & (cur.sensor_id == sid)]["p"].max() for r in refs]
        if ref_peaks and len(g):
            diffs.append((p, float(np.median([g["p"].max() / v - 1 for v in ref_peaks]))))
            if p == "post_gate":
                plat = {r: cur[(cur.shot_id == r) & (cur.sensor_id == sid) & (cur.t > 3) & (cur.t < 6)]["p"].median() for r in refs}
                this_plat = g[(g.t > 3) & (g.t < 6)]["p"].median()
                odd = {r: v / this_plat - 1 for r, v in plat.items() if abs(v / this_plat - 1) > 0.08}
    axes[0].set_ylabel("cavity pressure (bar)")
    axes[0].legend(frameon=False, fontsize=8)
    overlay_png = img(f, "run-to-run overlay")
    odd = locals().get("odd", {})
    diff_txt = "; ".join(f"{p.replace('_', '-')} peak {d * 100:+.1f}% against the median run" for p, d in diffs)
    overlay_cap = (f"This run's template sits close to the last {len(refs)} approved runs ({diff_txt})." if all(abs(d) < 0.05 for _, d in diffs)
                   else f"This run's template differs from the last {len(refs)} approved runs: {diff_txt}.")
    if odd:
        overlay_cap += " " + "; ".join(f"Run {ref_jobs[r]} held a pack plateau {v * 100:+.0f}% from this run's" for r, v in odd.items()) + "."

    # ── 5. distributions ──
    hist_metrics = [("cycle_integral", "post_gate"), ("fill_integral", "post_gate"), ("pack_integral", "post_gate"), ("end_of_fill_pressure_bar", "end_of_fill")]
    hist_metrics = [m for m in hist_metrics if m[1] in pos_sensor]
    f, axes = fig(4.6, 9.4, nrows=2, ncols=2)
    t0 = run_tmpl.drop_duplicates("sensor_id").set_index("sensor_id")
    out_band = []
    for ax, (metric, p) in zip(axes.ravel(), hist_metrics):
        for k, s in enumerate(sensors[sensors.sensor_position == p].itertuples()):
            v = sens[sens["sensor_id"] == s.sensor_id][metric].dropna()
            ax.hist(v, bins=60, histtype="step", lw=1.3, color=CAV_COLORS[k % 4], label=f"cavity {int(s.cavity_no)}")
            lo, hi = f"{metric}_alarm_low", f"{metric}_alarm_high"
            if lo in t0.columns and pd.notna(t0.loc[s.sensor_id, lo]):
                for col, ls in ((f"{metric}_warning_low", ":"), (f"{metric}_warning_high", ":"), (lo, "--"), (hi, "--")):
                    ax.axvline(t0.loc[s.sensor_id, col], color=CAV_COLORS[k % 4], lw=0.9, ls=ls, alpha=0.7)
                out_band.append((metric, int(((v < t0.loc[s.sensor_id, lo]) | (v > t0.loc[s.sensor_id, hi])).sum()), len(v)))
        ax.set_title(LABEL[metric] + (" (no band on the unit)" if metric == "cycle_integral" else ""), fontsize=9.5, fontweight="normal")
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    f.tight_layout(h_pad=2.5)
    hist_png = img(f, "distributions")
    worst = max(out_band, key=lambda x: x[1]) if out_band else ("", 0, 1)
    if not out_band:
        hist_cap = "Distributions of the run's summary values."
    elif worst[1]:
        hist_cap = ("Almost every shot sits inside the alarm bands (dashed; warning bands dotted); the value most often outside its band was the "
                    f"{LABEL[worst[0]].split(' (')[0].lower()}, on {worst[1]:,} of {worst[2]:,} shots on one sensor.")
    else:
        hist_cap = "Every shot sits inside the alarm bands (dashed; warning bands dotted)."

    # ── 6. timeline ──
    x = prod["shots_since_approval"]
    f, ax = fig(3.0, 9.6)
    rows = ["alarms and sorts", "warnings (per 100 shots)", "drift intervals", "anomaly flags", "events", "audits"]
    yrow = {r: len(rows) - i for i, r in enumerate(rows)}
    al = prod[prod["unit_alarm_state"] == "alarm"]
    ax.scatter(al["shots_since_approval"], np.full(len(al), yrow["alarms and sorts"]), marker="|", s=60,
               color=np.where(al["unit_sorted"], RED, AMBER), lw=0.8)
    wn = prod[prod["unit_alarm_state"] == "warning"]["shots_since_approval"]
    if len(wn):
        h, e = np.histogram(wn, bins=np.arange(0, x.max() + 100, 100))
        ax.bar(e[:-1], h / max(h.max(), 1) * 0.7, bottom=yrow["warnings (per 100 shots)"] - 0.35, width=100, align="edge", color=AMBER, alpha=0.6)
    end = x.max()
    for r in drift.itertuples():
        ax.plot([r.shots_since_approval, min(end, r.shots_since_approval + 400)], [yrow["drift intervals"]] * 2, color=ACCENT, lw=5, solid_capstyle="butt")
    ua = prod[unusual_on]
    ax.scatter(ua["shots_since_approval"], np.full(len(ua), yrow["anomaly flags"]), marker="o", s=14, facecolor="none", edgecolor="#7B3FA0")
    snum = lambda ts: int(prod.loc[(prod["shot_ts"] - pd.Timestamp(ts)).abs().idxmin(), "shots_since_approval"])
    ev = [(snum(c.change_ts), "s") for c in changes.itertuples()] + [(snum(m.event_ts), "m") for m in maint.itertuples()]
    real = loads.assign(prev=loads["resin_lot_id"].shift())
    real = real[(real["resin_lot_id"] != real["prev"]) & real["prev"].notna()]
    ev += [(snum(l.load_ts), "l") for l in real.itertuples()]
    ev += [(snum(t), "t") for t in run_tmpl.drop_duplicates("template_id")["effective_from"].iloc[1:]]
    marks = {"s": ("v", "setpoint change"), "m": ("s", "maintenance"), "l": ("D", "lot change"), "t": ("*", "template re-established")}
    for kind, (mk, lab) in marks.items():
        xs = [e for e, k in ev if k == kind]
        if xs:
            ax.scatter(xs, [yrow["events"]] * len(xs), marker=mk, s=28, color=BRAND_BLUE, label=lab)
    au = audits.assign(s=[snum(t) for t in audits["audit_ts"]])
    ax.scatter(au["s"], [yrow["audits"]] * len(au), marker="|", s=40, color=GREY)
    ax.set_yticks(list(yrow.values()), list(yrow.keys()), fontsize=8.5)
    ax.set_xlabel("shots after first-shot approval")
    ax.set_xlim(0, end)
    if ev:
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    timeline_png = img(f, "run timeline")
    timeline_cap = (f"{len(al):,} alarm shots ({int(prod['unit_sorted'].sum()):,} sorted), {len(drift)} drift signals and {int(unusual_on.sum())} "
                    f"anomaly flags over {len(prod):,} production shots; red ticks are sorted shots.")

    # ── 7. audit charts ──
    grp = pieces.groupby("audit_id").agg(ts=("audit_ts", "first"), w=("part_weight_g", "mean"), wr=("part_weight_g", lambda v: v.max() - v.min()),
                                         d=("critical_dimension_mm", "mean"), dr=("critical_dimension_mm", lambda v: v.max() - v.min()),
                                         n=("part_weight_g", "size")).sort_values("ts").reset_index()
    base = grp.head(25)
    nbar = base["n"].mean()
    A2 = 3 / (d2(nbar) * np.sqrt(nbar))
    D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}.get(int(round(min(max(nbar, 2), 10))), 1.777)
    hourly = prod.assign(h=prod["shot_ts"].dt.floor("h"), pd_=prod["shot_id"].map(pdm_shot)).groupby("h")["pd_"].mean()
    f, axes = fig(5.2, 9.6, nrows=4, sharex=True)
    viol = 0
    for (col, rcol, unit), axm, axr in ((("w", "wr", "weight (g)"), axes[0], axes[1]), (("d", "dr", "dimension (mm)"), axes[2], axes[3])):
        xb, rb = base[col].mean(), base[rcol].mean()
        if col == "d":
            axm.plot(hourly.index, nom + hourly.values * tol, color="#B9C7D6", lw=2.2, label="predicted dimension, hourly mean (virtual metrology)", zorder=0)
        axm.plot(grp["ts"], grp[col], marker="o", ms=3, color=BRAND_BLUE, lw=0.8, label="audit mean")
        if col == "d":
            span = max(A2 * rb * 3, (grp[col].max() - grp[col].min()) * 0.75)
            axm.set_ylim(xb - span, xb + span)
            axm.annotate(f"tolerance {nom - tol:.2f} to {nom + tol:.2f} mm, outside this view", (0.01, 0.04), xycoords="axes fraction",
                         fontsize=7.5, color="#666")
        for v in (xb + A2 * rb, xb - A2 * rb):
            axm.axhline(v, color=BRAND_BLUE, ls="--", lw=0.8)
        axm.axhline(xb, color="#888", lw=0.7)
        viol += int(((grp[col] > xb + A2 * rb) | (grp[col] < xb - A2 * rb)).sum())
        axm.set_ylabel(f"X-bar, {unit}", fontsize=8.5)
        axr.plot(grp["ts"], grp[rcol], marker="o", ms=3, color=ACCENT, lw=0.8)
        axr.axhline(rb, color="#888", lw=0.7)
        axr.axhline(D4 * rb, color=ACCENT, ls="--", lw=0.8)
        axr.set_ylabel(f"R, {unit}", fontsize=8.5)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    for a_ in axes:
        a_.tick_params(labelsize=8)
    audit_png = img(f, "audit charts")
    audit_cap = (f"{len(grp)} hourly audits; {viol} audit means fall outside the X-bar limits set from the first 25. "
                 f"The predicted dimension (grey band) follows the audit means between samples.")

    # ── 8. capability ──
    cap_rows = []
    c = cap_dim
    status = "meets target" if c["cpk"] >= REQ_TARGET else "meets minimum" if c["cpk"] >= REQ_MIN else "below minimum"
    cap_rows.append(("Critical dimension", f"{nom:.2f} ± {tol:.2f} mm", c["n"], c["subgroups"], f"{c['cp']:.2f}", f"{c['cpk']:.2f}", f"{c['ppk']:.2f}", status))
    wcap = linked["part_weight_g"].describe()
    cap_rows.append(("Part weight", "no specification on the drawing", int(wcap["count"]), int(grp.shape[0]), "", "", "",
                     f"mean {wcap['mean']:.3f} g, sd {wcap['std']:.3f} g"))
    cap = pd.DataFrame(cap_rows, columns=["Characteristic", "Specification", "Pieces", "Audits", "Cp", "Cpk", "Ppk", "Against 1.33 / 1.67"])

    # ── 9. cavity balance ──
    bal = []
    for s in sensors.itertuples():
        v = sens[sens["sensor_id"] == s.sensor_id]
        for metric in (["fill_integral", "pack_integral", "cycle_integral"] + (["end_of_fill_pressure_bar"] if s.sensor_position == "end_of_fill" else ["gate_seal_time_s"])):
            bal.append((f"cavity {int(s.cavity_no)} · {s.sensor_position.replace('_', '-')}", LABEL[metric], f"{v[metric].mean():,.2f}", f"{v[metric].std():,.2f}"))
    bal = pd.DataFrame(bal, columns=["Sensed cavity", "Value", "Mean", "Standard deviation"])

    # ── 10. reject-bin review ──
    if medical:
        rv = reviews.merge(shots[["shot_id", "cycle_no"]], on="shot_id", how="left")
        rvt = pd.DataFrame({"Reviewed": rv["reviewed_ts"].dt.strftime("%d %b %H:%M"), "Cycle (shot ID)": [f"{int(c):,} ({int(s)})" for c, s in zip(rv["cycle_no"], rv["shot_id"])],
                            "Pieces reviewed": rv["pieces_reviewed"], "Defective": rv["pieces_confirmed_defective"],
                            "Codes": rv["defect_codes"].map(codes_text),
                            "Good": rv["pieces_good"], "Inspector": rv["inspector_id"]})
        review_intro = (f"{len(rv)} sorted shots, each reviewed from its slot of the indexed reject tray; "
                        f"{pct((rv['pieces_confirmed_defective'] == 0).mean(), 0)} of them held no defective piece.")
    else:
        rvt = pd.DataFrame({"Shift": [f"{d} {s}" for d, s in zip(reviews["shift_date"].astype(str), reviews["shift"])],
                            "Reviewed": reviews["reviewed_ts"].dt.strftime("%d %b %H:%M"), "Shots sorted": reviews["sorted_shots"],
                            "Pieces reviewed": reviews["pieces_reviewed"], "Defective by code": reviews["defect_codes"].map(codes_text),
                            "Good": reviews["pieces_good"], "Inspector": reviews["inspector_id"]})
        review_intro = (f"The reject bin is reviewed once per shift: {len(reviews)} reviews of {sorted_pieces:,} pieces, "
                        f"{pct(found_good / max(sorted_pieces, 1), 0)} found good. Reviews carry the job and shift, not the shot.")

    # ── 11. reaction log ──
    rl = []
    real_changes = changes[changes["old_value"].notna()]
    for r in drift.itertuples():
        nxt = real_changes[(real_changes["change_ts"] >= r.shot_ts) & (real_changes["change_ts"] <= r.shot_ts + pd.Timedelta(hours=2))]
        prev = real_changes[(real_changes["change_ts"] <= r.shot_ts) & (real_changes["change_ts"] >= r.shot_ts - pd.Timedelta(minutes=10))]
        if len(prev) and r.detector in ("ewma_pack", "ewma_fill"):
            c = prev.iloc[-1]
            which = "pack" if r.detector == "ewma_pack" else "fill"
            way = "up" if c.new_value > c.old_value else "down"
            event = f"Shift: {which} integral {way} after {PARAM.get(c.parameter, (c.parameter, ''))[0]} change"
            finding = f"step from the {c.change_ts:%H:%M} setpoint change, not a slow drift"
        elif len(prev) and r.detector == "cusum_pack_var":
            event = "Variability signal following setpoint change; detector not reset because the change had no reason recorded"
            finding = "window spans both setpoint levels; cushion unchanged, so not a check-ring signal"
        else:
            event = "drift signal: " + r.detector.replace("_", " ")
            finding = ""
        action = f"{setpoint_text(nxt.iloc[0])} at {nxt['change_ts'].iloc[0]:%H:%M}" if len(nxt) else ""
        rl.append((r.shot_ts, event, finding, action))
    alh = prod[prod["unit_alarm_state"] == "alarm"].groupby(prod["shot_ts"].dt.floor("D"))
    for d, g in alh:
        vals = g["alarm_values"].fillna("match score").str.split(";").explode().value_counts()
        lab = vals.index[0].replace("_bar", "").replace("_s", "").replace("_", " ")
        plural = "s" if len(g) != 1 else ""
        rl.append((d, f"alarms that day: {len(g)} shot{plural}, mostly {lab}", "", ""))
    for c in changes.itertuples():
        reason = c.reason_code.replace("_", " ") if isinstance(c.reason_code, str) else "no reason recorded"
        rl.append((c.change_ts, "setpoint: " + setpoint_text(c), reason, f"by {c.technician_id}"))
    rl.sort(key=lambda x: x[0])
    rlt = pd.DataFrame([(f"{pd.Timestamp(t):%d %b %H:%M}", e, fd, ac) for t, e, fd, ac in rl], columns=["Time", "Event", "Technician finding", "Action taken"])

    css = """
*{box-sizing:border-box} body{margin:0;background:#F4F5F7;color:#222;font-family:"Segoe UI",Helvetica,Arial,sans-serif;font-size:13.5px;line-height:1.5}
.page{max-width:1000px;margin:0 auto;background:white;padding:28px 40px 48px}
.hdr{background:#3D5166;color:white;padding:16px 40px} .hdr h1{margin:0;font-size:21px} .hdr .sub{opacity:.85;font-size:12.5px}
h2{font-size:16px;color:#3D5166;border-bottom:2px solid #3D5166;padding-bottom:4px;margin:26px 0 10px;display:flex;justify-content:space-between;align-items:baseline;gap:8px}
h2 .n{font-size:10px;letter-spacing:2px;color:#6B8FA8;text-transform:uppercase;margin-right:8px}
.tw{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
th{background:#F7F8FA;text-align:left;padding:6px 8px;font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:#666;border-bottom:2px solid #EEE}
td{padding:5px 8px;border-bottom:1px solid #F0F0F0} td.n{text-align:right;font-variant-numeric:tabular-nums}
.chart{border:1px solid #EEE;border-radius:4px;padding:10px;margin:10px 0} .ct{font-weight:700;font-size:13.5px;display:flex;justify-content:space-between;gap:8px}
.cap{font-size:12px;color:#555;margin:6px 0 0} img{max-width:100%;display:block;margin:0 auto}
.src{font-size:10px;padding:1px 6px;border-radius:2px;font-weight:600;white-space:nowrap} .src.unit{background:#E6EBF1;color:#3D5166}
.src.spc{background:#3D5166;color:white} .src.model{background:#B86E12;color:white} .src.qms{background:#EEE;color:#555}
.note{font-size:12px;color:#666;background:#F7F9FB;border-left:3px solid #6B8FA8;padding:8px 12px;margin:8px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:800px){.two{grid-template-columns:1fr} .page{padding:16px}}
@media print{body{background:white} .page{max-width:none;padding:0} .chart,table{break-inside:avoid} h2{break-after:avoid}}
"""
    sec = lambda n, t, src: f'<h2><span><span class="n">{n}</span>{t}</span><span class="src {src[0]}">{src[1]}</span></h2>'
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Report {job_id}</title><style>{css}</style></head><body>
<div class="hdr"><h1>Run Report · {job_id} · {pa['description']}</h1><div class="sub">{mold} on {press} · {shots['shot_ts'].min():%d %b} to {shots['shot_ts'].max():%d %b %Y} · reject-bin review {'per sorted shot (indexed tray)' if medical else 'per shift'}</div></div>
<div class="page">
{sec("Section 1", "Job", ("qms", "MES, ERP, QMS"))}{tbl(hdr)}
{sec("Section 2", "Run summary", ("qms", "QMS, unit, SPC, models"))}
<div class="two"><div>{tbl(summary)}</div><div>{tbl(by_code, num=tuple(by_code.columns[1:]))}</div></div>
<div class="note">{maturity}</div>
{sec("Section 3", "First-shot approval", ("qms", "QMS"))}{tbl(fs)}
{sec("Section 4", "Run-to-run curve overlay", ("unit", "Monitoring unit (template)"))}
{chart("This run's template against the last approved runs", overlay_png, overlay_cap, ("unit", "Monitoring unit (template)"))}
{sec("Section 5", "Summary value distributions", ("unit", "Monitoring unit (template)"))}
{chart("Summary values per sensed cavity, with the template's bands", hist_png, hist_cap, ("unit", "Monitoring unit (template)"))}
{sec("Section 6", "Run timeline", ("spc", "Unit, SPC and models"))}
{chart("How the run went", timeline_png, timeline_cap, ("spc", "Unit, SPC and models"))}
{sec("Section 7", "Audit charts", ("spc", "SPC"))}
{chart("X-bar and R from the hourly audits", audit_png, audit_cap, ("spc", "SPC, with virtual metrology behind the dimension"))}
{sec("Section 8", "Capability", ("spc", "SPC"))}
{tbl(cap, num=("Pieces", "Audits", "Cp", "Cpk", "Ppk"))}
<div class="note">Cp and Cpk use the within-audit spread (mean range over d2); Ppk uses the overall spread of all audited pieces. The customer requires
1.33 minimum and targets 1.67 unless the program specifies otherwise.</div>
{sec("Section 9", "Cavity balance", ("unit", "Monitoring unit (template)"))}{tbl(bal, num=("Mean", "Standard deviation"))}
{sec("Section 10", "Reject-bin review", ("qms", "QMS"))}<p>{review_intro}</p>{tbl(rvt, num=("Pieces reviewed", "Defective", "Good", "Shots sorted"))}
{sec("Section 11", "Reaction log", ("spc", "SPC and the setpoint log"))}{tbl(rlt)}
</div></body></html>"""
    return html


def main(jobs):
    for j in jobs:
        out = HERE / f"run_report_{j}.html"
        out.write_text(run_report(j), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["J-250165"])
