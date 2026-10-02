"""
Cell-section additions to the quality dashboard: capability trend, Pareto charts with a selectable
period, false-reject rate over time, maintenance requests raised by drift signals, and monthly audit
trends. Every chart's caption states its finding.
"""
import json

import numpy as np
import pandas as pd

from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREY, LIGHT_BLUE, RED, fig, img, pct, table
from ml.src.features import connect

MOLD_COLORS = {"M-2041": "#2F4458", "M-2043": "#5F7F99", "M-2118": "#C2571A", "M-2119": "#E3A54D", "M-2260": "#5B8C3A", "M-2301": "#7B3FA0"}
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 12: 3.258, 15: 3.472, 20: 3.735}
PERIODS = [("last week", 1), ("last 4 weeks", 4), ("last 13 weeks", 13), ("all fifteen months", None)]


def q(sql):
    con = connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def d2(n):
    return D2[min(D2, key=lambda k: abs(k - n))]


def run_cpk(pieces, parts):
    """Cpk of the critical dimension per job from its audits (within-audit spread)."""
    rows = []
    for (job, mold), g in pieces.groupby(["job_id", "mold_id"]):
        p = parts.loc[mold]
        tol, nom = p["dimension_tolerance_mm"], p["critical_dimension_mm"]
        r = g.groupby("audit_id")["critical_dimension_mm"].agg(lambda v: v.max() - v.min() if len(v) > 1 else np.nan).dropna()
        n = g.groupby("audit_id").size().loc[r.index]
        if len(r) < 8:
            continue
        sw = (r / n.map(d2)).mean()
        mu = g["critical_dimension_mm"].mean()
        rows.append(dict(job_id=job, mold_id=mold, start=g["audit_ts"].min(), cpk=min(nom + tol - mu, mu - nom + tol) / (3 * sw), n=len(g)))
    return pd.DataFrame(rows)


def pareto_png(counts, title_unit):
    c = counts.sort_values(ascending=False).head(12)
    f, ax = fig(3.0, 8.0)
    ax.bar(range(len(c)), c.values, color=BRAND_BLUE)
    ax2 = ax.twinx()
    ax2.plot(range(len(c)), c.cumsum() / counts.sum() * 100, color=AMBER, marker="o", ms=3)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("cumulative %")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(range(len(c)), [str(i).replace("_", " ") for i in c.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(title_unit)
    return img(f, "pareto"), c


def build():
    parts = q("select * from stg_erp__part_attributes").set_index("mold_id")
    last = q("select max(shot_ts) from fct_shot").iloc[0, 0]
    out = []

    # ── capability trend ──
    pieces = q("""select p.audit_id, a.job_id, a.mold_id, a.audit_ts, p.dimension_1 as critical_dimension_mm, p.part_weight_g
                  from stg_qms__qc_audit_pieces p join stg_qms__qc_audits a using (audit_id)""")
    cp = run_cpk(pieces, parts)
    f, ax = fig(3.4)
    for m, g in cp.groupby("mold_id"):
        ax.plot(g["start"], g["cpk"], marker="o", ms=4, lw=1, color=MOLD_COLORS.get(m, GREY), label=m)
    ax.axhline(1.33, color=RED, lw=0.9, ls="--")
    ax.axhline(1.67, color="#1A7A3A", lw=0.9, ls="--")
    ax.set_ylabel("Cpk, critical dimension")
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    below = cp[cp["cpk"] < 1.33]
    worst_m = cp.groupby("mold_id")["cpk"].median().idxmin()
    out.append(("Capability trend", img(f, "capability trend"),
                f"{len(below)} of {len(cp)} runs fell below the 1.33 minimum on the critical dimension; {worst_m} has the lowest median Cpk "
                f"({cp[cp.mold_id == worst_m]['cpk'].median():.2f}). Weight has no drawing specification, so it is trended below rather than rated."))

    # ── Pareto charts with a selectable period ──
    alarms = q("""select s.shot_ts, s.sensor_id, s.sensor_position, s.alarm_values from stg_monitoring__shot_summary s
                  where s.alarm_state = 'alarm' and s.alarm_values is not null""")
    alarms = alarms.assign(key=alarms["alarm_values"].str.split(";")).explode("key")
    alarms["label"] = alarms["key"].str.replace("_bar", "").str.replace("_s", "").str.replace("_", " ") + " · " + alarms["sensor_position"].str.replace("_", "-")
    defects = q("select hour_ts, defect_code, qty from fct_confirmed_defects where source in ('sort', 'audit', 'tally') and hour_ts is not null")
    pareto_imgs, first_caps = {}, {}
    for name, weeks in PERIODS:
        start = pd.Timestamp(last) - pd.Timedelta(weeks=weeks) if weeks else pd.Timestamp("2000-01-01")
        a = alarms[alarms["shot_ts"] >= start]["label"].value_counts()
        dfx = defects[defects["hour_ts"] >= start].groupby("defect_code")["qty"].sum()
        pa_, ca = pareto_png(a, "alarm shots")
        pd_, cd = pareto_png(dfx, "confirmed defective pieces")
        pareto_imgs[name] = (pa_, pd_)
        first_caps[name] = (f"{ca.index[0]} leads with {pct(ca.iloc[0] / a.sum(), 0)} of alarms",
                            f"{cd.index[0].replace('_', ' ')} leads with {pct(cd.iloc[0] / dfx.sum(), 0)} of confirmed defects")
    opts = "".join(f'<option value="{i}"{" selected" if i == 1 else ""}>{n}</option>' for i, (n, _) in enumerate(PERIODS))
    panes = "".join(f'<div class="pp" data-i="{i}"{"" if i == 1 else " hidden"}><div class="ct">Alarms by triggering value and sensor</div>{pareto_imgs[n][0]}'
                    f'<p class="caption">{first_caps[n][0].capitalize()}.</p><div class="ct">Confirmed defects by code</div>{pareto_imgs[n][1]}'
                    f'<p class="caption">{first_caps[n][1].capitalize()}.</p></div>' for i, (n, _) in enumerate(PERIODS))
    pareto_html = (f'<label>Period: <select id="pareto-period">{opts}</select></label>{panes}'
                   '<script>(function(){var s=document.getElementById("pareto-period");function a(){document.querySelectorAll(".pp").forEach(function(d){'
                   'd.hidden=d.dataset.i!==s.value;});}s.addEventListener("change",a);a();})();</script>')
    out.append(("Pareto charts, weekly", pareto_html, None))

    # ── false-reject rate by mold over time ──
    rv = q("select mold_id, reviewed_ts, review_mode, pieces_reviewed, pieces_good from stg_qms__sort_dispositions")
    rv["month"] = rv["reviewed_ts"].dt.to_period("M").dt.start_time
    fr = rv.groupby(["month", "mold_id"])[["pieces_good", "pieces_reviewed"]].sum()
    fr = (fr["pieces_good"] / fr["pieces_reviewed"]).unstack()
    f, ax = fig(3.2)
    for m in fr.columns:
        ax.plot(fr.index, fr[m], marker="o", ms=3, lw=1.1, color=MOLD_COLORS.get(m, GREY), label=m)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_ylabel("sorted pieces found good")
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    tot = rv.groupby("mold_id")[["pieces_good", "pieces_reviewed"]].sum()
    tot = tot["pieces_good"] / tot["pieces_reviewed"]
    out.append(("False-reject rate by mold", img(f, "false rejects"),
                f"{tot.idxmax()} returns the largest share of sorted pieces as good ({pct(tot.max(), 0)} over the fifteen months), "
                f"{tot.idxmin()} the smallest ({pct(tot.min(), 0)})."))

    # ── maintenance requests from drift signals ──
    ds = q("""select d.detector, d.job_id, d.mold_id, d.press_id, d.shot_ts, f.mold_shot_no, f.press_shot_no
              from drift_signals d join int_shot_context f using (shot_id) where d.detector in ('cusum_eof', 'cusum_pack_var')""")
    ev = q("""select 'vent_cleaning' as kind, mold_id, null as press_id, event_ts from stg_toolroom__mold_maintenance where event_type = 'vent_cleaning'
              union all select 'check_ring', null, press_id, event_ts from stg_toolroom__equipment_service where equipment = 'check_ring'""")
    counters = q("select mold_id, press_id, shot_ts, mold_shot_no, press_shot_no from int_shot_context order by shot_ts")
    rows = []
    for r in ds.sort_values("shot_ts").itertuples():
        if r.detector == "cusum_eof":
            e = ev[(ev["kind"] == "vent_cleaning") & (ev["mold_id"] == r.mold_id) & (ev["event_ts"] > r.shot_ts)]
            what, scope = "vent cleaning", r.mold_id
        else:
            e = ev[(ev["kind"] == "check_ring") & (ev["press_id"] == r.press_id) & (ev["event_ts"] > r.shot_ts)]
            what, scope = "check-ring inspection", r.press_id
        done = e["event_ts"].min() if len(e) else pd.NaT
        if pd.notna(done):
            c = counters[(counters["mold_id"] == r.mold_id) if r.detector == "cusum_eof" else (counters["press_id"] == r.press_id)]
            n = int(((c["shot_ts"] > r.shot_ts) & (c["shot_ts"] <= done)).sum())
        else:
            n = None
        rows.append((r.shot_ts, what, scope, r.job_id, done, n))
    mr = pd.DataFrame(rows, columns=["raised", "request", "for", "job", "done", "shots"])
    first = mr.sort_values("raised").groupby(["request", "for", "done"], dropna=False).head(1)
    ft = first.assign(raised=first["raised"].dt.strftime("%d %b %Y"), done=first["done"].dt.strftime("%d %b %Y").fillna("not yet"),
                      shots=first["shots"].map(lambda v: "" if v is None or v != v else f"{v:,.0f}"))
    ft.columns = ["Signal raised", "Request", "Mold or press", "Job", "Maintenance done", "Shots between"]
    vent = first[(first["request"] == "vent cleaning") & first["shots"].notna()]
    out.append(("Maintenance requests from drift signals", table(ft.tail(25)),
                f"{len(first)} maintenance requests raised from vent and check-ring drift signals (first signal per maintenance interval, latest 25 shown); "
                + (f"vent cleaning followed a median {vent['shots'].median():,.0f} shots after the signal." if len(vent) else "")))

    # ── audit chart trends ──
    pieces["month"] = pieces["audit_ts"].dt.to_period("M").dt.start_time
    pieces["w_rel"] = pieces["part_weight_g"] / pieces["mold_id"].map(parts["nominal_weight_g"]) - 1
    pieces["d_rel"] = (pieces["critical_dimension_mm"] - pieces["mold_id"].map(parts["critical_dimension_mm"])) / pieces["mold_id"].map(parts["dimension_tolerance_mm"])
    t = pieces.groupby(["month", "mold_id"])[["w_rel", "d_rel"]].mean().reset_index()
    f, axes = fig(3.4, 9.4, ncols=2)
    for m, g in t.groupby("mold_id"):
        axes[0].plot(g["month"], g["w_rel"] * 100, color=MOLD_COLORS.get(m, GREY), lw=1.1, label=m)
        axes[1].plot(g["month"], g["d_rel"] * 100, color=MOLD_COLORS.get(m, GREY), lw=1.1)
    axes[0].set_ylabel("weight, % from nominal")
    axes[1].set_ylabel("critical dimension, % of tolerance")
    for a in axes:
        a.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    swing = t.groupby("mold_id")["d_rel"].agg(lambda v: v.max() - v.min())
    out.append(("Audit chart trends, monthly", img(f, "audit trends"),
                f"Monthly audit means stay close to nominal; {swing.idxmax()} swings most on the critical dimension "
                f"({swing.max() * 100:.0f}% of tolerance between its highest and lowest month)."))

    html = ""
    for title, body, cap in out:
        html += f"<h3>{title}</h3>{body}" + (f'<p class="caption">{cap}</p>' if cap else "")
    return html
