"""
ML overview for the quality manager.

The story, in order: what the shop's existing process control already catches, what
extending that SPC added without machine learning, what the models added on top, and
where they fell short. Every number is read from the current run.

Usage: python ml/reports/generate_ml_overview.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports import results as RS  # noqa: E402
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREY, LIGHT_BLUE, RED, fig, img  # noqa: E402
from ml.src.features import VALID_END  # noqa: E402

OUT = Path(__file__).resolve().parent / "ml_overview.html"
SPC_COLORS = {"template_sort": "#2F4458", "rules": "#5F7F99", "drift": "#9CB5C9"}
ML_COLORS = {"virtual_metrology": "#B86E12", "anomaly": "#E3A54D"}
NONE_COLOR = "#E4E4E4"
LAYER_NAMES = {"template_sort": "Template alarms and sort", "rules": "Control-chart rules 1 and 2",
               "drift": "Drift detection", "virtual_metrology": "Virtual metrology", "anomaly": "Anomaly detection",
               "none": "Caught by none"}
MEDICAL = ("M-2118", "M-2119")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pts(x):
    return f"{x * 100:.1f}"


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  background: #FFFFFF; color: #222222; font-size: 16px; line-height: 1.7; }
.page-header { background: #3D5166; color: white; padding: 20px 40px; }
.page-header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
.page-header .sub { font-size: 13px; opacity: 0.8; margin-top: 2px; }
.layout { display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }
.toc { width: 220px; flex-shrink: 0; padding: 40px 20px 40px 0; position: sticky; top: 0; height: 100vh;
  overflow-y: auto; border-right: 1px solid #EEEEEE; }
.toc-title { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #AAAAAA; margin-bottom: 14px; font-weight: 600; }
.toc a { display: block; font-size: 13px; color: #666; text-decoration: none; padding: 4px 0 4px 10px;
  border-left: 2px solid transparent; line-height: 1.4; }
.toc a:hover { color: #3D5166; border-left-color: #3D5166; }
.toc a.sub { font-size: 12px; padding-left: 20px; color: #AAAAAA; }
.toc a.sub:hover { color: #3D5166; border-left-color: #3D5166; }
.toc hr { border: none; border-top: 1px solid #EEEEEE; margin: 8px 0; }
.content { flex: 1; padding: 40px 0 80px 52px; max-width: 880px; min-width: 0; }
.section-title-block { margin: 48px 0 24px 0; padding-bottom: 12px; border-bottom: 2px solid #3D5166; }
.content > .section-title-block:first-child { margin-top: 12px; }
.section-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #3D5166; font-weight: 600; margin-bottom: 4px; }
.section-title { font-size: 22px; font-weight: 700; color: #222222; }
.section-title-block.sub { margin: 34px 0 14px 0; padding-bottom: 0; border-bottom: none; border-left: 3px solid #6B8FA8; padding-left: 12px; }
.section-title-block.sub .section-label { color: #888; margin-bottom: 2px; }
.section-title-block.sub .section-title { font-size: 16px; font-weight: 600; letter-spacing: 0.2px; }
.section-title-block.ml { border-bottom-color: #B86E12; }
.section-title-block.ml .section-label { color: #B86E12; }
.section-title-block.sub.ml { border-left-color: #E3A54D; }
p { margin-bottom: 16px; color: #333; font-size: 16px; }
ul.plain { margin: 4px 0 16px 22px; } ul.plain li { margin-bottom: 6px; font-size: 15px; color: #333; }
.kpi-row { display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }
.kpi-card { flex: 1; min-width: 160px; border: 1px solid #DDDDDD; border-radius: 8px; padding: 18px 20px 14px 20px;
  background: white; border-bottom: 4px solid #3D5166; }
.kpi-card.ml { border-bottom-color: #B86E12; }
.kpi-value { font-size: 28px; font-weight: 700; line-height: 1; margin-bottom: 6px; color: #3D5166; }
.kpi-card.ml .kpi-value { color: #B86E12; }
.kpi-label { font-size: 12px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-sub { font-size: 13px; color: #999; margin-top: 4px; }
.chart-title { font-size: 16px; font-weight: 700; color: #222222; text-align: center; margin-bottom: 6px; }
.chart-wrap { margin: 20px 0 6px; border: 1px solid #EEEEEE; border-radius: 4px; padding: 12px; }
.chart-wrap img { max-width: 100%; display: block; margin: 0 auto; }
.caption { font-size: 13px; color: #666; margin: 4px 0 22px; }
.table-wrap { overflow-x: auto; margin: 16px 0; }
.data-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.data-table th { background: #F7F8FA; padding: 10px 12px; text-align: left; font-size: 12px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px; color: #666; border-bottom: 2px solid #EEEEEE; }
.data-table td { padding: 9px 12px; border-bottom: 1px solid #F0F0F0; color: #333; vertical-align: top; }
.data-table td:first-child { white-space: nowrap; }
.data-table td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.data-table tr.total td { font-weight: 700; border-top: 2px solid #EEEEEE; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px; font-weight: 700;
  letter-spacing: 0.5px; text-transform: uppercase; color: white; }
.tag.spc { background: #3D5166; } .tag.ml { background: #B86E12; }
.box { border: 1px solid #DDE3EA; background: #F7F9FB; border-radius: 6px; padding: 16px 20px; margin: 20px 0; }
.box h4 { font-size: 13px; letter-spacing: 1px; text-transform: uppercase; color: #3D5166; margin-bottom: 8px; }
.box p, .box li { font-size: 15px; } .box ul { margin-left: 20px; }
.note { font-size: 13px; color: #777; margin-top: -8px; margin-bottom: 16px; }
@media (max-width: 900px) { .layout { padding: 0 16px; } .toc { display: none; } .content { padding: 24px 0 60px 0; }
  .page-header { padding: 18px 16px; } }
"""


def section(sid, label, title, kind="spc", sub=False):
    cls = "section-title-block" + (" sub" if sub else "") + (" ml" if kind == "ml" else "")
    return f'<div class="{cls}" id="{sid}"><div class="section-label">{label}</div><h2 class="section-title">{title}</h2></div>'


def chart(title, image, caption):
    return f'<div class="chart-wrap"><div class="chart-title">{title}</div>{image}</div><p class="caption">{caption}</p>'


def table(df, num_cols=(), total_last=False):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for i, (_, r) in enumerate(df.iterrows()):
        cls = ' class="total"' if total_last and i == len(df) - 1 else ""
        rows += f"<tr{cls}>" + "".join(f'<td class="{"num" if c in num_cols else ""}">{r[c]}</td>' for c in df.columns) + "</tr>"
    return f'<div class="table-wrap"><table class="data-table"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def kpi(value, label, sub, ml=False):
    return f'<div class="kpi-card{" ml" if ml else ""}"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div><div class="kpi-sub">{sub}</div></div>'


# ── figures ───────────────────────────────────────────────────────────────
def coverage_bar(share, budget):
    order = ["template_sort", "rules", "drift", "virtual_metrology", "anomaly", "none"]
    colors = {**SPC_COLORS, **ML_COLORS, "none": NONE_COLOR}
    f, ax = fig(3.0, 8.6)
    left = 0.0
    for k in order:
        v = share.get(k, 0.0)
        ax.barh([0], [v], left=left, color=colors[k], edgecolor="white", height=0.55)
        if v >= 0.03:
            ax.text(left + v / 2, 0, f"{v * 100:.0f}" if k != "template_sort" else f"{v * 100:.0f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white" if k not in ("drift", "none", "anomaly") else "#222")
        left += v
    spc = sum(share.get(k, 0) for k in ("template_sort", "rules", "drift"))
    ax.axvline(spc, color="#2F4458", lw=1.2, ls="--")
    ax.text(spc, 0.42, f"SPC subtotal {spc * 100:.1f}%", ha="right", va="bottom", fontsize=9.5, color="#2F4458")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 0.6)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.spines["left"].set_visible(False)
    handles = [ax.barh([0], [0], color=colors[k])[0] for k in order]
    labels = []
    for k in order:
        a = budget.get(k)
        al = "" if k == "none" or a is None else f"   {a:.2f} alarms/shift"
        val = f"{share.get(k, 0) * 100:.1f}%" if k in ("template_sort", "none") else f"+{share.get(k, 0) * 100:.1f} pts"
        labels.append(f"{LAYER_NAMES[k]}: {val}{al}")
    ax.legend(handles, labels, ncol=2, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return img(f, "coverage by layer")


def curve_figure():
    s = RS.q("""
        select f.shot_id, f.template_id from fct_shot f
        where f.mold_id = 'M-2118' and f.curve_retained and f.after_approval and f.unit_alarm_state = 'none'
          and f.shots_since_approval > 2000 order by f.shot_id limit 1""").iloc[0]
    ref = int(RS.q(f"select min(template_curve_ref) from stg_monitoring__templates where template_id = '{s.template_id}'").iloc[0, 0])
    summ = RS.q(f"""select sensor_id, sensor_position, fill_time_to_sensor_s, time_to_peak_s, gate_seal_time_s, peak_pressure_bar,
                    pressure_at_transfer_bar from stg_monitoring__shot_summary where shot_id = {int(s.shot_id)}""")
    pg = summ[summ["sensor_position"] == "post_gate"].sort_values("sensor_id").iloc[0]
    eo = summ[summ["sensor_position"] == "end_of_fill"].sort_values("sensor_id").iloc[0]
    c = RS.q(f"""select shot_id, sensor_id, t_ms / 1000.0 as t, pressure_bar as p from stg_monitoring__cavity_curves
                 where mold_id = 'M-2118' and shot_id in ({int(s.shot_id)}, {ref})
                 and sensor_id in ('{pg.sensor_id}', '{eo.sensor_id}') order by t_ms""")
    cur = c[(c["shot_id"] == s.shot_id) & (c["sensor_id"] == pg.sensor_id)]
    tpl = c[(c["shot_id"] == ref) & (c["sensor_id"] == pg.sensor_id)]
    ce = c[(c["shot_id"] == s.shot_id) & (c["sensor_id"] == eo.sensor_id)]
    t_arr, t_tr, t_gs = pg.fill_time_to_sensor_s, pg.time_to_peak_s - 0.02, pg.gate_seal_time_s
    f, ax = fig(4.4, 8.6)
    ax.fill_between(cur["t"], cur["p"], color="#EEF2F6", lw=0, label="cycle integral (whole area)")
    m = (cur["t"] >= t_arr) & (cur["t"] <= t_tr)
    ax.fill_between(cur["t"][m], cur["p"][m], color="#9CB5C9", lw=0, label="fill integral")
    m = (cur["t"] >= t_tr) & (cur["t"] <= t_gs)
    ax.fill_between(cur["t"][m], cur["p"][m], color="#C9D6E2", lw=0, label="pack integral")
    ax.plot(tpl["t"], tpl["p"], color=GREY, lw=3.2, alpha=0.6, label="template (reference shot)")
    ax.plot(cur["t"], cur["p"], color="#2F4458", lw=1.4, label="this shot, post-gate sensor")
    ax.plot(ce["t"], ce["p"], color=AMBER, lw=1.3, label="this shot, end-of-fill sensor")
    ax.scatter([pg.time_to_peak_s], [pg.peak_pressure_bar], color=RED, zorder=5, s=28)
    ax.annotate("peak pressure", (pg.time_to_peak_s, pg.peak_pressure_bar), xytext=(1.9, pg.peak_pressure_bar + 30),
                fontsize=9, arrowprops=dict(arrowstyle="-", color="#888"))
    gs_p = float(cur.loc[(cur["t"] - t_gs).abs().idxmin(), "p"])
    ax.scatter([t_gs], [gs_p], color=RED, zorder=5, s=28)
    ax.annotate("gate seal time", (t_gs, gs_p), xytext=(t_gs + 0.6, gs_p + 80), fontsize=9, arrowprops=dict(arrowstyle="-", color="#888"))
    ax.scatter([eo.time_to_peak_s - 0.02], [eo.pressure_at_transfer_bar], color=AMBER, edgecolor="#7A4A0C", zorder=5, s=28)
    ax.annotate("end-of-fill pressure", (eo.time_to_peak_s - 0.02, eo.pressure_at_transfer_bar),
                xytext=(2.6, eo.pressure_at_transfer_bar - 95), fontsize=9, arrowprops=dict(arrowstyle="-", color="#888"))
    ymax = float(cur["p"].max()) * 1.32
    for x, y, name in [((t_arr + t_tr) / 2, 0.95, "fill"), (t_tr + 0.15, 0.86, "transfer"), ((t_tr + t_gs) / 2, 0.95, "pack"),
                       (t_gs + 2.4, 0.95, "gate seal and cooling")]:
        ax.annotate(name, (x, ymax * y), ha="center" if name != "transfer" else "left", fontsize=9, color="#555")
    for x in (t_arr, t_tr, t_gs):
        ax.axvline(x, color="#BBBBBB", lw=0.8, ls=":")
    ax.set_xlim(0, t_gs + 5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time in cycle (s)")
    ax.set_ylabel("cavity pressure (bar)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    return img(f, "annotated cavity pressure curve")


def drift_example():
    """A vent CUSUM signal after a gradual end-of-fill decline, followed by a rise in burns tallied at packing."""
    sig = RS.q("""select d.job_id, d.mold_id, d.shot_ts from drift_signals d join fct_shot f using (shot_id)
                  where d.detector = 'cusum_eof' and f.shots_since_approval > 3000 order by d.shot_ts""")
    burns = RS.q("""select job_id, hour_ts, sum(qty) as burns from fct_confirmed_defects
                    where defect_code = 'burn' and source = 'tally' group by 1, 2 order by 2""")
    cands = []
    for r in sig.itertuples():
        b = burns[(burns["job_id"] == r.job_id) & (burns["hour_ts"] >= r.shot_ts)]
        hit = b[b["burns"] >= 3]
        if not len(hit) or (hit["hour_ts"].min() - r.shot_ts).total_seconds() < 2 * 3600:
            continue
        sh = RS.q(f"select shot_ts, eof_pressure_dev from fct_shot where job_id = '{r.job_id}' and after_approval order by shot_ts")
        k = int(sh["shot_ts"].searchsorted(r.shot_ts))
        if k < 3000:
            continue
        early, late = sh["eof_pressure_dev"].iloc[k - 3000:k - 1500].mean(), sh["eof_pressure_dev"].iloc[k - 500:k].mean()
        step = abs(sh["eof_pressure_dev"].iloc[k:k + 100].mean() - late)
        if late - early < -0.08 and step < 0.15:
            cands.append((late - early, r, hit["hour_ts"].min(), sh, k))
    cands.sort(key=lambda c: c[0])
    _, best, t_rise, s, k_sig = cands[len(cands) // 2]
    s = s.reset_index(drop=True)
    b = burns[burns["job_id"] == best.job_id].copy()
    b["shot_no"] = s["shot_ts"].searchsorted(b["hour_ts"])
    f, ax = fig(3.3)
    ax.plot(s.index, s["eof_pressure_dev"].rolling(50, min_periods=10).mean(), color=BRAND_BLUE, lw=1.3,
            label="end-of-fill pressure, 50-shot mean (bands from template)")
    ax.axvline(k_sig, color=RED, lw=1.2, ls="--", label="CUSUM signal")
    ax2 = ax.twinx()
    ax2.bar(b["shot_no"], b["burns"], width=max(len(s) / 250, 20), color=AMBER, alpha=0.8, label="burns tallied in the hour")
    ax2.spines["top"].set_visible(False)
    ax.set_xlim(1000, len(s))
    ax.set_xlabel("shots into the run (after the first 1,000, once setup has settled)")
    ax.set_ylabel("deviation from template (bands)")
    ax2.set_ylabel("burns per hour")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    lead_h = (t_rise - best.shot_ts).total_seconds() / 3600
    return best, img(f, "vent drift"), lead_h


def vm_scatter(pred):
    p = pred[(pred["design"] == "pooled") & (pred["target"] == "dimension")]
    p = p.assign(meas=(p["measured"] - p["nominal"]) / p["tolerance"], pr=(p["predicted"] - p["nominal"]) / p["tolerance"])
    f, ax = fig(3.6, 5.0)
    ax.scatter(p["pr"], p["meas"], s=2, alpha=0.15, color=ACCENT)
    ax.plot([-1, 1], [-1, 1], color=BRAND_BLUE, lw=1)
    for y in (-1, 1):
        ax.axhline(y, color=RED, lw=0.8, ls="--")
    ax.set_xlabel("predicted, fraction of tolerance")
    ax.set_ylabel("gauged, fraction of tolerance")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    return img(f, "virtual metrology")


def supervised_figure(full, audit):
    f, ax = fig(3.0, 7.0)
    x = np.arange(2)
    ax.bar(x - 0.2, [full["recall_template"], audit["recall_template"]], 0.38, color=GREY, label="template limits")
    ax.bar(x + 0.2, [full["recall_model"], audit["recall_model"]], 0.38, color="#B86E12", label="supervised model")
    for xi, (a, b) in enumerate([(full["recall_template"], full["recall_model"]), (audit["recall_template"], audit["recall_model"])]):
        ax.text(xi - 0.2, a + 0.01, f"{a * 100:.0f}%", ha="center", fontsize=9)
        ax.text(xi + 0.2, b + 0.01, f"{b * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x, ["all shot-linked labels\n(mostly found by the sort)", "audit-found defects only\n(pulled on a clock)"])
    ax.set_ylabel("recall at the template's alarm rate")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper right")
    return img(f, "supervised comparison")


def anomaly_figure(ev):
    e = ev[ev["split"] == "test"].reset_index(drop=True)
    f, ax = fig(2.8, 7.0)
    x = np.arange(len(e))
    ax.bar(x - 0.2, e["minutes_to_flag"].fillna(0), 0.38, color="#B86E12", label="anomaly model, first flag")
    ax.bar(x + 0.2, e["minutes_to_template_alarm"].fillna(0), 0.38, color=GREY, label="template, first alarm")
    ax.set_xticks(x, [k.replace("_", " ") for k in e["kind"]])
    ax.set_ylabel("minutes from event start")
    ax.legend(frameon=False)
    return img(f, "anomaly events")


def budget_figure(budget):
    names = [("rules", "Rules 1 and 2"), ("drift", "Drift onsets"), ("anomaly", "Anomaly (2 in 10)"), ("virtual_metrology", "Virtual metrology")]
    f, ax = fig(2.8, 7.0)
    vals = [budget.get(k, 0) for k, _ in names]
    ax.bar([n for _, n in names], vals, color=[SPC_COLORS["rules"], SPC_COLORS["drift"], ML_COLORS["anomaly"], ML_COLORS["virtual_metrology"]])
    ax.axhline(budget["added_layers"], color="#2F4458", lw=0, label="")
    ax.set_ylabel("alarms per shift")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    return img(f, "alarm budget")


# ── page ──────────────────────────────────────────────────────────────────
def main():
    R = RS.load()
    X = json.loads((RS.RES / "overview_extras.json").read_text())
    L = R["layers"]
    share, budget = L["share"], L["alarms_per_shift"]
    S = R["shots"]
    vmh = RS.vm_headline(R)
    dt = RS.defect_table(R)
    any_ = dt[dt["target"] == "y_any"].iloc[0]
    ao = X["audit_only"]
    an = R["anomaly"]
    ev = R["anomaly_events"]
    ev_test = ev[ev["split"] == "test"]
    n_ev, n_ev_flag = len(ev_test), int(ev_test["detected"].sum())
    pts_spc = sum(share.get(k, 0) for k in ("template_sort", "rules", "drift"))
    spc_alarms = budget["rules"] + budget["drift"]
    ml_alarms = budget["anomaly"] + budget["virtual_metrology"]
    added = budget["added_layers"]
    total = budget["total_with_template"]
    link = RS.linkage(R)
    sl = L["shot_level"]
    summer = X["summer_layers"]
    bt = X["summer_backtest"]

    parts = R["parts"].set_index("mold_id")
    sort_m = RS.q("""
        select r.mold_id, any_value(r.review_mode) as review_mode, sum(r.sorted_shots) as sorted_shots,
               sum(r.pieces_good) / sum(r.pieces_reviewed) as found_good, sum(r.pieces_reviewed) as sorted_pieces,
               sum(r.pieces_good) as good_pieces, sum(r.pieces_good * p.standard_cost) as good_cost
        from stg_qms__sort_dispositions r join stg_erp__part_attributes p on p.mold_id = r.mold_id
        group by 1 order by 1""")
    prod = RS.q("select mold_id, count(*) as shots from fct_shot where after_approval group by 1")
    sort_m = sort_m.merge(prod, on="mold_id")
    good_pieces, good_cost = int(sort_m["good_pieces"].sum()), float(sort_m["good_cost"].sum())
    fr = R["sort"]["pieces_found_good_share"]
    fr_tray = R["sort"]["false_reject_share"]
    sorted_share = S["sorted_shots"] / S["prod_shots"]
    pts_rules = R["spc_points"]
    lead = RS.q(f"""
        with dr as (select job_id, shot_ts from drift_signals where detector <> 'lot_step' and shot_ts >= '{VALID_END.date()}'),
        al as (select job_id, shot_ts from fct_shot where unit_alarm_state = 'alarm' and after_approval)
        select dr.job_id, dr.shot_ts, min(al.shot_ts) as alarm_ts
        from dr join al on al.job_id = dr.job_id and al.shot_ts > dr.shot_ts and al.shot_ts <= dr.shot_ts + interval 8 hour
        group by 1, 2""")
    if len(lead):
        cnt = RS.q("select job_id, shot_ts from fct_shot where after_approval")
        cnt = cnt.sort_values(["job_id", "shot_ts"])
        n_between = []
        for r in lead.itertuples():
            g = cnt[cnt["job_id"] == r.job_id]["shot_ts"]
            n_between.append(int(((g >= r.shot_ts) & (g < r.alarm_ts)).sum()))
        lead["shots"] = n_between
    lead_shots = float(lead["shots"].median()) if len(lead) else np.nan
    lead_hours = float(((lead["alarm_ts"] - lead["shot_ts"]).dt.total_seconds() / 3600).median()) if len(lead) else np.nan
    vent = RS.check(R, "G2 CUSUM lead")["value"]
    vent_cyc = float(RS.q("select avg(cycle_time_s) from fct_shot where mold_id in (select mold_id from stg_monitoring__shot_summary where sensor_position = 'end_of_fill' group by 1)").iloc[0, 0])
    lot = R["lot_steps"]
    n_lot_flag, n_lot = int((lot["step_t"].abs() > 3).sum()), len(lot)
    best, drift_png, burn_lead_h = drift_example()

    vs = RS.vm_summary(R)
    vt = vs.pivot_table(index="cell", columns="target", values=["rmse_over_gauge", "r2"])
    ok = (vt[("rmse_over_gauge", "weight")] <= 1.5) & (vt[("rmse_over_gauge", "dimension")] <= 1.8)
    by_mold = ok.groupby(vt.index.str.split(" / ").str[0]).all()
    stretch = by_mold[by_mold].index.tolist()
    abl = R["vm_abl"].groupby(["target", "tag"])["rmse_over_gauge"].median()
    codes = R["codes"].pivot_table(index="defect_code", columns="source", values="qty", aggfunc="sum", fill_value=0)

    # ── 1. executive summary ──
    top_added = max(("rules", "drift", "virtual_metrology", "anomaly"), key=lambda k: share.get(k, 0))
    s1 = f"""
{section("summary", "Section 1", "Executive Summary")}
<p>The cell's two monitoring units already catch the largest share of what the cavity pressure curve can tell: their template alarms and sort cover
<strong>{pct(share.get('template_sort', 0))}</strong> of confirmed defects. Extending that SPC to every shot, with control-chart rules 1 and 2
(<strong>+{pts(share.get('rules', 0))} points</strong>) and drift detection (<strong>+{pts(share.get('drift', 0))} points</strong>), brings coverage to
<strong>{pct(pts_spc)}</strong> without any machine learning. On top of that, virtual metrology puts a predicted weight and critical dimension
on every shot, and adds <strong>+{pts(share.get('virtual_metrology', 0))} points</strong>; the anomaly model flagged <strong>{n_ev_flag} of {n_ev}</strong>
novel events in the test period and adds <strong>+{pts(share.get('anomaly', 0))} points</strong>. A supervised defect model was tested and not deployed,
because the defects it can learn from were mostly found by the template's own sort. <strong>{pct(share.get('none', 0))}</strong> of confirmed defects
left no trace in the curve. The added layers raise <strong>{added:.1f} alarms per shift</strong>, inside the budget of three to six; with the template's
own alarms, {total:.1f}.</p>
{chart("Confirmed defects covered, by layer (job-hour coverage, test period)", coverage_bar(share, budget),
       f"SPC layers (slate) cover {pct(pts_spc)} of confirmed defects; the two ML layers (amber) add {pts(share.get('virtual_metrology', 0) + share.get('anomaly', 0))} points; "
       f"{pct(share.get('none', 0))} were caught by none. Alarms per shift are printed beside each layer.")}
<div class="kpi-row">
{kpi(pct(share.get('template_sort', 0)), "Template and sort", "job-hour coverage of confirmed defects")}
{kpi(pct(pts_spc), "SPC subtotal", "template, rules 1 and 2, drift")}
{kpi("+" + pts(share.get('virtual_metrology', 0) + share.get('anomaly', 0)), "ML points added", "virtual metrology and anomaly", ml=True)}
{kpi(f"{added:.1f}", "Added alarms per shift", f"{total:.1f} including the template")}
</div>
"""

    # ── 2. the cell ──
    molds = []
    for m_id, r in parts.iterrows():
        sens = RS.q(f"select distinct sensor_position, cavity_no from stg_monitoring__shot_summary where mold_id = '{m_id}' order by 2, 1")
        molds.append(dict(Mold=m_id, Part=r["description"], Cavities=int(r["cavities"]), Resin=r["resin"],
                          Sensors=", ".join(f"cavity {int(c)} {p.replace('_', '-')}" for p, c in zip(sens["sensor_position"], sens["cavity_no"]))))
    molds = pd.DataFrame(molds)
    s2 = f"""
{section("cell", "Section 2", "The Cell and How Quality Is Controlled")}
<p>Two 200-ton presses, IM-11 and IM-12, run six instrumented molds in runs of two to six days. Each mold carries cavity pressure sensors in one or
two of its cavities. A <strong>post-gate</strong> sensor sits just inside the gate and sees the melt first; an <strong>end-of-fill</strong> sensor
sits at the far end of the flow path and sees the cavity fill last. The other cavities are not sensed and follow the sensed one at a stable offset.</p>
{table(molds, num_cols=("Cavities",))}
{section("curve", "2.1", "One shot's curve", sub=True)}
<p>Each shot produces one pressure curve per sensor. The <strong>template</strong> is the curve of the approved process, taken at the first-shot
approval of each run. The monitoring unit reduces every curve to a few <strong>summary values</strong>: the <strong>fill integral</strong> (area under
the curve while the cavity fills), the <strong>pack integral</strong> (area while hold pressure packs the part, up to gate seal), the <strong>peak
pressure</strong>, the <strong>gate seal time</strong> (when the gate freezes and pressure starts to fall), the <strong>end-of-fill pressure</strong>
(at the end-of-fill sensor when the cavity is full) and the <strong>cycle integral</strong> (the whole area).</p>
{chart("A good M-2118 shot against its template", curve_figure(),
       "The shot tracks its template through fill, transfer and pack; each summary value is read from a fixed part of the curve.")}
{section("unit", "2.2", "The monitoring unit, the sort and the audits", sub=True)}
<p>The unit compares each summary value with the template. Each value has a <strong>warning band</strong> and a wider <strong>alarm band</strong>
around the template value, set from the launch experiment; a deviation is stated in <strong>bands</strong>, where 1 band is the distance from the template
value to the alarm limit. A shot outside any alarm band is an <strong>alarm</strong>, and the press robot drops it in the reject bin: the
<strong>sort</strong>. An inspector reviews the bin every shift (the <strong>reject-bin review</strong>), confirms the defective pieces and returns
the good ones. Separately, quality pulls an <strong>hourly audit</strong> of 6 to 20 pieces, measures weight and the critical dimension, checks them
visually, and plots the mean weight on <strong>audit control charts</strong> with the Western Electric rules 1, 2, 4 and 5.</p>
{section("labels", "2.3", "How outcomes are known", sub=True)}
<p>Four records say which pieces were defective. On the medical molds the reject-bin review is tied to its shot, through the indexed tray; on the
other molds it covers a whole shift and carries the job, not the shot. Audit measurements are tied to their shot where the robot records the cycle it
sampled, on {pct(R['audits']['linked'], 0)} of audit pieces. Packing tallies record defects by job, hour and code with no shot,
and customer returns arrive weeks later. A job's quality is taken as final 21 days after its last shot. <strong>Linkage coverage</strong>, the share of
confirmed defective pieces tied to a shot, is {pct(link)}.</p>
<div class="box"><h4>How catch is measured</h4><ul>
<li><strong>Shot-level catch:</strong> on defects linked to a shot, whether the layer alarmed on that shot. Only the medical molds' reject-bin reviews and
the robot-linked audits are tied to a shot, so shot-level figures mostly reflect the medical molds.</li>
<li><strong>Job-hour coverage:</strong> on all confirmed defects (sort review, audits and packing tallies), whether the layer alarmed on that job in that
hour or shortly before (two hours; eight for drift, whose signal stays open until its reset). Each defect is credited to the layer that alarmed first in time.</li>
<li><strong>Points</strong> means percentage points of confirmed defects under job-hour coverage. <strong>Alarms per shift</strong> counts the job-hours
in which a layer fired, per eight hours of production.</li>
<li><strong>SPC</strong> layers use the shop's existing statistical process control; <strong>ML</strong> layers use models trained on the data.
Test period: December 2025 to March 2026, after training through September and validation in October and November.</li>
</ul></div>
"""

    # ── 3. existing control ──
    st = sort_m.copy()
    st["Mold"] = st["mold_id"] + np.where(st["mold_id"].isin(MEDICAL), " (medical)", "")
    st["Shots sorted"] = (st["sorted_shots"] / st["shots"]).map(lambda v: pct(v, 2))
    st["Review"] = st["review_mode"].map({"indexed_tray": "per shot", "per_shift": "per shift"})
    st["Sorted pieces found good"] = st["found_good"].map(lambda v: pct(v, 0))
    st["Good pieces sorted"] = st["good_pieces"].map(lambda v: f"{v:,.0f}")
    st["At standard cost"] = st["good_cost"].map(lambda v: f"${v:,.0f}")
    st = st[["Mold", "Review", "Shots sorted", "Sorted pieces found good", "Good pieces sorted", "At standard cost"]]
    med = sort_m[sort_m["mold_id"].isin(MEDICAL)]
    oth = sort_m[~sort_m["mold_id"].isin(MEDICAL)]
    med_fr = float(med["good_pieces"].sum() / med["sorted_pieces"].sum())
    oth_fr = float(oth["good_pieces"].sum() / oth["sorted_pieces"].sum())
    hi = sort_m.sort_values("found_good").iloc[-1]
    s3 = f"""
{section("existing", "Section 3", "What the Existing Process Control Caught")}
<p><span class="tag spc">SPC</span> The template alarms and sort cover <strong>{pct(share.get('template_sort', 0))}</strong> of confirmed defects
(job-hour coverage) at <strong>{budget['template_sort']:.2f}</strong> alarm hours per shift. At the shot level, {pct(sl['template_sort'])} of audit-found
defects (pieces pulled on a clock, independent of the sort) fell on shots the template had alarmed. Every sort-found defect tied to a shot is caught by
construction; those are the medical molds' tray reviews.</p>
<p>Over the fifteen months the units sorted <strong>{pct(sorted_share, 2)}</strong> of production shots. The reject-bin review found
<strong>{pct(fr, 0)}</strong> of sorted pieces good: <strong>{good_pieces:,.0f}</strong> good pieces in the reject bin, <strong>${good_cost:,.0f}</strong> at
standard cost. The medical molds sort into an indexed tray and are reviewed shot by shot; there, {pct(fr_tray, 0)} of sorted shots held no defective piece.
The other molds' bins are reviewed once per shift, so their results are known per job and shift, not per shot.</p>
{table(st, num_cols=("Shots sorted", "Sorted pieces found good", "Good pieces sorted", "At standard cost"))}
<p>All six molds use alarm bands of the same width. The two medical molds returned {pct(med_fr, 0)} of their sorted pieces as good against {pct(oth_fr, 0)}
for the others; {hi.mold_id} has the highest share ({pct(hi.found_good, 0)}). The difference does not come from the bands, which are the same on every mold.</p>
<p>The units' recorded alarm states agree with the platform's recomputation from the summary values and templates on
<strong>{pct(S['unit_agreement'], 2)}</strong> of production shots, so every layer below builds on the alarm logic the technicians see.
What this layer cannot see are defects that leave no trace in the curve: handling damage, contamination and specks.</p>
"""

    # ── 4. extending SPC ──
    s4 = f"""
{section("spc", "Section 4", "What Extending SPC Added, Without Machine Learning")}
<p>Nothing in this section is machine learning. It runs the shop's own control-chart rules on every shot instead of the hourly audit sample, and adds
two standard drift statistics.</p>
{section("every-shot", "4.1", "Charting every shot", sub=True)}
<p><span class="tag spc">SPC</span> With limits from the first 500 shots after each approval, rule 1 (one point beyond 3 sigma) fires on
{pct(pts_rules['we1'])} of points, rule 2 (two of three beyond 2 sigma) on {pct(pts_rules['we2'])}, rule 4 (four of five beyond 1 sigma) on
{pct(pts_rules['we4'])} and rule 5 (eight in a row on one side) on <strong>{pct(pts_rules['we5'])}</strong>. Consecutive shots are correlated: the melt,
the mold and the material change slowly, so one shot looks much like the last. Patterns that are rare in an hourly sample are normal from shot to shot,
and the run rules (4 and 5) read them as shifts. Rules 1 and 2 are kept: they add <strong>+{pts(share.get('rules', 0))} points</strong> at
<strong>{budget['rules']:.2f}</strong> alarms per shift.</p>
{section("drift", "4.2", "Drift detection", sub=True)}
<p><span class="tag spc">SPC</span> An <strong>EWMA</strong> (exponentially weighted moving average) smooths each shot's fill and pack integral so a
slow shift stands out from shot-to-shot noise. A <strong>CUSUM</strong> (cumulative sum) adds up small deviations in one direction until they cross a limit,
and here runs on end-of-fill pressure (vents), on pack-integral variance (check ring) and on gate seal time (mold temperature). Each resets at an approval,
a documented correction, a vent cleaning, PM, a ring replacement or a lot change.</p>
<p>Drift detection adds <strong>+{pts(share.get('drift', 0))} points</strong> at <strong>{budget['drift']:.2f}</strong> alarms per shift. Where a drift
signal and a template alarm both fired on a job, the drift signal came first by a median <strong>{lead_shots:,.0f} shots</strong> (about
{lead_hours:.1f} hours) across {len(lead)} cases in the test period. The vent CUSUM leads the rise in burn tallies by a median
<strong>{vent:,.0f} shots</strong> (about {vent * vent_cyc / 3600:,.0f} hours of production).</p>
{chart(f"Vent restriction on job {best.job_id} ({best.mold_id})", drift_png,
       "End-of-fill pressure falls as the vents load, and the CUSUM signals before burns start to climb at packing"
       + ("" if burn_lead_h != burn_lead_h else f" (about {burn_lead_h:.0f} hours before the first hour with three or more burns)") + ".")}
<p>The same statistics carry two maintenance signals. Check-ring leakage on IM-12 shows as rising pack-integral variance until the ring was replaced, with
no change in the mean. Resin lot changes step the fill integral at the load: the lot-step test flagged <strong>{n_lot_flag} of {n_lot}</strong> lot changes
that fell inside a job.</p>
{section("spc-total", "4.3", "SPC subtotal", sub=True)}
<p><span class="tag spc">SPC</span> The template, rules 1 and 2 and drift detection together cover <strong>{pct(pts_spc)}</strong> of confirmed defects.
The two added SPC layers raise <strong>{spc_alarms:.2f}</strong> alarms per shift on top of the template's {budget['template_sort']:.2f}.</p>
"""

    # ── 5. ML ──
    vt2 = pd.DataFrame({"Mold / press": vt.index,
                        "Weight error vs gauge": vt[("rmse_over_gauge", "weight")].map(lambda v: f"{v:.2f}x").values,
                        "Weight R2": vt[("r2", "weight")].map(lambda v: f"{v:.2f}").values,
                        "Dimension error vs gauge": vt[("rmse_over_gauge", "dimension")].map(lambda v: f"{v:.2f}x").values,
                        "Dimension R2": vt[("r2", "dimension")].map(lambda v: f"{v:.2f}").values})
    stretch_molds = sorted(stretch)
    cal = R["vm"][R["vm"]["design"] == "pooled"].groupby("target")["calibration_slope"].median()
    pts_per = {k: share.get(k, 0) * 100 / budget[k] for k in ("rules", "drift", "anomaly") if budget.get(k)}
    evt = ev_test[["kind", "mold_id", "press_id", "minutes_to_flag", "minutes_to_template_alarm", "lead_over_response_min"]].copy()
    evt["kind"] = evt["kind"].str.replace("_", " ")
    for c in ("minutes_to_flag", "minutes_to_template_alarm", "lead_over_response_min"):
        evt[c] = evt[c].map(lambda v: "" if v != v else f"{v:+.0f}" if c == "lead_over_response_min" else f"{v:.0f}")
    evt.columns = ["Event", "Mold", "Press", "Minutes to first flag", "Minutes to first template alarm", "Flag lead over the shop's response (min)"]
    s5 = f"""
{section("ml", "Section 5", "What Machine Learning Added", kind="ml")}
<p>Two model components were deployed. Each section says what the model outputs, how it is computed, how it was checked, and what it enables.</p>
{section("vm", "5.1", "Virtual metrology", kind="ml", sub=True)}
<p><span class="tag ml">ML</span> <strong>Output.</strong> A predicted part weight and critical dimension for every shot, for each cavity; the unit screen
shows the mean over the active cavities. <strong>How it is computed.</strong> A gradient-boosted regression trained on audited pieces linked to their shot
({R['audits']['pieces']:,.0f} pieces), using that shot's summary values normalized to the template, the press's machine-side values and the context
(lot, dryer, maintenance counters, setpoints). It estimates the parts just made; it does not forecast future shots.</p>
<p><strong>How it was checked.</strong> On the test period the error is <strong>{vmh[('pooled', 'weight')]['ratio']:.2f} times</strong> the gauge R&amp;R
on weight and <strong>{vmh[('pooled', 'dimension')]['ratio']:.2f} times</strong> on the dimension, with R&sup2; of {vmh[('pooled', 'weight')]['r2']:.2f}
and {vmh[('pooled', 'dimension')]['r2']:.2f} (medians over mold and press, five seeds; calibration slopes {cal.get('weight', np.nan):.2f} and
{cal.get('dimension', np.nan):.2f}). Weight R&sup2; is lower because a shot moves weight by only a few tenths of a percent, close to the scale's own
error, so the gauge noise is a larger share of what is measured. Without the cavity pressure values the dimension error rises to
{abl.get(('dimension', 'no_cavity_signal'), np.nan):.2f} times gauge.</p>
{table(vt2, num_cols=("Weight error vs gauge", "Weight R2", "Dimension error vs gauge", "Dimension R2"))}
{chart("Predicted against gauged critical dimension, test period", vm_scatter(R['vm_pred']),
       "Predictions follow the gauge closely across the tolerance band; the spread is mostly the gauge's own error.")}
<p><strong>Summer backtest.</strong> The test period (December to March) carried little mold-temperature drift, which is part of why virtual metrology
added no catch there. A backtest trained on January to May and evaluated on June to September, when mold temperatures drifted with the season, gives an
error of {bt['weight']['ratio']:.2f} times gauge on weight and {bt['dimension']['ratio']:.2f} times on the dimension (R&sup2;
{bt['weight']['r2']:.2f} and {bt['dimension']['r2']:.2f}), and adds <strong>+{pts(summer['share'].get('virtual_metrology', 0))} points</strong> in that window
({pts(summer['dimensional_share'].get('virtual_metrology', 0))} points of dimensional defects).</p>
<p><strong>Why it adds little catch.</strong> {pct(L['dimensional_share'].get('template_sort', 0), 0)} of dimensional defects in the test period fell on
shots the template had already alarmed, because the pack excursions that push a dimension out of tolerance also leave the pack band. The value of
virtual metrology is a measurement on every shot, not more catches.</p>
<p><strong>What it enables.</strong></p>
<ul class="plain">
<li>Dimensional drift visible between audits, shot by shot, instead of at the next hourly sample.</li>
<li>A longer audit interval on molds where the error sits within 1.5 times the gauge error on weight and 1.8 times on the dimension on both presses:
{", ".join(stretch_molds) if stretch_molds else "no mold on this run"}.</li>
<li>A predicted dimension for every part in the customer lot record.</li>
<li>A prediction-triggered audit: an extra sample pulled when the predicted dimension passes 75% of tolerance
({budget.get('vm_advisory_audits', 0):.2f} per shift on the test period). It is advisory and never sorts a part.</li>
</ul>
{section("anomaly", "5.2", "Anomaly detection", kind="ml", sub=True)}
<p><span class="tag ml">ML</span> <strong>Output.</strong> An anomaly score for every shot, and an <strong>unusual</strong> state when two of the last ten
shots score above the threshold. <strong>How it is computed.</strong> An isolation forest per mold and press, trained only on validated good shots (after
approval, no alarm, no linked defect), with no defect labels. It scores how far each shot's values have moved from the run's own recent shots, so a run
sitting steadily off its template is left to the template and the charts.</p>
<p><strong>How it was checked.</strong> It flagged <strong>{n_ev_flag} of {n_ev}</strong> novel events in the test period (events no rule was written for:
nozzle drool, a failing heater zone). {pct(an['share_on_template_pass'], 0)} of its flags fall on shots the template passed. It adds
<strong>+{pts(share.get('anomaly', 0))} points</strong> at <strong>{budget['anomaly']:.2f}</strong> alarms per shift: {pts_per.get('anomaly', 0):.1f} points per
alarm per shift, against {pts_per.get('rules', 0):.1f} for rules 1 and 2 and {pts_per.get('drift', 0):.1f} for drift detection.</p>
{table(evt, num_cols=("Minutes to first flag", "Minutes to first template alarm", "Flag lead over the shop's response (min)"))}
{chart("Novel events in the test period: minutes to the first signal", anomaly_figure(ev),
       "On the heater zone failure the anomaly model flagged well before the template alarmed; on nozzle drool both fired at once.")}
"""

    # ── 6. shortfalls ──
    lab_counts = codes.copy()
    lab_counts["Linked to a shot"] = lab_counts.get("sort", 0) + lab_counts.get("audit", 0)
    lab_counts = lab_counts.sort_values("Linked to a shot", ascending=False).reset_index()
    lab_counts["defect_code"] = lab_counts["defect_code"].str.replace("_", " ")
    lab_counts = lab_counts.rename(columns={"defect_code": "Code", "sort": "Sort review", "audit": "Audit", "tally": "Packing tally", "return": "Returns"})
    lab_counts = lab_counts[["Code", "Sort review", "Audit", "Packing tally", "Returns", "Linked to a shot"]]
    for c in lab_counts.columns[1:]:
        lab_counts[c] = lab_counts[c].map(lambda v: f"{v:,.0f}")
    full = dict(recall_model=float(any_["recall_model"]), recall_template=float(any_["recall_template"]))
    ae_ev, ae_non = an.get("ae_event_flag_rate", np.nan), an.get("ae_nonevent_flag_rate", np.nan)
    s6 = f"""
{section("short", "Section 6", "Where It Fell Short", kind="ml")}
{section("supervised", "6.1", "Supervised defect model: tested and not deployed", kind="ml", sub=True)}
<p>A gradient-boosted classifier trained on confirmed defects linked to a shot catches <strong>{pct(full['recall_model'])}</strong> of defective shots against
the template's <strong>{pct(full['recall_template'])}</strong> at the template's own alarm rate. Most of those labels exist because the template sorted the shot
and the reviewer confirmed it, so the model is largely learning to reproduce the template. Scored only on audit-found defects, pieces the robot pulled on a
clock and independent of the template ({ao['n_test_pos']:,.0f} defective of {ao['n_test']:,.0f} audited shots in the test period), it catches
<strong>{pct(ao['recall_model'])}</strong> against the template's <strong>{pct(ao['recall_template'])}</strong> at the template's alarm rate on those shots
({pct(ao['alarm_rate'], 1)}).</p>
{chart("Recall at the template's alarm rate, test period", supervised_figure(full, ao),
       "On all shot-linked labels the model tracks the template; on defects found independently of the sort both catch far less, and the model's edge is small.")}
<p>Confirmed defective pieces by code and how they were found. Rare codes have too few shot-linked labels to train on:</p>
{table(lab_counts, num_cols=tuple(lab_counts.columns[1:]))}
{section("none", "6.2", "Caught by none", kind="ml", sub=True)}
<p><strong>{pct(share.get('none', 0))}</strong> of confirmed defects had no alarm from any layer. Most are handling damage, contamination, black specks and
splay that leave no trace in the cavity pressure curve; no layer built on the curve can see them.</p>
{section("threshold", "6.3", "Anomaly threshold", kind="ml", sub=True)}
<p>The threshold was set so 0.45% of shots on the fit window are flagged. On the test period {pct(an['test_flag_rate'], 1)} of production shots were flagged,
{pct(an['false_positive_rate'], 2)} of validated good shots; the two-in-ten rule turns those flags into {budget['anomaly']:.2f} alarms per shift.</p>
{section("autoencoder", "6.4", "Curve autoencoder", kind="ml", sub=True)}
<p>An autoencoder on the retained curves was run as a comparison. It flagged {pct(ae_ev, 1)} of novel-event curves against {pct(ae_non, 1)} of other
retained curves, so it did not separate the events and was not deployed.</p>
{section("linkage", "6.5", "Linkage", kind="ml", sub=True)}
<p>Only {pct(link)} of confirmed defective pieces are tied to a shot; packing tallies and returns carry a job and hour at most. That is why job-hour
coverage is the main measure, and why it credits a layer for alarming near a defect rather than on the defective shot itself.</p>
"""

    # ── 7. budget and outputs ──
    bt_rows = pd.DataFrame([
        ("Template alarms and sort", "SPC (existing)", f"{budget['template_sort']:.2f}", pct(share.get('template_sort', 0))),
        ("Control-chart rules 1 and 2", "SPC (added)", f"{budget['rules']:.2f}", "+" + pts(share.get('rules', 0))),
        ("Drift detection", "SPC (added)", f"{budget['drift']:.2f}", "+" + pts(share.get('drift', 0))),
        ("Virtual metrology", "ML", f"{budget['virtual_metrology']:.2f}", "+" + pts(share.get('virtual_metrology', 0))),
        ("Anomaly detection", "ML", f"{budget['anomaly']:.2f}", "+" + pts(share.get('anomaly', 0))),
        ("Added layers", "", f"{added:.2f}", "+" + pts(sum(share.get(k, 0) for k in ('rules', 'drift', 'virtual_metrology', 'anomaly')))),
        ("All layers, including the template", "", f"{total:.2f}", pct(1 - share.get('none', 0))),
    ], columns=["Layer", "Kind", "Alarms per shift", "Points (job-hour coverage)"])
    most_alarms = max(("rules", "drift", "virtual_metrology", "anomaly"), key=lambda k: budget.get(k, 0))
    uses = pd.DataFrame([
        ("Unit screen at the press", "Process technician", "Template status, the anomaly state, predicted weight and dimension against tolerance and the last audit"),
        ("Shift chart", "Technician, quality", "Per-shot values with EWMA and drift onset markers; the predicted-dimension trace with audit points"),
        ("Quality dashboard", "Quality engineer, weekly", "Alarms and sorts by mold and lot, false-reject rate, virtual metrology error against each audit, label maturity"),
        ("Job record: the <a href=\"run_report_J-250188.html\">run report</a>", "Quality engineer at job close; customer evidence",
         "Per-job summary, first-shot approval, curve overlay, distributions, timeline, audit charts, capability, reject-bin review, reaction log"),
    ], columns=["Where", "Who acts on it", "What it shows"])
    s7 = f"""
{section("budget", "Section 7", "Alarm Budget and How the Outputs Are Used")}
<p>Thresholds are set for what one technician per shift can investigate. The added layers raise <strong>{added:.2f}</strong> alarms per shift, inside the
budget of three to six; with the template's own {budget['template_sort']:.2f}, the total is {total:.2f}. Of the added layers, {LAYER_NAMES[most_alarms].lower()}
raises the most alarms and {LAYER_NAMES[top_added].lower()} adds the most points.</p>
{table(bt_rows, num_cols=("Alarms per shift", "Points (job-hour coverage)"), total_last=True)}
{chart("Added alarms per shift by layer, test period", budget_figure(budget),
       f"The added layers sum to {added:.2f} alarms per shift; {LAYER_NAMES[most_alarms].lower()} contributes the largest share.")}
{table(uses)}
<p>The outputs go on the monitoring unit's display where the vendor accepts additions, and on an MES companion view beside the press otherwise.</p>
{section("limits", "7.1", "Other limitations", sub=True)}
<ul class="plain">
<li>The layer comparison credits a layer for alarming on the job near a defect; it does not prove the alarm would have prevented that defect.</li>
<li>Returns arrive weeks after the job; a job's figures are final only 21 days after its last shot.</li>
<li>Virtual metrology is trained on the six molds and two presses here; a new mold needs its own audit history before its predictions are used.</li>
</ul>
"""

    toc = [("summary", "1 · Executive Summary", False), None,
           ("cell", "2 · The Cell and Quality Control", False), ("curve", "One Shot's Curve", True), ("unit", "Unit, Sort and Audits", True),
           ("labels", "How Outcomes Are Known", True), None,
           ("existing", "3 · Existing Process Control", False), None,
           ("spc", "4 · Extending SPC", False), ("every-shot", "Charting Every Shot", True), ("drift", "Drift Detection", True),
           ("spc-total", "SPC Subtotal", True), None,
           ("ml", "5 · Machine Learning", False), ("vm", "Virtual Metrology", True), ("anomaly", "Anomaly Detection", True), None,
           ("short", "6 · Where It Fell Short", False), ("supervised", "Supervised Model", True), ("none", "Caught by None", True),
           ("threshold", "Anomaly Threshold", True), ("autoencoder", "Curve Autoencoder", True), ("linkage", "Linkage", True), None,
           ("budget", "7 · Alarm Budget and Use", False), ("limits", "Other Limitations", True)]
    nav = '<nav class="toc"><div class="toc-title">Contents</div>' + "".join(
        "<hr>" if t is None else f'<a href="#{t[0]}"{" class=\"sub\"" if t[2] else ""}>{t[1]}</a>' for t in toc) + "</nav>"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Overview: Molding Cell Cavity Pressure Monitoring</title><style>{CSS}</style></head>
<body>
<div class="page-header"><h1>ML Overview: Molding Cell Cavity Pressure Monitoring</h1>
<div class="sub">IM-11 and IM-12 · January 2025 to March 2026 · test period December 2025 to March 2026</div></div>
<div class="layout">{nav}<main class="content">
{s1}{s2}{s3}{s4}{s5}{s6}{s7}
</main></div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
