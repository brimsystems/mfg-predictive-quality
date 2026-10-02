"""
ML overview for the quality manager.

The story, in order: what the template detects on the shot, measured against how often it flags good pieces; what the
curve cannot see; and what each added layer adds on its own measure (incremental detection above chance for the
control-chart rules and the models, the drift detectors against their mechanisms, the anomaly model per novel event,
virtual metrology as a measurement). Every layer is evaluated May 2025 to March 2026 at its deployed threshold. Every
number is read from the current run.

Usage: python ml/reports/generate_ml_overview.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports import results as RS  # noqa: E402
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREY, RED, fig, img  # noqa: E402

OUT = Path(__file__).resolve().parent / "ml_overview.html"
ORDER = ["template_sort", "rules", "drift", "anomaly", "virtual_metrology"]
COLORS = {"template_sort": "#2F4458", "rules": "#5F7F99", "drift": "#9CB5C9", "anomaly": "#E3A54D", "virtual_metrology": "#B86E12"}
LAYER_NAMES = {"template_sort": "Template sort", "rules": "Control-chart rules 1 and 2", "drift": "Drift detection",
               "anomaly": "Anomaly detection", "virtual_metrology": "Virtual metrology advisory"}
GROUPS = ["Fill volume", "Packing and shrinkage", "Flow front", "Gate and other", "Material"]
GROUP_CODES = {"Fill volume": "short shot, flash", "Packing and shrinkage": "sink, void, dimensional, warp",
               "Flow front": "weld line, burn", "Gate and other": "gate vestige, other",
               "Material": "splay, black specks, contamination"}
MEDICAL = ("M-2118", "M-2119")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pts(x, d=1):
    v = round(x * 100, d)
    return f"{0.0:.{d}f}" if v == 0 else f"{v:+.{d}f}"


def ci(r):
    lo, hi = r["ci_above_chance"]
    return f"[{lo * 100:+.1f}, {hi * 100:+.1f}]"


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
def detection_bar(inc):
    """Incremental shot-level detection by layer: defective audited pieces each layer flagged that no earlier layer
    flagged, against the same share of good pieces (the chance level)."""
    f, ax = fig(3.4, 8.6)
    y = np.arange(len(ORDER))[::-1]
    for yi, k in zip(y, ORDER):
        r = inc[k]["All defects"]
        ax.barh(yi, r["rate_defective"] * 100, color=COLORS[k], height=0.55, label=None)
        ax.plot([r["rate_good"] * 100] * 2, [yi - 0.36, yi + 0.36], color="#111111", lw=2.2)
        ax.text(max(r["rate_defective"], r["rate_good"]) * 100 + 0.5, yi,
                f"{pts(r['above_chance'])} pts above chance {ci(r)}", va="center", fontsize=9.5, color="#222")
    ax.plot([], [], color="#111111", lw=2.2, label="good pieces flagged (chance level)")
    ax.set_yticks(y, [LAYER_NAMES[k] for k in ORDER])
    ax.set_xlabel("defective audited pieces newly flagged (%)")
    ax.set_xlim(0, max(inc["template_sort"]["All defects"]["rate_defective"] * 100 * 1.75, 10))
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return img(f, "incremental shot-level detection by layer")


def group_chart(layer, inc):
    """By defect group: defective pieces flagged by the template and by any layer, each with its good-piece rate."""
    f, ax = fig(3.6, 8.6)
    x = np.arange(len(GROUPS))
    for off, name, col in ((-0.2, "template sort", COLORS["template_sort"]), (0.2, "any layer", ACCENT)):
        for xi, g in zip(x, GROUPS):
            if name == "template sort":
                r = layer["template_sort"][g]
                d, gd = r["rate_defective"], r["rate_good"]
            else:
                d = sum(inc[k][g]["rate_defective"] for k in ORDER)
                gd = sum(inc[k][g]["rate_good"] for k in ORDER)
            ax.bar(xi + off, d * 100, 0.38, color=col, label=name if xi == 0 else None)
            ax.plot([xi + off - 0.19, xi + off + 0.19], [gd * 100] * 2, color="#111111", lw=2)
            ax.text(xi + off, d * 100 + 1, f"{d * 100:.0f}%", ha="center", fontsize=8.5)
    ax.plot([], [], color="#111111", lw=2, label="good pieces flagged (chance level)")
    ax.set_xticks(x, [g.replace(" and ", "\nand ") for g in GROUPS])
    ax.set_ylabel("defective audited pieces flagged (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return img(f, "detection by defect group")


def not_detected_chart(inc):
    f, ax = fig(2.8, 8.6)
    vals = [1 - sum(inc[k][g]["rate_defective"] for k in ORDER) for g in GROUPS]
    ns = [inc["template_sort"][g]["n_defective"] for g in GROUPS]
    y = np.arange(len(GROUPS))[::-1]
    ax.barh(y, [v * 100 for v in vals], color=GREY, height=0.55)
    for yi, v, n in zip(y, vals, ns):
        ax.text(v * 100 + 1, yi, f"{v * 100:.0f}% of {n:,} pieces", va="center", fontsize=9.5)
    ax.set_yticks(y, GROUPS)
    ax.set_xlim(0, 115)
    ax.set_xlabel("defective audited pieces flagged by no layer (%)")
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)
    return img(f, "not detected by group")


def drift_tests_chart(DM):
    v, l, r = DM["vent"], DM["lot_steps"], DM["ring"]
    rows = [("End-of-fill CUSUM:\nfirst firing before the burn rise", v["preceded"]["share"], v["preceded"]["ci"],
             v["chance_preceded"]["mean"], v["chance_preceded"]["range95"], "random burn-rise time"),
            ("Fill EWMA:\nlot changes with a fill step flagged", l["with_step"]["share"], l["with_step"]["ci"],
             l["without_step"]["share"], l["without_step"]["ci"], "lot changes without a step"),
            ("Variance CUSUM:\nworn-ring jobs flagged before the first sink or void", r["fired_before_defect"]["share"],
             r["fired_before_defect"]["ci"], r["matched_unworn"]["share"], r["matched_unworn"]["ci"], "matched unworn jobs")]
    f, ax = fig(3.3, 8.6)
    y = np.arange(len(rows))[::-1]
    for yi, (name, a, aci, b, bci, bname) in zip(y, rows):
        ax.errorbar(a * 100, yi + 0.12, xerr=[[a * 100 - aci[0] * 100], [aci[1] * 100 - a * 100]], fmt="o", color=BRAND_BLUE, capsize=3)
        ax.errorbar(b * 100, yi - 0.12, xerr=[[b * 100 - bci[0] * 100], [bci[1] * 100 - b * 100]], fmt="o", mfc="white", color=GREY, capsize=3)
        ax.text(102, yi - 0.12, f"vs {bname}", va="center", fontsize=8.5, color="#666")
    ax.errorbar([], [], fmt="o", color=BRAND_BLUE, label="detector")
    ax.errorbar([], [], fmt="o", mfc="white", color=GREY, label="comparison")
    ax.set_yticks(y, [rr[0] for rr in rows])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("share (%), with 95% intervals")
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    return img(f, "drift detectors against their mechanisms")


def vm_month_chart(vm_month):
    v = vm_month.groupby(["target", "variant", "month"])["rmse_over_gauge"].median().reset_index()
    f, ax = fig(3.4, 8.6)
    ax.axvspan(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01"), color="#FBEBD3", alpha=0.6, lw=0)
    ax.text(pd.Timestamp("2025-08-01"), 0.2, "warm months", ha="center", fontsize=9, color="#9A6A1E")
    for (t, var), g in v.groupby(["target", "variant"]):
        col = BRAND_BLUE if t == "dimension" else AMBER
        ax.plot(g["month"] + pd.Timedelta(days=14), g["rmse_over_gauge"], color=col, lw=1.6 if var == "full" else 1.1,
                ls="-" if var == "full" else "--", marker="o" if var == "full" else None, ms=3.5,
                label=f"{t}, {'all features' if var == 'full' else 'without cavity pressure'}")
    ax.axhline(1, color="#999", lw=0.8, ls=":")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("error over gauge R&R (median mold and press)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return img(f, "virtual metrology error by month")


def vm_scatter(pred):
    p = pred[pred["target"] == "dimension"]
    p = p.assign(meas=(p["measured"] - p["nominal"]) / p["tolerance"], pr=(p["predicted"] - p["nominal"]) / p["tolerance"])
    f, ax = fig(3.6, 5.0)
    ax.scatter(p["pr"], p["meas"], s=2, alpha=0.12, color=ACCENT)
    ax.plot([-1.3, 1.3], [-1.3, 1.3], color=BRAND_BLUE, lw=1)
    for yv in (-1, 1):
        ax.axhline(yv, color=RED, lw=0.8, ls="--")
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
    ax.set_ylim(0, max(full["recall_template"], full["recall_model"]) * 1.3)
    ax.legend(frameon=False, loc="upper right")
    return img(f, "supervised comparison")


def budget_figure(bud):
    names = [k for k in ORDER[1:]]
    f, ax = fig(2.8, 7.0)
    vals = [bud[k]["episodes_per_shift"] for k in names]
    ax.bar([LAYER_NAMES[k].replace(" advisory", "").replace("Control-chart rules", "Rules") for k in names], vals,
           color=[COLORS[k] for k in names])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(1, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("alarm episodes per shift")
    ax.tick_params(axis="x", labelsize=9)
    return img(f, "alarm budget")


# ── page ──────────────────────────────────────────────────────────────────
def main():
    R = RS.load()
    M, DM = R["M"], R["DM"]
    SL = M["shot_level"]
    lay, inc, incw = SL["layer"], SL["incremental"], SL["incremental_warm"]
    bud = M["budget"]
    S = R["shots"]
    T = lay["template_sort"]["All defects"]
    vmo = RS.vm_overall(R)
    vmo_abl = RS.vm_overall(R, "no_cavity_signal")
    vbc = RS.vm_by_cell(R)
    dt = RS.defect_table(R)
    any_ = dt[dt["target"] == "y_any"].iloc[0]
    full = dict(recall_model=float(any_["recall_model"]), recall_template=float(any_["recall_template"]))
    ao = R["audit_only"]
    A = M["anomaly"]
    ev = pd.DataFrame(A["rows"])
    ev["start"] = pd.to_datetime(ev["start"])
    C15 = M["context_15_months"]
    textbook = bud["rules_at_textbook_limits"]
    added = bud["added_layers_total"]
    notdet_all = 1 - sum(inc[k]["All defects"]["rate_defective"] for k in ORDER)
    ml_all, ml_warm = inc["ml_layers"]["All defects"], incw["ml_layers"]["All defects"]
    vm_pack, vm_dw_warm = inc["virtual_metrology"]["Packing and shrinkage"], incw["virtual_metrology"]["Dimensional and warp"]
    heater = ev[ev["kind"] == "heater_zone"]
    heater = heater.iloc[0] if len(heater) else None
    lv, ls_, rg = DM["vent"], DM["lot_steps"], DM["ring"]

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

    # ── 1. executive summary ──
    chance_groups = [g for g in GROUPS if all(lay[k][g]["ci_above_chance"][0] <= 0.0 for k in ("template_sort",))]
    s1 = f"""
{section("summary", "Section 1", "Executive Summary")}
<p>The template sort is the cell's main detection layer and the only one that removes parts. From May 2025 to March 2026 it flagged
<strong>{pct(T['rate_defective'])}</strong> of defective audited pieces against <strong>{pct(T['rate_good'])}</strong> of good ones:
<strong>{pts(T['above_chance'])} points above chance</strong> {ci(T)}. It detects fill-volume defects best
({pts(lay['template_sort']['Fill volume']['above_chance'])} points) and packing and shrinkage defects next
({pts(lay['template_sort']['Packing and shrinkage']['above_chance'])}).</p>
<p>What the curve cannot see sets the limit for every layer built on it. Material defects (splay, black specks, contamination) sit at chance for the
template ({pts(lay['template_sort']['Material']['above_chance'])} points), and gate and flow-front defects close to it
({pts(lay['template_sort']['Gate and other']['above_chance'])} and {pts(lay['template_sort']['Flow front']['above_chance'])}).
<strong>{pct(notdet_all, 0)}</strong> of defective audited pieces were flagged by no layer at all.</p>
<p>Every other layer raises an investigation, or for virtual metrology an advisory audit, and each is set to the alarm budget and judged on its own
measure:</p>
<ul class="plain">
<li><strong>Control-chart rules 1 and 2</strong> detect {pts(inc['rules']['All defects']['above_chance'])} points above chance beyond the
template, mostly material defects the template cannot see ({pts(inc['rules']['Material']['above_chance'])} points).</li>
<li><strong>Drift detection</strong> adds {pts(inc['drift']['All defects']['above_chance'])} points at the shot. Tested on the mechanisms it is built for,
the fill EWMA flags <strong>{pct(ls_['with_step']['share'], 0)}</strong> of resin lot changes that step the fill integral against
{pct(ls_['without_step']['share'], 0)} of those that do not; the vent and check-ring detectors do no better than chance.</li>
<li><strong>Anomaly detection</strong> flagged <strong>{A['flagged']} of {A['events']}</strong> novel events in the period, {A['before_template']} of them
before the template; at the shot it adds {pts(inc['anomaly']['All defects']['above_chance'])} points.</li>
<li><strong>Virtual metrology</strong> puts a measurement on every shot: dimension error {vmo['dimension']['all']['rmse_over_gauge']:.2f} times the gauge
R&amp;R ({vmo['dimension']['warm']['rmse_over_gauge']:.2f} in the warm months) and weight {vmo['weight']['all']['rmse_over_gauge']:.2f} times. As an
advisory it adds <strong>{pts(vm_pack['above_chance'])} points</strong> on packing and shrinkage defects and {pts(vm_dw_warm['above_chance'])} on
dimensional and warp defects in the warm months.</li>
</ul>
<p>The two model layers together add <strong>{pts(ml_all['above_chance'])} points</strong> above chance overall {ci(ml_all)}. The added layers raise
<strong>{added:.2f} alarm episodes per shift</strong>, inside the budget of three to four investigations.</p>
{chart("Shot-level detection by layer, incremental, May 2025 to March 2026", detection_bar(inc),
       "Each bar is the share of defective audited pieces a layer flagged that no earlier layer had flagged; the black tick is the same share of good "
       "pieces, the chance level. Layers are added in order from the top.")}
<div class="kpi-row">
{kpi(pts(T['above_chance']), "Template, points above chance", f"{pct(T['rate_defective'])} of defective vs {pct(T['rate_good'])} of good pieces")}
{kpi(pts(vm_pack['above_chance']), "Virtual metrology, packing", "incremental points above chance", ml=True)}
{kpi(f"{A['flagged']} of {A['events']}", "Novel events flagged", f"{A['before_template']} before the template", ml=True)}
{kpi(f"{added:.2f}", "Investigations per shift", "rules, drift, anomaly and the advisory")}
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
<strong>sort</strong>. An inspector reviews the bin (the <strong>reject-bin review</strong>), confirms the defective pieces and returns the good ones.
Separately, quality pulls an <strong>hourly audit</strong> of 6 to 20 pieces, measures weight and the critical dimension and checks them visually.</p>
{section("measure", "2.3", "How detection is measured", sub=True)}
<p>Audits pull pieces on a clock, independent of every layer, and {pct(R['audits']['linked'], 0)} of audit pieces are tied to the shot they came from
on all six molds. That makes them the fair test of detection: a layer is credited only for flagging the shot a defective piece came from, and is
charged for the good pieces it flags in the same audits.</p>
<div class="box"><h4>Definitions</h4><ul>
<li><strong>Detected above chance:</strong> the share of defective audited pieces whose shot a layer flagged, minus the share of good audited pieces from
the same audits whose shot it flagged. The good-piece share is the <strong>chance level</strong>: what flagging shots at random would score. 95%
intervals come from resampling whole audits.</li>
<li><strong>Incremental:</strong> layers are added in the order template, rules, drift, anomaly, virtual metrology; each is credited only for pieces no
earlier layer flagged, minus the same share for good pieces.</li>
<li><strong>Evaluation period:</strong> May 2025 to March 2026 for every layer. Virtual metrology and the anomaly model are retrained at the start of each
month on all earlier data and applied to that month only, so every month is out of sample, including the summer of 2025.</li>
<li><strong>Alarm episode:</strong> a layer's flagged shots on a job, with gaps under 30 shots, count as one investigation. Each layer's deployed
threshold is set from this budget, and every figure in this report uses those thresholds.</li>
<li><strong>Defect groups:</strong> {"; ".join(f"{g.lower()} ({c})" for g, c in GROUP_CODES.items())}.</li>
</ul></div>
"""

    # ── 3. the template ──
    st = sort_m.copy()
    st["Mold"] = st["mold_id"] + np.where(st["mold_id"].isin(MEDICAL), " (medical)", "")
    st["Shots sorted"] = (st["sorted_shots"] / st["shots"]).map(lambda v: pct(v, 2))
    st["Review"] = st["review_mode"].map({"indexed_tray": "per shot", "per_shift": "per shift"})
    st["Sorted pieces found good"] = st["found_good"].map(lambda v: pct(v, 0))
    st["Good pieces sorted"] = st["good_pieces"].map(lambda v: f"{v:,.0f}")
    st["At standard cost"] = st["good_cost"].map(lambda v: f"${v:,.0f}")
    st = st[["Mold", "Review", "Shots sorted", "Sorted pieces found good", "Good pieces sorted", "At standard cost"]]
    gt = pd.DataFrame([dict(Group=g, Codes=GROUP_CODES[g], **{"Defective pieces": f"{lay['template_sort'][g]['n_defective']:,}",
                                                            "Template flagged": pct(lay['template_sort'][g]['rate_defective']),
                                                            "Above chance": f"{pts(lay['template_sort'][g]['above_chance'])} {ci(lay['template_sort'][g])}",
                                                            "Any layer flagged": pct(sum(inc[k][g]['rate_defective'] for k in ORDER)),
                                                            "Flagged by none": pct(1 - sum(inc[k][g]['rate_defective'] for k in ORDER))})
                       for g in GROUPS])
    c15 = C15["layer"]
    s3 = f"""
{section("template", "Section 3", "What the Template Detects")}
<p><span class="tag spc">SPC</span> The template flags <strong>{pct(T['rate_defective'])}</strong> of defective audited pieces and
<strong>{pct(T['rate_good'])}</strong> of good ones, <strong>{pts(T['above_chance'])} points above chance</strong>. Detection falls from fill volume
down to material, as the physics says it should: a short shot or flash changes the fill and pack integrals the unit watches, while a black speck or a
contamination leaves the curve untouched. Fill-volume defects are flagged on {pct(lay['template_sort']['Fill volume']['rate_defective'], 0)} of pieces;
material defects on no more than the good-piece rate.</p>
{chart("Defective audited pieces flagged, by defect group", group_chart(lay, inc),
       "Bars: defective pieces flagged by the template sort and by any layer. Black ticks: the share of good pieces flagged, the chance level for that bar.")}
<p>The groups the template cannot see are the groups no layer sees. Most material, gate and flow-front defects are flagged by nothing, because none
of the layers looks at anything but the curve and the machine values that move with it.</p>
{chart("Defective audited pieces flagged by no layer, by defect group", not_detected_chart(inc),
       "Share of each group's defective audited pieces whose shot no layer flagged, May 2025 to March 2026.")}
{table(gt, num_cols=("Defective pieces", "Template flagged", "Above chance", "Any layer flagged", "Flagged by none"))}
<p><strong>Fifteen-month context.</strong> Over all fifteen months the template is {pts(c15['template_sort']['All defects']['above_chance'])} points above
chance, the rules {pts(c15['rules']['All defects']['above_chance'])} and drift {pts(c15['drift']['All defects']['above_chance'])}. These figures are
context only and are not comparable with the table above: January to April 2025 holds {pct(C15['early_share_of_defective'], 0)} of all defective
audited pieces, from two new technicians approving runs before setup had converged and one January job that ran a low-flow resin lot on vents due
for cleaning. Those are fill-volume defects the template detects well, so including them lifts every layer.</p>
<p><strong>The sort.</strong> Over the fifteen months the units sorted <strong>{pct(sorted_share, 2)}</strong> of production shots. The reject-bin
review found <strong>{pct(fr, 0)}</strong> of sorted pieces good: <strong>{good_pieces:,.0f}</strong> good pieces in the reject bin,
<strong>${good_cost:,.0f}</strong> at standard cost. The medical molds sort into an indexed tray reviewed shot by shot; there, {pct(fr_tray, 0)} of
sorted shots held no defective piece. The other molds' bins are reviewed once per shift.</p>
{table(st, num_cols=("Shots sorted", "Sorted pieces found good", "Good pieces sorted", "At standard cost"))}
<p>The units' recorded alarm states agree with the platform's recomputation on <strong>{pct(S['unit_agreement'], 2)}</strong> of production shots, so
every layer below builds on the alarm logic the technicians see.</p>
"""

    # ── 4. extending SPC ──
    s4 = f"""
{section("spc", "Section 4", "What Extending SPC Added, Without Machine Learning")}
<p>Nothing in this section is machine learning. It runs the shop's own control-chart rules on every shot instead of the hourly audit sample, and adds
standard drift statistics.</p>
{section("rules", "4.1", "Control-chart rules on every shot", sub=True)}
<p><span class="tag spc">SPC</span> Consecutive shots are correlated: the melt, the mold and the material change slowly, so one shot looks much like
the last. The charts therefore plot each shot against what the previous shot predicts, which leaves close to independent points for rules 1 and 2.
Their deployed limits are <strong>4.05 sigma</strong> (rule 1) and <strong>2.70 sigma</strong> (rule 2), wider than the standard 3 and 2 sigma,
and chosen to meet the alarm budget: at the standard limits the rules would raise <strong>{textbook['episodes_per_shift']:.1f} alarm episodes per
shift</strong>, against {bud['rules']['episodes_per_shift']:.2f} at the deployed limits. Rules 4 and 5 are not run on every shot.</p>
<p>At the deployed limits the rules detect {pts(lay['rules']['All defects']['above_chance'])} points above chance on their own and
<strong>{pts(inc['rules']['All defects']['above_chance'])}</strong> beyond the template. Almost all of that is material defects
({pts(inc['rules']['Material']['above_chance'])} points): wet resin moves the fill from shot to shot without leaving the template's band.</p>
{section("drift", "4.2", "Drift detection", sub=True)}
<p><span class="tag spc">SPC</span> An <strong>EWMA</strong> (exponentially weighted moving average) smooths the fill and pack integrals so a step
stands out from shot-to-shot noise; a <strong>CUSUM</strong> (cumulative sum) adds up small deviations in one direction until they cross a limit, here on
end-of-fill pressure (vents), on pack-integral variance (check ring) and on gate seal time (mold temperature). Each resets at an approval, a documented
correction, a vent cleaning, PM, a ring replacement or a lot change. The deployed limits are the standard ones; after a detector fires, a new excursion
of the same detector within 3,000 shots is the same investigation unless a reset comes between. Drift raises
{bud['drift']['episodes_per_shift']:.2f} episodes per shift and adds {pts(inc['drift']['All defects']['above_chance'])} points of shot-level
detection beyond the template and rules.</p>
<p>A drift detector is meant to signal before defects appear, so each is also tested against the mechanism it was built for:</p>
<ul class="plain">
<li><strong>Fill EWMA at resin lot changes.</strong> It flags <strong>{pct(ls_['with_step']['share'], 0)}</strong>
[{pct(ls_['with_step']['ci'][0], 0)}, {pct(ls_['with_step']['ci'][1], 0)}] of the {ls_['with_step']['n']} mid-run lot changes that step the fill
integral, within a median {ls_['median_shots_to_flag']:.0f} shots, against {pct(ls_['without_step']['share'], 0)}
[{pct(ls_['without_step']['ci'][0], 0)}, {pct(ls_['without_step']['ci'][1], 0)}] of the {ls_['without_step']['n']} that do not. It works as a step detector.</li>
<li><strong>End-of-fill CUSUM and vent restriction.</strong> Its first firing after a vent cleaning comes before the rise in burns on
{pct(lv['preceded']['share'], 0)} of {lv['preceded']['n']} cleaning intervals, against {pct(lv['chance_preceded']['mean'], 0)} with the burn rise
placed at a random time. It fires about as early as chance would put it, a median {lv['lead']['median']:,.0f} shots ahead, so it is not timed to the vents.</li>
<li><strong>Variance CUSUM and check-ring wear.</strong> It flagged none of {rg['fired_before_defect']['n']} jobs run on a worn ring before their first
sink or void. Ring wear builds over months and barely moves within a job, while the detector's baseline is each job's own first 500 shots, so wear present
at the start is part of the baseline.</li>
</ul>
{chart("Drift detectors against the mechanisms they are built for", drift_tests_chart(DM),
       "Filled: the detector. Hollow: the comparison. Only the fill EWMA separates from its comparison.")}
"""

    # ── 5. ML ──
    vt = vbc.pivot_table(index="cell", columns="target", values=["rmse_over_gauge", "r2", "calibration_slope"])
    vt2 = pd.DataFrame({"Mold / press": vt.index,
                        "Weight error vs gauge": vt[("rmse_over_gauge", "weight")].map(lambda v: f"{v:.2f}x").values,
                        "Weight R2": vt[("r2", "weight")].map(lambda v: f"{v:.2f}").values,
                        "Dimension error vs gauge": vt[("rmse_over_gauge", "dimension")].map(lambda v: f"{v:.2f}x").values,
                        "Dimension R2": vt[("r2", "dimension")].map(lambda v: f"{v:.2f}").values})
    evt = ev.copy()
    evt["Event"] = evt["kind"].str.replace("_", " ")
    evt["Mold / press"] = evt["mold_id"] + " / " + evt["press_id"]
    evt["Start"] = evt["start"].dt.strftime("%d %b %Y %H:%M")
    evt["Flagged"] = evt["detected"].map({True: "yes", False: "no"})
    fmt = lambda v, sign=False: "" if v != v else (f"{v:+.0f}" if sign else f"{v:.0f}")
    evt["Minutes to first flag"] = evt["minutes_to_flag"].map(fmt)
    evt["Minutes to first template alarm"] = evt["minutes_to_template_alarm"].map(lambda v: "none" if v != v else f"{v:.0f}")
    evt["Flag lead over the shop's response (min)"] = evt["lead_over_response_min"].map(lambda v: fmt(v, True))
    evt = evt[["Event", "Mold / press", "Start", "Flagged", "Minutes to first flag", "Minutes to first template alarm",
               "Flag lead over the shop's response (min)"]]
    heater_txt = ""
    if heater is not None:
        heater_txt = (f" On the heater-zone failure it flagged {heater['minutes_to_flag']:.0f} minutes into the event, "
                      f"{heater['minutes_to_template_alarm'] - heater['minutes_to_flag']:.0f} minutes before the template, but "
                      f"{-heater['lead_over_response_min']:.0f} minutes after the shop had already responded, so on that event it would not have changed what "
                      f"the technician did.")
    s5 = f"""
{section("ml", "Section 5", "What Machine Learning Added", kind="ml")}
<p>Two model components were deployed. Both are retrained at the start of each month on everything before it, so each month is scored by the model
that would have been in use then.</p>
{section("vm", "5.1", "Virtual metrology", kind="ml", sub=True)}
<p><span class="tag ml">ML</span> <strong>Output.</strong> A predicted part weight and critical dimension for every shot and cavity.
<strong>How it is computed.</strong> A gradient-boosted regression trained on audited pieces linked to their shot, using that shot's summary values
normalized to the template, the press's machine-side values (including the mold temperatures and cycle time) and the context (lot, dryer, maintenance
counters, setpoints). It estimates the parts just made; it does not forecast future shots.</p>
<p><strong>As a measurement.</strong> Over the eleven months the dimension error is <strong>{vmo['dimension']['all']['rmse_over_gauge']:.2f} times</strong>
the gauge R&amp;R (R&sup2; {vmo['dimension']['all']['r2']:.2f}) and the weight error <strong>{vmo['weight']['all']['rmse_over_gauge']:.2f} times</strong>
(R&sup2; {vmo['weight']['all']['r2']:.2f}), medians over mold and press. In the warm months, June to September, the dimension error rises to
<strong>{vmo['dimension']['warm']['rmse_over_gauge']:.2f} times</strong> gauge against {vmo['dimension']['other']['rmse_over_gauge']:.2f} in the other
months, because summer chiller load and controller drift move the dimension through the mold temperature as well as the pack. Weight holds at
{vmo['weight']['warm']['rmse_over_gauge']:.2f} times. Calibration slopes are {vmo['dimension']['all']['calibration_slope']:.2f} (dimension) and
{vmo['weight']['all']['calibration_slope']:.2f} (weight): predictions are slightly compressed. Without the cavity pressure values the dimension error
rises to {vmo_abl['dimension']['all']['rmse_over_gauge']:.2f} times gauge and the weight error to {vmo_abl['weight']['all']['rmse_over_gauge']:.2f}.</p>
{chart("Virtual metrology error against gauge R&R, by month", vm_month_chart(R['vm_month']),
       "Median over mold and press of each month's error over the gauge's own error; dashed: the same model without cavity pressure values. "
       "Shaded: the warm months.")}
{table(vt2, num_cols=("Weight error vs gauge", "Weight R2", "Dimension error vs gauge", "Dimension R2"))}
<p><strong>What it supports.</strong> Weight predictions sit close to the gauge's own error and can support stretching audit intervals. Dimension
predictions are two to three times the gauge error: good enough to flag drift between audits, shot by shot, but not to replace the audits.</p>
{chart("Predicted against gauged critical dimension, May 2025 to March 2026", vm_scatter(R['vm_pred']),
       "Predictions follow the gauge across the tolerance band; the spread is the model's error plus the gauge's.")}
<p><strong>As an advisory audit.</strong> When a cavity's predicted dimension passes 75% of tolerance, an extra audit sample is requested; the
advisory never sorts a part. It raises {bud['virtual_metrology']['episodes_per_shift']:.2f} episodes per shift and adds
<strong>{pts(vm_pack['above_chance'])} points</strong> {ci(vm_pack)} on packing and shrinkage defects beyond every SPC layer, and
{pts(vm_dw_warm['above_chance'])} on dimensional and warp defects in the warm months, when the template misses parts moved by mold temperature.
Overall it adds {pts(inc['virtual_metrology']['All defects']['above_chance'])} points.</p>
{section("anomaly", "5.2", "Anomaly detection", kind="ml", sub=True)}
<p><span class="tag ml">ML</span> <strong>Output.</strong> An anomaly score for every shot, and an <strong>unusual</strong> state when two of the last
ten shots score above the threshold. <strong>How it is computed.</strong> An isolation forest per mold and press, trained only on validated good shots,
with no defect labels. It scores how far each shot's values have moved from the run's own recent shots, so a run sitting steadily off its template is
left to the template and the charts.</p>
<p><strong>Per event.</strong> {A['events']} novel events fall in May 2025 to March 2026; the model flagged all {A['flagged']}, {A['before_template']}
before the template, with a median lead of {A['median_lead_over_response_min']:.0f} minutes over the first confirmed defect or technician action.
On nozzle drool the template alarmed first.{heater_txt} It raises {A['episodes_per_shift']:.2f} episodes per shift.</p>
{table(evt, num_cols=("Minutes to first flag", "Minutes to first template alarm", "Flag lead over the shop's response (min)"))}
<p><strong>At the shot.</strong> Beyond the template, rules and drift it adds {pts(inc['anomaly']['All defects']['above_chance'])} points above chance
{ci(inc['anomaly']['All defects'])}: most of the shots it flags were already flagged by an earlier layer, and the events it exists for are rare.</p>
{section("ml-total", "5.3", "The two models together", kind="ml", sub=True)}
<p>Beyond every SPC layer the two models add <strong>{pts(ml_all['above_chance'])} points</strong> {ci(ml_all)} above chance overall, and
{pts(ml_warm['above_chance'])} in the warm months, almost all of it virtual metrology on packing, shrinkage and dimensional defects.</p>
"""

    # ── 6. shortfalls ──
    codes = R["codes"].pivot_table(index="defect_code", columns="source", values="qty", aggfunc="sum", fill_value=0)
    lab = codes.copy()
    lab["Linked to a shot"] = lab.get("sort", 0) + lab.get("audit", 0)
    lab = lab.sort_values("Linked to a shot", ascending=False).reset_index()
    lab["defect_code"] = lab["defect_code"].str.replace("_", " ")
    lab = lab.rename(columns={"defect_code": "Code", "sort": "Sort review", "audit": "Audit", "tally": "Packing tally", "return": "Returns"})
    lab = lab[["Code", "Sort review", "Audit", "Packing tally", "Returns", "Linked to a shot"]]
    for c in lab.columns[1:]:
        lab[c] = lab[c].map(lambda v: f"{v:,.0f}")
    ae = R["ae"]
    s6 = f"""
{section("short", "Section 6", "Where It Fell Short", kind="ml")}
{section("supervised", "6.1", "Supervised defect model: tested and not deployed", kind="ml", sub=True)}
<p>A gradient-boosted classifier trained on confirmed defects linked to a shot detects <strong>{pct(full['recall_model'])}</strong> of defective shots
against the template's <strong>{pct(full['recall_template'])}</strong> at the template's own alarm rate. Most of those labels exist because the template
sorted the shot and the reviewer confirmed it, so the model largely learns the template. Scored only on audit-found defects
({ao['n_test_pos']:,.0f} defective of {ao['n_test']:,.0f} audited shots in December to March), it detects <strong>{pct(ao['recall_model'])}</strong>
against the template's <strong>{pct(ao['recall_template'])}</strong> at the template's alarm rate ({pct(ao['alarm_rate'])}): no material gain.</p>
{chart("Recall at the template's alarm rate, December 2025 to March 2026", supervised_figure(full, ao),
       "On all shot-linked labels the model tracks the template; on defects found independently of the sort both detect far less, and the model's edge is small.")}
<p>Confirmed defective pieces by code and how they were found:</p>
{table(lab, num_cols=tuple(lab.columns[1:]))}
{section("blind", "6.2", "Defects the curve cannot see", kind="ml", sub=True)}
<p>{pct(notdet_all, 0)} of defective audited pieces were flagged by no layer. Material, gate and flow-front defects leave little or no trace in the cavity
pressure curve; no layer built on the curve can detect them, and the shop's visual checks remain the control for them.</p>
{section("drift-short", "6.3", "Drift detectors as early warnings", kind="ml", sub=True)}
<p>Only the fill EWMA does what it was built for. The vent CUSUM's first firing after a cleaning precedes the burn rise no more often than chance, and
the variance CUSUM cannot see check-ring wear that is already present when a job starts. The realism requirement that the end-of-fill CUSUM lead the burn
rise by 2,000 to 6,000 shots on at least 70% of vent-cleaning intervals is not met at the deployed limits: the lead falls in that range on
{pct(lv['lead_2000_6000']['share'], 0)} of intervals.</p>
{section("anomaly-short", "6.4", "Anomaly detection at the shot", kind="ml", sub=True)}
<p>The anomaly model flags every novel event, but adds {pts(inc['anomaly']['All defects']['above_chance'])} points at the shot: its flags mostly
fall on shots an earlier layer already flagged, and novel events are a few per year. A curve autoencoder was run as a comparison: it flagged
{pct(ae['event_flag_rate'], 1)} of retained novel-event curves against {pct(ae['other_flag_rate'], 1)} of other retained curves.</p>
"""

    # ── 7. budget and outputs ──
    bt_rows = pd.DataFrame([
        ("Template sort", "unchanged: the template's alarm bands", pct(bud["template_sort"]["shots_flagged"], 2),
         f"{bud['template_sort']['episodes_per_shift']:.2f}"),
        ("Control-chart rules 1 and 2", "4.05 and 2.70 sigma on every shot", pct(bud["rules"]["shots_flagged"], 2), f"{bud['rules']['episodes_per_shift']:.2f}"),
        ("Drift detection", "standard limits; one investigation per detector per 3,000 shots", pct(bud["drift"]["shots_flagged"], 2),
         f"{bud['drift']['episodes_per_shift']:.2f}"),
        ("Anomaly detection", "two of ten shots above a threshold set for 0.30% of validated shots", pct(bud["anomaly"]["shots_flagged"], 2),
         f"{bud['anomaly']['episodes_per_shift']:.2f}"),
        ("Virtual metrology advisory", "predicted dimension past 75% of tolerance", pct(bud["virtual_metrology"]["shots_flagged"], 2),
         f"{bud['virtual_metrology']['episodes_per_shift']:.2f}"),
        ("Added layers", "", "", f"{added:.2f}"),
    ], columns=["Layer", "Deployed threshold", "Shots flagged", "Episodes per shift"])
    uses = pd.DataFrame([
        ("Unit screen at the press", "Process technician", "Template status, the anomaly state, predicted weight and dimension against tolerance and the last audit"),
        ("Shift chart", "Technician, quality", "Per-shot values with EWMA and drift markers; the predicted-dimension trace with audit points"),
        ("Quality dashboard", "Quality engineer, weekly", "Alarms and sorts by mold and lot, false rejects, capability, the Pareto, maintenance requests from drift, audit trends"),
        ("Job record: the <a href=\"run_report_J-250165.html\">run report</a>", "Quality engineer at job close; customer evidence",
         "Per-job summary, first-shot approval, curve overlay, distributions, timeline, audit charts, capability, reject-bin review, reaction log"),
    ], columns=["Where", "Who acts on it", "What it shows"])
    s7 = f"""
{section("budget", "Section 7", "Alarm Budget and How the Outputs Are Used")}
<p>Thresholds are set for what one technician per shift can investigate: about one alarm episode per shift each for the rules and drift, and three to
four in total with the anomaly model and the virtual metrology advisory. The added layers raise <strong>{added:.2f}</strong> episodes per shift; the
template sort, which removes parts without an investigation, runs at {bud['template_sort']['episodes_per_shift']:.2f}.</p>
{table(bt_rows, num_cols=("Shots flagged", "Episodes per shift"), total_last=True)}
{chart("Alarm episodes per shift by added layer, May 2025 to March 2026", budget_figure(bud),
       f"The added layers sum to {added:.2f} episodes per shift; the dotted line marks one per shift.")}
{table(uses)}
<p>The outputs go on the monitoring unit's display where the vendor accepts additions, and on an MES companion view beside the press otherwise.</p>
{section("limits", "7.1", "Other limitations", sub=True)}
<ul class="plain">
<li>Shot-level detection rests on audit pieces, {SL['n_defective']:,} of them defective in the evaluation period; small groups carry wide intervals.</li>
<li>A detection credits a layer for flagging the shot a defective piece came from; it does not show that the alarm would have prevented the defect.</li>
<li>Returns arrive weeks after the job; a job's figures are final only 21 days after its last shot.</li>
<li>Virtual metrology is trained on the six molds and two presses here; a new mold needs its own audit history before its predictions are used.</li>
</ul>
"""

    toc = [("summary", "1 · Executive Summary", False), None,
           ("cell", "2 · The Cell and Quality Control", False), ("curve", "One Shot's Curve", True), ("unit", "Unit, Sort and Audits", True),
           ("measure", "How Detection Is Measured", True), None,
           ("template", "3 · What the Template Detects", False), None,
           ("spc", "4 · Extending SPC", False), ("rules", "Rules on Every Shot", True), ("drift", "Drift Detection", True), None,
           ("ml", "5 · Machine Learning", False), ("vm", "Virtual Metrology", True), ("anomaly", "Anomaly Detection", True),
           ("ml-total", "The Two Models Together", True), None,
           ("short", "6 · Where It Fell Short", False), ("supervised", "Supervised Model", True), ("blind", "What the Curve Cannot See", True),
           ("drift-short", "Drift as Early Warning", True), ("anomaly-short", "Anomaly at the Shot", True), None,
           ("budget", "7 · Alarm Budget and Use", False), ("limits", "Other Limitations", True)]
    nav = '<nav class="toc"><div class="toc-title">Contents</div>' + "".join(
        "<hr>" if t is None else f'<a href="#{t[0]}"{" class=\"sub\"" if t[2] else ""}>{t[1]}</a>' for t in toc) + "</nav>"
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Overview: Molding Cell Cavity Pressure Monitoring</title><style>{CSS}</style></head>
<body>
<div class="page-header"><h1>ML Overview: Molding Cell Cavity Pressure Monitoring</h1>
<div class="sub">IM-11 and IM-12 · January 2025 to March 2026 · every layer evaluated May 2025 to March 2026</div></div>
<div class="layout">{nav}<main class="content">
{s1}{s2}{s3}{s4}{s5}{s6}{s7}
</main></div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
