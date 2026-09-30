"""
ML overview for the quality manager.

Usage: python ml/reports/generate_ml_overview.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports import results as RS  # noqa: E402
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREEN, GREY, LIGHT_BLUE, RED, fig, img, pct, shell, table  # noqa: E402
from ml.src.features import VALID_END  # noqa: E402

OUT = Path(__file__).resolve().parent / "ml_overview.html"
LAYER_COLORS = [BRAND_BLUE, ACCENT, LIGHT_BLUE, GREEN, AMBER, GREY]


def drift_example():
    """A vent cleaning interval on an end-of-fill mold: pressure decline, the CUSUM signal, burns tallied."""
    sig = RS.q("""
        select d.job_id, d.mold_id, d.shot_ts from drift_signals d
        where d.detector = 'cusum_eof' and d.mold_id in ('M-2041', 'M-2043')
        order by d.shot_ts""")
    burns = RS.q("""select job_id, hour_ts, sum(qty) as burns from fct_confirmed_defects
                    where defect_code = 'burn' and source = 'tally' group by 1, 2""")
    best = None
    for r in sig.itertuples():
        b = burns[burns["job_id"] == r.job_id]
        if b["burns"].sum() >= 10:
            best = r
            break
    if best is None:
        best = sig.iloc[0]
    s = RS.q(f"""select shot_ts, eof_pressure_dev, shots_since_vent_cleaning from fct_shot
                 where job_id = '{best.job_id}' and after_approval order by shot_ts""")
    b = burns[burns["job_id"] == best.job_id]
    return best, s, b


def main():
    R = RS.load()
    S = R["shots"]
    vmh = RS.vm_headline(R)
    L = R["layers"]
    lay = RS.layer_table(R)
    dt = RS.defect_table(R)
    any_ = dt[dt["target"] == "y_any"].iloc[0]
    link = RS.linkage(R)
    removal = RS.check(R, "sort removal")["value"]
    sorted_share = S["sorted_shots"] / S["prod_shots"]
    fr = R["sort"]["false_reject_share"]
    budget = L["alarms_per_shift"]
    an = R["anomaly"]
    ev = R["anomaly_events"]
    ev_test = ev[ev["split"] == "test"]
    pts = R["spc_points"]
    abl = R["vm_abl"].groupby(["target", "tag"])["rmse_over_gauge"].median()
    months = (pd.Timestamp(S["last_ts"]).to_period("M") - pd.Timestamp(S["first_ts"]).to_period("M")).n + 1
    added = {k: L["share"].get(k, 0) for k in ("rules", "drift", "anomaly")}
    top_new = max(added, key=added.get)
    top_label = {"rules": "the control-chart rules", "drift": "drift detection", "anomaly": "the anomaly layer"}[top_new]
    tech = budget["technician_total"]
    budget_phrase = ("inside the three to six the shop can work" if 3 <= tech <= 6 else
                     "below the three to six the shop can work, which leaves room to loosen a threshold" if tech < 3 else
                     "above the three to six the shop can work")
    vm_dim = L["dimensional_share"].get("virtual_metrology", 0.0)
    dim_tmpl = L["dimensional_share"].get("template_sort", 0.0)

    # ── layer chart ──
    f, ax = fig(1.9)
    left = 0
    for (lbl, v), c in zip(zip(lay["layer"], lay["share"]), LAYER_COLORS):
        ax.barh([0], [v], left=left, color=c, edgecolor="white")
        if v > 0.035:
            ax.text(left + v / 2, 0, f"{v * 100:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
    ax.legend([p for p in ax.patches], lay["layer"], ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.35), loc="upper center", fontsize=9)
    layer_png = img(f, "layer comparison")

    # ── alarm budget chart ──
    names = [("template_sort", "Template alarm hours\n(sort is automatic)"), ("rules", "Rules 1 and 2"), ("drift", "Drift onsets"),
             ("anomaly", "Anomaly (2 in 10)"), ("virtual_metrology", "Virtual metrology\ndimension flags")]
    f, ax = fig(3.0)
    ax.bar([n for _, n in names], [budget.get(k, 0) for k, _ in names], color=[GREY, ACCENT, LIGHT_BLUE, AMBER, GREEN])
    ax.axhspan(3, 6, color="#E8F0E8", zorder=0)
    ax.set_ylabel("investigations per shift")
    ax.tick_params(axis="x", labelsize=9)
    budget_png = img(f, "alarm budget")

    # ── drift example ──
    best, ds, db = drift_example()
    f, ax = fig(3.4)
    ax.plot(ds["shot_ts"], ds["eof_pressure_dev"].rolling(50, min_periods=10).mean(), color=BRAND_BLUE, lw=1.3,
            label="end-of-fill pressure, 50-shot mean (fraction of alarm band)")
    ax.axvline(best.shot_ts, color=RED, lw=1.2, ls="--", label="CUSUM signal")
    ax2 = ax.twinx()
    ax2.bar(db["hour_ts"], db["burns"], width=0.035, color=AMBER, alpha=0.8, label="burns tallied per hour")
    ax2.spines["top"].set_visible(False)
    ax.set_ylabel("deviation from template (bands)")
    ax2.set_ylabel("burns per hour")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=9)
    drift_png = img(f, "drift example")
    lead_h = (db[db["hour_ts"] >= best.shot_ts]["hour_ts"].min() - best.shot_ts).total_seconds() / 3600 if len(db) else np.nan

    # ── virtual metrology chart ──
    p = R["vm_pred"]
    p = p[(p["design"] == "pooled") & (p["target"] == "dimension")]
    p = p.assign(meas=(p["measured"] - p["nominal"]) / p["tolerance"], pred=(p["predicted"] - p["nominal"]) / p["tolerance"])
    f, ax = fig(3.6, 5.2)
    ax.scatter(p["pred"], p["meas"], s=2, alpha=0.15, color=ACCENT)
    ax.plot([-1, 1], [-1, 1], color=BRAND_BLUE, lw=1)
    for y in (-1, 1):
        ax.axhline(y, color=RED, lw=0.8, ls="--")
    ax.set_xlabel("predicted, fraction of tolerance")
    ax.set_ylabel("gauged, fraction of tolerance")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    vm_png = img(f, "virtual metrology")

    # ── defect chart ──
    d = dt[dt["target"].isin(["y_any", "y_short_shot", "y_flash", "y_sink", "y_dimensional", "y_weld_line", "y_burn"])]
    f, ax = fig(3.2)
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["ap_template"], 0.4, color=GREY, label="template limits")
    ax.bar(x + 0.2, d["ap_model"], 0.4, color=BRAND_BLUE, label="supervised model")
    ax.set_xticks(x, d["code"])
    ax.set_ylabel("average precision, test period")
    ax.legend(frameon=False)
    defect_png = img(f, "supervised defect")

    kp = [("Shots on the cell", f"{S['shots']:,.0f}"), ("Months of history", f"{months}"),
          ("Shots sorted", pct(sorted_share, 2)), ("Sorted shots found good", pct(fr, 0)),
          ("Confirmed defects caught by some layer", pct(1 - L["share"].get("none", 0), 0)),
          ("Technician alarms per shift", f"{budget['technician_total']:.1f}")]
    kpis = '<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in kp) + "</div>"
    lt = lay.assign(share=lay["share"].map(lambda v: f"{v * 100:.1f}%"),
                    alarms_per_shift=lay["alarms_per_shift"].map(lambda v: "" if v != v else f"{v:.2f}"))
    lt.columns = ["Layer", "Share of confirmed defects (test period)", "Alarms per shift"]

    body = f"""
<div class="note">The cell's two monitoring units already catch most of what the curve can tell. Keeping every shot, charting it and
adding drift detection raised the share of confirmed defects with an alarm from {pct(L['share'].get('template_sort', 0), 0)} to
{pct(1 - L['share'].get('none', 0), 0)} at {budget['technician_total']:.1f} technician alarms per shift. The per-shot weight and dimension
prediction is the model that earns its place; the supervised defect model does not beat the template limits on common codes.</div>
{kpis}
<p>Every quality figure here carries its linkage coverage: {pct(link, 0)} of confirmed defective pieces can be tied to a shot (the sort
review and the robot-linked audits); packing tallies and returns are tied to a job and hour only. Test period: December 2025 to March 2026,
after training through September and validation in October and November.</p>

<h2 id="units">What the units already caught</h2>
<p>The template alarms and the sort remove a real share of defects and cost the shop false rejects in return. Over the fifteen months the
units sorted {pct(sorted_share, 2)} of production shots; the reject-bin review found {pct(fr, 0)} of those shots good, and the sorted shots
held {pct(removal, 0)} of all defective pieces. The rest of the defects either carried no curve signal or sat inside the bands.</p>
<p>The units' recorded alarm states agree with the platform's recomputation from the summary values and templates on
{pct(S['unit_agreement'], 2)} of production shots, so the layers below build on the same alarm logic the technicians see.</p>

<h2 id="every-shot">What charting every shot added</h2>
<p>Running the audit chart's rules on every shot floods the technician unless the rule set is cut down. With limits from the first 500 shots
after approval, rule 5 (eight in a row on one side) fires on {pct(pts['we5'], 0)} of points and rule 4 on {pct(pts['we4'], 0)}: cavity
pressure values drift slowly and are correlated from shot to shot, which those rules read as a shift. Rules 1 and 2 alone raise
{budget['rules']:.2f} alarms per shift and add {L['share'].get('rules', 0) * 100:.1f} points of confirmed defects to what the template caught.</p>

<h2 id="drift">Drift detection and the maintenance signals</h2>
<p>Drift detection warns early because the slow mechanisms move the process for hours before parts go bad. EWMA on the pack
and fill integrals and CUSUMs on end-of-fill pressure, pack-integral variance and gate seal time raise {budget['drift']:.2f} signal onsets per
shift and add {L['share'].get('drift', 0) * 100:.1f} points. The chart below follows one job on {best.mold_id}: end-of-fill pressure declines
as the vents load, the CUSUM signals, and the burns tallied at packing follow{'' if lead_h != lead_h else f' about {lead_h:.0f} hours later'}.</p>
{drift_png}
<p class="caption">Job {best.job_id}. The CUSUM resets at each approval, drift correction, vent cleaning, PM, ring replacement and lot change.</p>
<p>The same layer carries the maintenance signals: the check ring's leakage shows as pack-integral variance on IM-12 until the ring was replaced,
and resin lot changes step the fill integral at the load (lot-step tests flag {int((R['lot_steps']['step_t'].abs() > 3).sum())} of
{len(R['lot_steps'])} lot changes inside a job).</p>

<h2 id="vm">The per-shot measurement</h2>
<p>Virtual metrology works because its labels are abundant: {R['audits']['pieces']:,.0f} audited pieces, {pct(R['audits']['linked'], 0)}
of them linked to their shot by the robot's recorded cycle. On the test period the pooled model predicts part weight within
{vmh[('pooled', 'weight')]['ratio']:.2f} times the gauge R&amp;R and the critical dimension within {vmh[('pooled', 'dimension')]['ratio']:.2f} times,
with R&sup2; of {vmh[('pooled', 'weight')]['r2']:.2f} and {vmh[('pooled', 'dimension')]['r2']:.2f} (medians over mold and press, five seeds).</p>
{vm_png}
<p class="caption">Test-period audit pieces, pooled model. Dashed lines are the drawing tolerance.</p>
<p>The value is between audits: a predicted weight and dimension on every shot instead of one sample an hour. As an alarm it added
{vm_dim * 100:.1f} points on dimensional codes in the test period, because the winter months carried little mold-temperature drift and
{pct(dim_tmpl, 0)} of the dimensional defects fell on shots the template had already alarmed on. Its use there is the audit plan: where the
prediction sits within about 1.5 times the gauge error, the shop could stretch the audit interval on that mold (per mold in the technical report). Without the cavity pressure values the dimension error rises to
{abl.get(('dimension', 'no_cavity_signal'), np.nan):.1f} times gauge, which is the in-mold signal's value in one number.</p>

<h2 id="supervised">The supervised defect model</h2>
<p>A model trained on confirmed defects does not beat the template limits on the common codes, and this section is kept to say so. At the
template's own alarm rate it catches {pct(any_['recall_model'], 1)} of confirmed defective shots against the template's
{pct(any_['recall_template'], 1)}. It ranks sink and dimensional defects better than the band does, because the pack integral carries graded
information the band throws away. On the rare codes the label counts are too small for anything usable.</p>
{defect_png}

<h2 id="anomaly">The anomaly layer</h2>
<p>The anomaly model covers what no rule was written for. It flagged {an['detected']} of {an['events']} novel events (a failing check ring,
a failing heater zone, nozzle drool, a wrong material), {len(ev_test)} of them in the test period; {pct(an['share_on_template_pass'], 0)} of its
flags fall on shots the template passed. With two flags in ten shots required, it raises {budget['anomaly']:.2f} alarms per shift and adds
{L['share'].get('anomaly', 0) * 100:.1f} points.</p>

<h2 id="layers">The layer comparison</h2>
<p>Confirmed defects in the test period, credited at the job-hour grain to the first layer that alarmed on that job in the hour or shortly before:</p>
{layer_png}
{table(lt)}
<p class="caption">{L['n_defects']:,} confirmed defective pieces (sort review, audits and packing tallies) in {L['n_test_hours']:,} production hours.</p>

<h2 id="budget">The alarm budget</h2>
<p>Thresholds are set for what one technician per shift can investigate. The deployed rule set, drift onsets and the anomaly layer together
raise {tech:.1f} alarms per shift, {budget_phrase}; the sort runs on its own. Of the new layers, {top_label} added the most
({added[top_new] * 100:.1f} points).</p>
{budget_png}

<h2 id="limits">Limitations</h2>
<p>Labels for rare codes are too few to train on. Packing tallies carry no shot, so the layer comparison works at the job-hour grain and credits a
layer for alarming near a defect, not for the defect itself. {pct(L['share'].get('none', 0), 0)} of confirmed defects had no alarm from any layer;
most carry no curve signal (handling, contamination, specks) and stay unexplained by design. The anomaly threshold is set on the fit window
(0.45%); on validated test-period shots the flag rate was {pct(an['false_positive_rate'], 2)}. The curve autoencoder, run as a comparison,
did not separate the novel events from other retained curves; the technical report gives its numbers.</p>
"""
    toc = [("units", "Units"), ("every-shot", "Every shot"), ("drift", "Drift"), ("vm", "Measurement"), ("supervised", "Supervised"),
           ("anomaly", "Anomaly"), ("layers", "Layers"), ("budget", "Budget"), ("limits", "Limitations")]
    OUT.write_text(shell("Cavity Pressure ML Overview", "Molding quality · IM-11 and IM-12",
                         f"January 2025 to March 2026 · test period December 2025 to March 2026", body, toc), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
