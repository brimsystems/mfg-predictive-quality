"""
Quality dashboard: plant-wide scrap by press, mold and code, and the instrumented cell.

Plant section: scrap quantity and cost by month, press, mold and code from the QMS
job-level entries across all 24 presses. Cell section (IM-11, IM-12): alarms by layer
per week, drift signals open, sort and false-reject rates by mold, virtual metrology
error against gauge, label maturity and linkage coverage.

Usage: python analytics/reports/generate_dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREY, LIGHT_BLUE, RED, fig, img, pct, shell, table  # noqa: E402
from ml.src.features import DATA_DIR, connect  # noqa: E402

OUT = Path(__file__).resolve().parent / "dashboard.html"


def main():
    con = connect()
    scrap = con.execute("select * from mart_plant_scrap").df()
    output = con.execute("select * from mart_plant_output").df()
    jobs = con.execute("select * from fct_job").df()
    jh = con.execute("select * from fct_job_hour").df()
    alarms = con.execute("select job_id, mold_id, shot_ts, rule from spc_alarms").df()
    drift = con.execute("select job_id, mold_id, shot_ts, detector from drift_signals").df()
    sort_rev = con.execute("""
        select s.mold_id, count(*) as sorted_shots, avg((r.pieces_confirmed_defective = 0)::int) as false_reject_share
        from stg_qms__sort_dispositions r join fct_shot s using (shot_id) group by 1 order by 1""").df()
    link = con.execute("""
        select source, sum(qty) as qty, sum(case when shot_id is not null then qty else 0 end) as linked
        from fct_confirmed_defects group by 1""").df()
    snap = con.execute("select max(shot_ts) from fct_shot").fetchone()[0]
    con.close()

    # ── Plant KPIs ── (months through the last production month; later scrap entries are late postings)
    end_m = pd.Timestamp(snap).to_period("M").to_timestamp()
    scrap = scrap[scrap["month"] <= end_m]
    output = output[output["month"] <= end_m]
    last_m = scrap["month"].max()
    cur = scrap[scrap["month"] == last_m]
    prev = scrap[scrap["month"] == last_m - pd.DateOffset(months=1)]
    insp = output[output["month"] == last_m]
    kpis = [("Scrap cost, " + pd.Timestamp(last_m).strftime("%b %Y"), f"${cur['scrap_cost'].sum():,.0f}"),
            ("Prior month", f"${prev['scrap_cost'].sum():,.0f}"),
            ("Pieces scrapped", f"{cur['quantity_scrapped'].sum():,.0f}"),
            ("Final inspection fail rate", pct(insp['quantity_failed'].sum() / max(insp['quantity_inspected'].sum(), 1), 2))]
    kpi_html = '<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in kpis) + "</div>"

    m = scrap.groupby("month")["scrap_cost"].sum()
    f, ax = fig(3.2)
    ax.bar(m.index, m.values, width=20, color=BRAND_BLUE)
    ax.set_ylabel("scrap cost ($)")
    ax.set_title("Plant scrap cost by month")
    c1 = img(f, "plant scrap by month")

    bp = scrap.groupby("press_id")["scrap_cost"].sum().sort_values(ascending=False)
    f, ax = fig(3.4)
    ax.bar(bp.index, bp.values, color=[AMBER if p in ("IM-11", "IM-12") else ACCENT for p in bp.index])
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Scrap cost by press (cell presses in amber)")
    c2 = img(f, "scrap by press")

    bc = scrap.groupby("defect_code")["quantity_scrapped"].sum().sort_values()
    f, ax = fig(3.4)
    ax.barh(bc.index, bc.values, color=ACCENT)
    ax.set_title("Pieces scrapped by code, all presses")
    c3 = img(f, "scrap by code")

    top_molds = (scrap.groupby(["mold_id", "press_id"])[["quantity_scrapped", "scrap_cost"]].sum()
                 .sort_values("scrap_cost", ascending=False).head(12).reset_index())

    # ── Cell section ──
    wk = lambda s: pd.to_datetime(s).dt.to_period("W").dt.start_time
    by_layer = pd.DataFrame({
        "sorted shots": jh.groupby(wk(jh["hour_ts"]))["sorted_shots"].sum(),
        "control-chart rules 1 and 2": alarms[alarms["rule"].isin(["WE1", "WE2"])].groupby(wk(alarms["shot_ts"])).size(),
        "drift signals": drift.groupby(wk(drift["shot_ts"])).size(),
    }).fillna(0)
    f, ax = fig(3.4)
    for col, c in zip(by_layer.columns, [BRAND_BLUE, ACCENT, AMBER]):
        ax.plot(by_layer.index, by_layer[col], color=c, lw=1.5, label=col)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Cell alarms by layer per week (log scale)")
    c4 = img(f, "alarms by layer")

    last_seg = drift.sort_values("shot_ts").groupby(["job_id", "detector"]).tail(1)
    open_jobs = jobs[jobs["last_shot_ts"] >= snap - pd.Timedelta(days=7)]["job_id"]
    open_sig = last_seg[last_seg["job_id"].isin(open_jobs)].groupby("detector").size().rename("signals in jobs of the last 7 days")

    vm = pd.read_csv(DATA_DIR / "results" / "vm_metrics_full.csv")
    vm = (vm[vm["design"] == "pooled"].groupby(["target", "cell"])[["rmse", "gauge_sd", "rmse_over_gauge", "r2"]].mean()
          .reset_index())

    mat = jobs.assign(state=np.where(jobs["is_matured"], "matured (21 days after last shot)", "provisional"))
    maturity = mat.groupby("state").size().rename("jobs").reset_index()
    link["coverage"] = link["linked"] / link["qty"]

    body = f"""
<h2 id="plant">Plant scrap, all presses</h2>
<p>Plant scrap is read from the QMS job-level entries on all 24 presses; the two cell presses are shown in amber.
The cell's shot-level view follows below.</p>
{kpi_html}{c1}{c2}{c3}
<h3>Highest-cost mold and press combinations</h3>
{table(top_molds.rename(columns={"mold_id": "Mold", "press_id": "Press", "quantity_scrapped": "Pieces scrapped", "scrap_cost": "Scrap cost"}), {"Pieces scrapped": lambda v: f"{v:,.0f}", "Scrap cost": lambda v: f"${v:,.0f}"})}
<h2 id="cell">Instrumented cell, IM-11 and IM-12</h2>
<p>The template alarms and sort carry the week-to-week load; the control-chart rules and drift signals are the layers
added on every shot. Rules 4 and 5 of the audit chart are not deployed on every shot (see the ML overview for why).</p>
{c4}
<h3>Sort and false rejects by mold</h3>
{table(sort_rev.rename(columns={"mold_id": "Mold", "sorted_shots": "Sorted shots", "false_reject_share": "Found good at review"}), {"Sorted shots": lambda v: f"{v:,.0f}", "Found good at review": pct})}
<h3>Drift signals open</h3>
{table(open_sig.reset_index().rename(columns={'index': 'Detector', 'detector': 'Detector'}))}
<h3>Virtual metrology error against gauge (test period, pooled model)</h3>
{table(vm.rename(columns={"target": "Target", "cell": "Mold / press", "rmse": "RMSE", "gauge_sd": "Gauge sd", "rmse_over_gauge": "RMSE over gauge", "r2": "R2"}), {"RMSE": lambda v: f"{v:.4f}", "Gauge sd": lambda v: f"{v:.4f}", "RMSE over gauge": lambda v: f"{v:.2f}x", "R2": lambda v: f"{v:.2f}"})}
<h3>Label maturity and linkage coverage</h3>
{table(maturity.rename(columns={"state": "Label state", "jobs": "Jobs"}))}
{table(link.rename(columns={"source": "Found by", "qty": "Confirmed pieces", "linked": "Linked to a shot", "coverage": "Coverage"}), {"Confirmed pieces": lambda v: f"{v:,.0f}", "Linked to a shot": lambda v: f"{v:,.0f}", "Coverage": pct})}
<p class="caption">Linkage coverage: the share of confirmed defective pieces whose outcome is tied to a shot. Packing tallies and
returns carry no shot; the sort review and linked audits do.</p>
"""
    OUT.write_text(shell("Quality Dashboard", "Molding quality", f"Data through {pd.Timestamp(snap):%d %B %Y}",
                         body, toc=[("plant", "Plant"), ("cell", "Cell")]), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
