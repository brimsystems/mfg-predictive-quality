"""
MLOps monitoring: label maturity, alarm-rate drift per layer, expected feature shifts after
equipment events, and virtual metrology error tracked against each audit as it arrives,
with the investigation rule between the monthly retrains.

Usage: python ml/reports/generate_monitoring_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports import results as RS  # noqa: E402
from ml.reports.style import ACCENT, AMBER, BRAND_BLUE, GREEN, GREY, LIGHT_BLUE, RED, fig, img, pct, shell, table  # noqa: E402
EVAL_START = pd.Timestamp("2025-05-01")

OUT = Path(__file__).resolve().parent / "monitoring_report.html"
RETRAIN = {"weight": 1.5, "dimension": 3.0}      # rolling error over gauge that triggers an investigation before the monthly retrain
WINDOW = 200                                     # audit pieces in the rolling window


def maturity():
    """Share of each job's final confirmed defects known 1, 7 and 21 days after its last shot."""
    d = RS.q("""
        with j as (select job_id, last_shot_ts from fct_job),
        e as (
            select r.job_id, r.reviewed_ts as known_ts, r.pieces_confirmed_defective as qty
            from stg_qms__sort_dispositions r
            union all select job_id, entered_ts, qty from stg_qms__scrap_tallies
            union all select a.job_id, a.audit_ts, 1 from int_audit_pieces_linked a where a.defect_code is not null
            union all select attributed_job_id, received_date, qty from stg_qms__customer_returns where attributed_job_id is not null)
        select e.job_id, datediff('hour', j.last_shot_ts, e.known_ts) / 24.0 as days_after, e.qty
        from e join j using (job_id)""")
    tot = d["qty"].sum()
    return {k: float(d.loc[d["days_after"] <= k, "qty"].sum() / tot) for k in (0, 1, 7, 21, 60)}


def weekly_layers(R):
    wk = lambda s: pd.to_datetime(s).dt.to_period("W").dt.start_time
    shots = RS.q("select date_trunc('hour', shot_ts) as h, count(*) as n from fct_shot where after_approval group by 1")
    hrs = shots.groupby(wk(shots["h"])).size() / 8.0
    srt = RS.q("select job_id, date_trunc('hour', shot_ts) as h from fct_shot where unit_alarm_state = 'alarm' and after_approval group by 1, 2")
    rul = RS.q("select job_id, date_trunc('hour', shot_ts) as h from spc_alarms where rule in ('WE1', 'WE2') group by 1, 2")
    dr = RS.q("select job_id, date_trunc('hour', shot_ts) as h from drift_signals group by 1, 2")
    out = pd.DataFrame({"template alarms": srt.groupby(wk(srt["h"])).size(), "rules 1 and 2": rul.groupby(wk(rul["h"])).size(),
                        "drift onsets": dr.groupby(wk(dr["h"])).size()}).fillna(0)
    return out.div(hrs, axis=0).dropna()


def event_shifts():
    """Mean change in normalized features over the 500 shots after an event against the 500 before."""
    ev = RS.q("""
        select 'vent cleaning' as event, mold_id, null as press_id, event_ts from stg_toolroom__mold_maintenance where event_type = 'vent_cleaning'
        union all select 'sensor recalibration', mold_id, null, event_ts from stg_toolroom__mold_maintenance where event_type = 'sensor_recalibration'
        union all select 'ring replacement', null, press_id, event_ts from stg_toolroom__equipment_service where equipment = 'check_ring'
        union all select 'template re-established', mold_id, press_id, approval_ts from fct_job""")
    s = RS.q("""select shot_id, mold_id, press_id, shot_ts, pg_pack_dev, pg_fill_dev, eof_pressure_dev, cushion_mm, pg_gate_seal_dev
                from fct_shot where after_approval order by shot_ts""")
    feats = ["pg_pack_dev", "pg_fill_dev", "eof_pressure_dev", "cushion_mm", "pg_gate_seal_dev"]
    rows = []
    for e in ev.itertuples():
        has_mold, has_press = pd.notna(e.mold_id), pd.notna(e.press_id)
        g = s[s["mold_id"] == e.mold_id] if has_mold else s[s["press_id"] == e.press_id]
        if has_mold and has_press:
            g = g[g["press_id"] == e.press_id]
        i = g["shot_ts"].searchsorted(pd.Timestamp(e.event_ts))
        b, a = g.iloc[max(0, i - 500):i], g.iloc[i:i + 500]
        if len(b) < 300 or len(a) < 300:
            continue
        r = {"event": e.event}
        for f in feats:
            sd = b[f].std()
            r[f] = abs(a[f].mean() - b[f].mean()) / sd if sd and sd == sd else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).groupby("event")[feats].median()
    t["events"] = pd.DataFrame(rows).groupby("event").size()
    return t.reset_index()


def vm_tracking(R):
    p = R["vm_pred"]
    p = p.sort_values("shot_ts").copy()
    parts = R["parts"].set_index("mold_id")
    p["mold_id"] = p["cell"].str.split(" / ").str[0]
    p["gauge"] = np.where(p["target"] == "weight", p["mold_id"].map(parts["gauge_sd_weight_g"]),
                          p["mold_id"].map(parts["gauge_sd_dimension_mm"]))
    p["z"] = (p["measured"] - p["predicted"]) / p["gauge"]
    out = []
    for (t, cell), g in p.groupby(["target", "cell"]):
        g = g.sort_values("shot_ts").copy()
        g["rolling_ratio"] = np.sqrt((g["z"] ** 2).rolling(WINDOW, min_periods=WINDOW // 2).mean())
        over = (g["rolling_ratio"] > RETRAIN[t]).astype(int)
        g["triggered"] = over.rolling(2 * WINDOW, min_periods=2 * WINDOW).min() == 1
        out.append(g)
    return pd.concat(out)


def main():
    R = RS.load()
    mat = maturity()
    wl = weekly_layers(R)
    es = event_shifts()
    tr = vm_tracking(R)

    f, ax = fig(3.2)
    for c, col in zip(wl.columns, [GREY, ACCENT, AMBER]):
        ax.plot(wl.index, wl[c], label=c, color=col, lw=1.4)
    ax.axvline(EVAL_START, color="#999", ls=":", lw=1)
    ax.set_ylabel("alarm hours per shift")
    ax.legend(frameon=False)
    layers_png = img(f, "alarm rate by week")

    f, axes = fig(3.0, 9.2, ncols=2)
    for ax, t in zip(axes, ["weight", "dimension"]):
        for cell, g in tr[tr["target"] == t].groupby("cell"):
            ax.plot(g["shot_ts"], g["rolling_ratio"], lw=0.9, alpha=0.8, label=cell)
        ax.axhline(RETRAIN[t], color=RED, ls="--", lw=1)
        ax.set_title(f"{t} (retrain above {RETRAIN[t]}x)", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel(f"rolling RMSE over gauge ({WINDOW} pieces)")
    for ax in axes:
        ax.axvspan(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-10-01"), color="#FBEBD3", alpha=0.5, lw=0)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    track_png = img(f, "virtual metrology tracking")
    trig = tr[tr["triggered"]].groupby(["target", "cell"])["shot_ts"].min().reset_index()
    trig_text = ("did not trigger" if trig.empty else "triggered on " + "; ".join(
        f"{r.target} for {r.cell} from {pd.Timestamp(r.shot_ts):%d %b %Y}" for r in trig.itertuples()))

    esf = es.copy()
    for c in ["pg_pack_dev", "pg_fill_dev", "eof_pressure_dev", "cushion_mm", "pg_gate_seal_dev"]:
        esf[c] = esf[c].map(lambda v: "" if v != v else f"{v:.2f}")
    esf.columns = ["Event", "Pack integral", "Fill integral", "End-of-fill pressure", "Cushion", "Gate seal", "Events"]

    matt = pd.DataFrame([(f"{k} day{'s' if k != 1 else ''}" if k else "at the last shot", pct(v, 0)) for k, v in mat.items()],
                        columns=["Time after a job's last shot", "Share of final confirmed defects known"])
    jobs = R["jobs"]
    prov = int((~jobs["is_matured"]).sum())

    body = f"""
<div class="note">Virtual metrology and the anomaly model are retrained at the start of every month on all earlier data. Three things are watched
between retrains: whether labels are complete enough to score against, whether each alarm layer's rate is drifting, and whether the per-shot
measurement still sits within its gauge-error band. The investigation rule acts only on matured labels.</div>

<h2 id="maturity">Label maturity</h2>
<p>Most of a job's confirmed defects are known within a day of its last shot, because the sort review and the packing tallies close within the shift;
returns arrive weeks to months later and are why a job is scored only after {21} days. {prov} of {len(jobs)} jobs are still provisional at the
extract date.</p>
{table(matt)}

<h2 id="alarm-drift">Alarm-rate drift per layer</h2>
<p>Alarm hours per shift by week, for the template alarms, the rules at their deployed limits (4.05 and 2.70 sigma) and drift onsets. A layer's rate
that moves outside its range for two weeks is investigated before any threshold is changed; the dotted line marks the start of the evaluation period,
May 2025.</p>
{layers_png}

<h2 id="shifts">Expected feature shifts after equipment events</h2>
<p>Some shifts in the inputs are expected and must not be read as model drift. The table gives the median shift (in standard deviations of the 500
shots before) over the 500 shots after each event type: template re-establishment recentres the cavity values; vent cleaning moves end-of-fill
pressure; the ring replacement moves cushion and pack-integral spread; sensor recalibration moves all cavity values slightly. Most maintenance falls at a mold change, so those rows
include the template re-establishment's own shift; the ring replacement's cushion row is the cleanest single signature.</p>
{table(esf)}

<h2 id="vm-tracking">Virtual metrology against each audit</h2>
<p>Each audit piece is scored against the prediction made for its shot by the model in force that month. The investigation rule: when the rolling
error over the last {WINDOW} pieces of a mold and press stays above {RETRAIN['weight']}x gauge on weight or {RETRAIN['dimension']}x on the dimension for
{2 * WINDOW} consecutive pieces, the mold and press is investigated before the next monthly retrain, using only matured labels. The dimension limit
sits higher than the weight limit because the dimension error normally runs two to three times gauge, and higher in the warm months (shaded), when mold
temperature moves the part. Over the evaluation period the rule {trig_text}.</p>
{track_png}
"""
    toc = [("maturity", "Maturity"), ("alarm-drift", "Alarm drift"), ("shifts", "Event shifts"), ("vm-tracking", "Measurement tracking")]
    OUT.write_text(shell("Model Monitoring", "Molding quality · MLOps", "Evaluation May 2025 to March 2026, models retrained monthly", body, toc), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
