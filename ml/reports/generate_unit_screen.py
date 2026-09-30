"""
The monitoring unit's screen and the shift chart for one shift on the cell.

Screen: the cycle graph of the latest retained shot against the template's reference
curve, the summary values against their bands, the alarm and sort state, the
predicted weight and critical dimension, and the anomaly state. Shift chart: the
pack integral on an individuals chart with its EWMA, the alarm and sort log, and the
reject-bin review sheet.

Usage: python ml/reports/generate_unit_screen.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.reports.style import fig, img  # noqa: E402
from ml.src import vm  # noqa: E402
from ml.src.features import DATA_DIR, VALID_END, connect  # noqa: E402

OUT = Path(__file__).resolve().parent / "unit_screen.html"
SHIFT_H = {"A": 6, "B": 14, "C": 22}


def pick_shift(con):
    """A test-period shift on one job with sorts, a drift signal and a technician change: a working shift."""
    return con.execute(f"""
        with s as (
            select job_id, press_id, mold_id, date_trunc('day', shot_ts - interval 6 hour) as shift_date,
                   case when hour(shot_ts) between 6 and 13 then 'A' when hour(shot_ts) between 14 and 21 then 'B' else 'C' end as shift,
                   count(*) as shots, count(*) filter (where unit_sorted) as sorts,
                   count(*) filter (where drift_any_signal) as drift_shots
            from fct_shot where shot_ts >= '{VALID_END.date()}' and after_approval and shots_since_approval > 1500
            group by all)
        select * from s where shots > 600 and sorts between 3 and 25 and drift_shots > 0
        order by sorts desc, drift_shots desc limit 1
    """).df().iloc[0]


def main():
    con = connect()
    sh = pick_shift(con)
    t0 = pd.Timestamp(sh["shift_date"]) + pd.Timedelta(hours=SHIFT_H[sh["shift"]])
    t1 = t0 + pd.Timedelta(hours=8)
    shots = con.execute(f"""
        select * from fct_shot where job_id = '{sh['job_id']}' and shot_ts >= '{t0}' and shot_ts < '{t1}' order by shot_ts
    """).df()
    job = con.execute(f"select * from fct_job where job_id = '{sh['job_id']}'").df().iloc[0]
    sensors = con.execute(f"""
        select n.*, t.* exclude (mold_id, press_id, sensor_id)
        from int_sensor_shot_normalized n join stg_monitoring__templates t using (template_id, sensor_id)
        where n.shot_id = (select max(shot_id) from fct_shot where job_id = '{sh['job_id']}' and shot_ts < '{t1}' and curve_retained)
    """).df()
    last_id = int(sensors["shot_id"].iloc[0])
    ref_id = int(sensors["template_curve_ref"].iloc[0])
    curves = con.execute(f"""
        select shot_id, sensor_id, t_ms, pressure_bar from stg_monitoring__cavity_curves
        where mold_id = '{sh['mold_id']}' and shot_id in ({last_id}, {ref_id}) order by t_ms
    """).df()
    chart = con.execute(f"""
        select p.shot_id, p.shot_ts, p.value, p.center, p.sigma, p.we1, p.we2, p.we4, p.we5
        from spc_chart_points p
        where p.job_id = '{sh['job_id']}' and p.metric = 'pack_integral' and p.shot_ts >= '{t0}' and p.shot_ts < '{t1}'
          and p.sensor_id = (select min(sensor_id) from stg_monitoring__shot_summary
                             where mold_id = '{sh['mold_id']}' and sensor_position = 'post_gate')
        order by p.shot_ts
    """).df()
    alarms = con.execute(f"""
        select shot_ts, 'rule ' || rule || ' on ' || metric as event, sensor_id as source from spc_alarms
        where job_id = '{sh['job_id']}' and shot_ts >= '{t0}' and shot_ts < '{t1}' and rule in ('WE1', 'WE2')
        union all
        select shot_ts, 'drift: ' || detector, '' from drift_signals where job_id = '{sh['job_id']}' and shot_ts >= '{t0}' and shot_ts < '{t1}'
        union all
        select change_ts, 'setpoint change: ' || parameter || ' ' || coalesce(cast(old_value as varchar), '') || ' to ' || cast(new_value as varchar),
               technician_id from stg_qms__setpoint_changes where job_id = '{sh['job_id']}' and change_ts >= '{t0}' and change_ts < '{t1}'
        order by 1
    """).df()
    review = con.execute(f"""
        select s.shot_ts, r.* from stg_qms__sort_dispositions r join fct_shot s using (shot_id)
        where s.job_id = '{sh['job_id']}' and s.shot_ts >= '{t0}' and s.shot_ts < '{t1}' order by s.shot_ts
    """).df()
    con.close()

    # predicted weight and critical dimension on every shot of the shift
    pw = vm.predict_shots(shots, "weight").groupby("shot_id")["y"].mean()
    pdim = vm.predict_shots(shots, "dimension").groupby("shot_id")["y"].mean()
    an = pd.read_parquet(DATA_DIR / "results" / "anomaly_if_scores.parquet")
    an = an[an["seed"] == an["seed"].min()].set_index("shot_id")["if_flag"]
    shots["anomaly"] = shots["shot_id"].map(an).fillna(False)

    parts = pd.read_csv(Path(__file__).resolve().parents[2] / "data_source" / "raw" / "erp" / "part_attributes.csv").set_index("mold_id")
    pa = parts.loc[sh["mold_id"]]
    last = shots[shots["shot_id"] == last_id].iloc[0]
    w_pred = (1 + pw.get(last_id, np.nan)) * pa["nominal_weight_g"]
    d_pred = pa["critical_dimension_mm"] + pdim.get(last_id, np.nan) * pa["dimension_tolerance_mm"]

    # ── cycle graph ──
    f, ax = fig(3.6, 7.6)
    f.patch.set_facecolor("#10161d")
    ax.set_facecolor("#10161d")
    for sp in ax.spines.values():
        sp.set_color("#3a4654")
    ax.tick_params(colors="#9fb0c2")
    ax.yaxis.grid(True, color="#1f2a36")
    colors = {"post_gate": "#4fc3f7", "end_of_fill": "#ffb74d"}
    pos = dict(zip(sensors["sensor_id"], sensors["sensor_position"]))
    for sid, g in curves.groupby("sensor_id"):
        r = g[g["shot_id"] == ref_id]
        c = g[g["shot_id"] == last_id]
        ax.plot(r["t_ms"] / 1000, r["pressure_bar"], color="#56657a", lw=3.5, alpha=0.7)
        ax.plot(c["t_ms"] / 1000, c["pressure_bar"], color=colors.get(pos.get(sid), "#ccc"), lw=1.3,
                label=f"{sid} ({pos.get(sid, '').replace('_', ' ')})")
    ax.set_xlabel("time in cycle (s)", color="#9fb0c2")
    ax.set_ylabel("cavity pressure (bar)", color="#9fb0c2")
    ax.set_xlim(0, float(curves["t_ms"].max()) / 1000 * 0.55)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#c9d6e3")
    cycle_png = img(f, "cycle graph")

    # ── shift chart: individuals and EWMA on the post-gate pack integral ──
    f, ax = fig(2.8, 11.5)
    x = np.arange(len(chart))
    c0, s0 = chart["center"].iloc[0], chart["sigma"].iloc[0]
    ax.plot(x, chart["value"], color="#6B8FA8", lw=0.6)
    ew = chart["value"].ewm(alpha=0.2, adjust=False).mean()
    ax.plot(x, ew, color="#3D5166", lw=1.6, label="EWMA (lambda 0.2)")
    for k, ls in [(3, "--"), (-3, "--")]:
        ax.axhline(c0 + k * s0, color="#CC0000", lw=0.9, ls=ls)
    ax.axhline(c0, color="#555", lw=0.8)
    f1 = chart["we1"] | chart["we2"]
    ax.scatter(x[f1], chart.loc[f1, "value"], color="#CC0000", s=14, zorder=3, label="rule 1 or 2")
    ax.set_xlabel("shot in shift")
    ax.set_ylabel("pack integral (bar s)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    shift_png = img(f, "shift chart")

    # ── summary value rows ──
    rows = ""
    for _, r in sensors.iterrows():
        for key, label in [("fill_integral", "Fill integral"), ("pack_integral", "Pack integral"), ("peak_pressure_bar", "Peak pressure"),
                           ("gate_seal_time_s", "Gate seal time"), ("end_of_fill_pressure_bar", "End-of-fill pressure")]:
            tv = r.get(f"{key}_template_value")
            if tv is None or tv != tv:
                continue
            v, lo, hi, wl, wh = r[key], r[f"{key}_alarm_low"], r[f"{key}_alarm_high"], r[f"{key}_warning_low"], r[f"{key}_warning_high"]
            state = "alarm" if (v < lo or v > hi) else ("warn" if (v < wl or v > wh) else "ok")
            pos_ = (v - lo) / (hi - lo) * 100
            rows += (f'<tr><td>{r["sensor_id"]}</td><td>{label}</td><td class="n">{v:,.2f}</td><td class="n">{tv:,.2f}</td>'
                     f'<td class="n">{lo:,.2f} to {hi:,.2f}</td><td><div class="bar"><div class="wz" style="left:{(wl - lo) / (hi - lo) * 100:.0f}%;'
                     f'width:{(wh - wl) / (hi - lo) * 100:.0f}%"></div><div class="mk {state}" style="left:{min(max(pos_, 0), 100):.0f}%"></div></div></td>'
                     f'<td><span class="st {state}">{state.upper() if state != "ok" else "OK"}</span></td></tr>')

    n_sort = int(shots["unit_sorted"].sum())
    n_alarm = int((shots["unit_alarm_state"] == "alarm").sum())
    state = last["unit_alarm_state"]
    log_rows = "".join(f'<tr><td>{pd.Timestamp(a.shot_ts):%H:%M:%S}</td><td>{a.event}</td><td>{a.source}</td></tr>' for a in alarms.itertuples())
    sorted_rows = shots[shots["unit_sorted"]]
    log_rows += "".join(f'<tr><td>{pd.Timestamp(s.shot_ts):%H:%M:%S}</td><td>sorted: {s.alarm_values or "match score"}</td><td>cycle {s.cycle_no}</td></tr>'
                        for s in sorted_rows.itertuples())
    rev_rows = "".join(f'<tr><td>{pd.Timestamp(r.shot_ts):%H:%M}</td><td>{r.shot_id}</td><td class="n">{r.pieces_reviewed}</td>'
                       f'<td class="n">{r.pieces_confirmed_defective}</td><td>{r.defect_codes if r.defect_codes == r.defect_codes else ""}</td>'
                       f'<td class="n">{r.pieces_good}</td><td>{r.inspector_id}</td></tr>' for r in review.itertuples())
    w_tol = pa["nominal_weight_g"] * 0.01
    d_ok = abs(d_pred - pa["critical_dimension_mm"]) <= 0.8 * pa["dimension_tolerance_mm"]

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unit Screen</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#0b1015;color:#dde6ee;font-family:"Segoe UI",Tahoma,sans-serif;font-size:13px}}
.top{{display:flex;justify-content:space-between;align-items:center;background:#16202b;border-bottom:2px solid #263444;padding:8px 16px}}
.top .id{{font-weight:700;font-size:15px;letter-spacing:.4px}} .top .id span{{font-weight:400;color:#8fa3b8;margin-left:10px;font-size:12px}}
.lamp{{display:inline-block;padding:4px 12px;border-radius:3px;font-weight:700;letter-spacing:.6px}}
.lamp.none{{background:#1d6b3a}} .lamp.warning{{background:#9a6a12}} .lamp.alarm{{background:#a11b1b}}
.grid{{display:grid;grid-template-columns:1.25fr 1fr;gap:12px;padding:12px 16px}}
.panel{{background:#121a23;border:1px solid #243140;padding:10px 12px}} .panel h3{{margin:0 0 8px;font-size:12px;color:#8fa3b8;text-transform:uppercase;letter-spacing:1px}}
table{{border-collapse:collapse;width:100%}} td,th{{padding:4px 6px;border-bottom:1px solid #1f2b38;text-align:left}} th{{color:#8fa3b8;font-weight:600;font-size:11.5px}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} td:first-child{{white-space:nowrap}}
.bar{{position:relative;height:10px;background:#3a1515;border-radius:2px;min-width:110px}} .wz{{position:absolute;top:0;bottom:0;background:#1d4d2e}}
.mk{{position:absolute;top:-3px;width:3px;height:16px;background:#fff}} .mk.warn{{background:#ffb300}} .mk.alarm{{background:#ff3b3b}}
.st{{font-weight:700;font-size:11px}} .st.ok{{color:#5fd38a}} .st.warn{{color:#ffb300}} .st.alarm{{color:#ff5c5c}}
.counters{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .ctr{{background:#0f161e;border:1px solid #243140;padding:8px}}
.ctr .v{{font-size:22px;font-weight:700}} .ctr .l{{font-size:11px;color:#8fa3b8}}
.light{{background:#F4F5F7;color:#222;padding:16px}} .light .panel{{background:white;border-color:#ddd;color:#222}} .light h3{{color:#3D5166}}
.light td,.light th{{border-color:#eee}} .light th{{color:#555}} img{{max-width:100%;display:block}}
.foot{{color:#8fa3b8;font-size:11px;padding:6px 16px 14px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="top"><div class="id">{last['press_id']} · CPU-{last['press_id'][-2:]}<span>Mold {sh['mold_id']} · Job {sh['job_id']} · Shift {sh['shift']} {pd.Timestamp(sh['shift_date']):%Y-%m-%d} · Cycle {last['cycle_no']:,}</span></div>
<div><span class="lamp {state}">{state.upper() if state != 'none' else 'IN TEMPLATE'}</span></div></div>
<div class="grid">
 <div class="panel"><h3>Cycle graph · shot {last_id} against template reference (grey)</h3>{cycle_png}</div>
 <div class="panel"><h3>Shift counters</h3><div class="counters">
   <div class="ctr"><div class="v">{len(shots):,}</div><div class="l">shots this shift</div></div>
   <div class="ctr"><div class="v">{n_alarm}</div><div class="l">alarm shots</div></div>
   <div class="ctr"><div class="v">{n_sort}</div><div class="l">sorted to reject bin</div></div>
   <div class="ctr"><div class="v">{w_pred:,.3f} g</div><div class="l">predicted weight (nominal {pa['nominal_weight_g']:.2f})</div></div>
   <div class="ctr"><div class="v" style="color:{'#5fd38a' if d_ok else '#ffb300'}">{d_pred:,.3f}</div><div class="l">predicted critical dimension (±{pa['dimension_tolerance_mm']:.2f})</div></div>
   <div class="ctr"><div class="v" style="color:{'#ffb300' if last['anomaly'] else '#5fd38a'}">{'FLAG' if last['anomaly'] else 'NORMAL'}</div><div class="l">anomaly state · {int(shots['anomaly'].sum())} flags this shift</div></div>
 </div>
 <h3 style="margin-top:14px">Summary values against template bands</h3>
 <table><thead><tr><th>Sensor</th><th>Value</th><th>Shot</th><th>Template</th><th>Alarm band</th><th>Position</th><th></th></tr></thead><tbody>{rows}</tbody></table>
 </div>
</div>
<div class="foot">Warning band shaded green inside the alarm band; the marker is this shot. Predicted values are the virtual metrology model's mean over the active cavities.</div>
<div class="light">
 <div class="panel"><h3>Shift chart · post-gate pack integral, individuals with EWMA (limits from the first 500 shots after approval)</h3>{shift_png}</div>
 <div class="grid" style="padding:12px 0 0">
  <div class="panel"><h3>Alarm and sort log</h3><table><thead><tr><th>Time</th><th>Event</th><th>Source</th></tr></thead><tbody>{log_rows}</tbody></table></div>
  <div class="panel"><h3>Reject-bin review sheet</h3><table><thead><tr><th>Sorted</th><th>Shot</th><th>Reviewed</th><th>Defective</th><th>Codes</th><th>Good</th><th>Inspector</th></tr></thead><tbody>{rev_rows}</tbody></table></div>
 </div>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT, "shift", sh["shift_date"], sh["shift"], sh["job_id"])


if __name__ == "__main__":
    main()
