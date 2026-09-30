"""
Instrumented cell: presses IM-11 and IM-12 with six instrumented molds.

Builds the records the cell's systems hold:

  schedule        runs of the six molds on the two presses, with the jobs,
                  mold sets, first-shot approvals and templates
  material        resin lots, hopper loads, dryer log
  equipment       mold maintenance, mold temperature controllers, check rings
  changes         setpoint changes by the technicians
  shots           per-shot cavity pressure curves, the summary values the
                  monitoring unit computes from them, its alarm and sort
                  decisions, and the press's own per-shot record
  quality         every piece's weight, critical dimension and defects

Each shot's curve is constructed from the shot's effective state: melt
viscosity (resin lot, melt temperature, moisture, regrind), check-ring
leakage, vent restriction, gate and hot-runner tip condition, mold
temperatures, active cavities and the setpoints in force, plus the sensor's
own offset, gain and noise. Summary values are computed from the curve the way
the unit computes them, and each piece's quality follows from its shot's
values. The mechanisms that move the state over time:

  G1 resin lot viscosity       G6 mold temperature drift and summer load
  G2 vent restriction          G7 hot-runner tip wear
  G3 setup convergence         G8 technician adjustments
  G4 check-ring leakage        G9 cavity imbalance and blocking
  G5 moisture                  G10 novel signatures (ring failing, heater zone,
  G0 no curve signal               nozzle drool, wrong material)
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import (CELL as C, CELL_PRESSES, CERT_MFI_ERR_FRAC, DEFECTS, END_DATE, LABELS, LOT_MFI_SD_FRAC, MOLDS,
                      RANDOM_SEED, RESINS,
                      START_DATE)

HZ = 500
SUMMARY_KEYS = ["fill_integral", "pack_integral", "peak_pressure_bar", "end_of_fill_pressure_bar", "gate_seal_time_s"]
# values the unit monitors against the template, by sensor position
MONITORED = {"post_gate": ["fill_integral", "pack_integral", "peak_pressure_bar", "gate_seal_time_s"],
             "end_of_fill": ["end_of_fill_pressure_bar", "pack_integral"]}
CODES = ["short_shot", "flash", "sink", "void", "splay", "burn", "weld_line", "warp", "dimensional",
         "black_specks", "contamination", "gate_vestige", "other"]
MECHS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10"]
TECHS = [("TC-01", "2016-03-01"), ("TC-02", "2018-09-10"), ("TC-03", "2021-01-18"),
         ("TC-04", "2023-05-08"), ("TC-05", "2024-11-04"), ("TC-06", "2025-02-10")]
TECH_SHIFT = {"A": ["TC-01", "TC-04"], "B": ["TC-02", "TC-05"], "C": ["TC-03", "TC-06"]}


def shift_of(ts: pd.Timestamp) -> str:
    h = ts.hour
    return "A" if 6 <= h < 14 else ("B" if 14 <= h < 22 else "C")


def in_production(ts: pd.Timestamp) -> bool:
    """Monday 06:00 to Saturday 06:00."""
    d = ts.weekday()
    if d < 5:
        return not (d == 0 and ts.hour < 6)
    return d == 5 and ts.hour < 6


def next_production(ts: pd.Timestamp) -> pd.Timestamp:
    while not in_production(ts):
        ts = (ts + pd.Timedelta(hours=1)).floor("h")
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# Schedule, material, equipment
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule(rng):
    mold_ids = list(MOLDS)
    weights = np.array([MOLDS[m]["runs_per_year"] for m in mold_ids], float)
    runs = []
    for press in CELL_PRESSES:
        t = pd.Timestamp(START_DATE) + pd.Timedelta(hours=6 + float(rng.uniform(0, 30)))
        last = None
        while True:
            t = next_production(t + pd.Timedelta(days=float(rng.uniform(0.4, 3.6))))
            if t >= pd.Timestamp(END_DATE) - pd.Timedelta(days=2):
                break
            w = weights.copy()
            if last is not None:
                w[mold_ids.index(last)] *= 0.25
            mold = mold_ids[int(rng.choice(len(mold_ids), p=w / w.sum()))]
            days = float(rng.uniform(*C["run_days"]))
            runs.append(dict(press_id=press, mold_id=mold, set_ts=t, prod_days=days))
            # a run occupies prod_days of production time
            end = t
            left = days * 24.0
            while left > 0:
                end = next_production(end)
                step = min(left, 1.0)
                end = end + pd.Timedelta(hours=step)
                left -= step
            t = end
            last = mold
    runs = sorted(runs, key=lambda r: r["set_ts"])
    for i, r in enumerate(runs):
        r["job_id"] = f"J-{250100 + i}"
        r["run_no"] = i
    return runs


def build_lots(rng):
    rows = []
    n = 5000
    for resin, spec in RESINS.items():
        d = pd.Timestamp(START_DATE) - pd.Timedelta(days=60)
        while d < pd.Timestamp(END_DATE):
            sup = str(rng.choice(spec["suppliers"]))
            z = rng.normal(0, 1)
            if sup == "RS-3":
                z = z - 1.3          # this supplier's lots sit low in the flow band
            true_mfi = spec["mfi"] * (1 + LOT_MFI_SD_FRAC * z)
            lo, hi = spec["band"]
            true_mfi = float(np.clip(true_mfi, lo * 0.97, hi * 1.03))
            cert = true_mfi * (1 + rng.normal(0, CERT_MFI_ERR_FRAC))
            rows.append(dict(lot_id=f"RL-{n}", resin_id=resin, grade=resin, supplier_id=sup,
                             received_date=d.normalize(), true_mfi=true_mfi,
                             cert_melt_flow_index=round(float(np.clip(cert, lo, hi)), 1),
                             cert_moisture_pct=round(float(rng.uniform(0.05, 0.18) if spec["hygroscopic"] else rng.uniform(0.01, 0.04)), 3),
                             cert_density=round(float(rng.normal(1.36 if "PA66" in resin else 1.15 if "PC" in resin else 1.41 if resin == "POM" else 0.90, 0.004)), 3),
                             qty_kg=int(rng.integers(900, 2200))))
            n += 1
            d += pd.Timedelta(days=float(rng.uniform(12, 26)))
    return pd.DataFrame(rows)


def build_equipment(rng):
    """Mold temperature controllers (one drifting between services) and check rings."""
    tcu = {}
    for press in CELL_PRESSES:
        services, d = [], pd.Timestamp(START_DATE) - pd.Timedelta(days=30)
        while d < pd.Timestamp(END_DATE) + pd.Timedelta(days=30):
            services.append(d)
            d += pd.Timedelta(days=float(rng.uniform(*C["tcu_service_days"])))
        tcu[press] = np.array(services, dtype="datetime64[ns]")
    ring_change = {"IM-12": [pd.Timestamp("2024-09-02"), pd.Timestamp(C["ring_change_date"])],
                   "IM-11": [pd.Timestamp("2024-06-10")]}
    return tcu, ring_change


def season_c(ts_days: np.ndarray) -> np.ndarray:
    """Summer chiller load on mold temperature, degrees above setpoint."""
    doy = ((ts_days - np.datetime64("2025-01-01")) / np.timedelta64(1, "D")) % 365.25
    return C["summer_peak_c"] * np.clip(np.cos((doy - 205) / 365.25 * 2 * np.pi), 0, None) ** 3


# ══════════════════════════════════════════════════════════════════════════════
# Curve model
# ══════════════════════════════════════════════════════════════════════════════

def curve_params(m, sens_pos, st):
    """Curve parameters per shot for one sensor, from the shot's effective state.
    st holds per-shot arrays; returns dict of per-shot arrays."""
    lv, leak, vent = st["lv"], st["leak"], st["vent"]
    eof = sens_pos == "end_of_fill"
    t_tr = m["fill_s"] * np.exp(-st["dlog_vel"]) * (1 + st["sw_delay"])
    a = 0.28 * (m["flow_ratio"] if eof else 1.0)
    t_arr = t_tr * np.clip(a * np.exp(0.12 * lv + 1.5 * leak), 0.05, 0.92)
    fill_level = st["fill_level"]
    p_tr = m["transfer_bar"] * np.exp(0.8 * lv - 2.5 * leak + 0.25 * st["dlog_vel"] + fill_level)
    if eof:
        p_tr = p_tr * m["eof_ratio"] * np.exp(-C["vent_eof_coef"] * vent)
    k = 1.8 * np.exp(0.3 * lv)
    spike = 0.04 + np.clip(-st["sw_delay"] * 1.5, 0, 0.15)
    p_pack = m["pack_bar"] * np.exp(st["pack_level"])
    if eof:
        p_pack = p_pack * 0.72 * np.exp(-0.4 * vent)
    tau_p = 0.15 * np.exp(0.5 * lv)
    t_gs = t_tr + m["seal_s"] * np.exp(0.02 * st["dT"] + st["gs_level"])
    tau_d = m["decay_s"] * np.exp(0.02 * st["dT"] + 0.04 * np.abs(st["dT_ab"]))
    return dict(t_arr=t_arr, t_tr=t_tr, p_tr=p_tr, k=k, spike=spike, p_pack=p_pack, tau_p=tau_p,
                t_gs=t_gs, tau_d=tau_d)


def construct(par, n_samples, drool=None):
    """Curves at 500 Hz for a chunk of shots: array (n_shots, n_samples)."""
    t = (np.arange(n_samples, dtype=np.float32) / HZ)[None, :]
    g = {k: v[:, None].astype(np.float32) for k, v in par.items()}
    fill = g["p_tr"] * np.clip((t - g["t_arr"]) / (g["t_tr"] - g["t_arr"]), 0, 1) ** g["k"]
    p_peak = g["p_tr"] * (1 + g["spike"])
    since_tr = t - g["t_tr"]
    spike_ramp = np.where(since_tr < 0.02, g["p_tr"] + (p_peak - g["p_tr"]) * since_tr / 0.02, p_peak)
    pack = g["p_pack"] + (p_peak - g["p_pack"]) * np.exp(-np.clip(since_tr - 0.02, 0, None) / g["tau_p"])
    pack = np.where(since_tr < 0.02, spike_ramp, pack)
    p_gs = g["p_pack"] + (p_peak - g["p_pack"]) * np.exp(-np.clip(g["t_gs"] - g["t_tr"] - 0.02, 0, None) / g["tau_p"])
    decay = p_gs * np.exp(-np.clip(t - g["t_gs"], 0, None) / g["tau_d"])
    P = np.where(t < g["t_arr"], 0.0, np.where(t < g["t_tr"], fill, np.where(t < g["t_gs"], pack, decay)))
    if drool is not None:
        P = P + drool[:, None] * np.abs(np.sin(t * 37.0)) * (t < g["t_tr"]) * 40.0
    return P.astype(np.float32)


def summarize(P, par, eof, tail_tau):
    """Summary values from the curve, as the monitoring unit computes them."""
    dt = 1.0 / HZ
    n, T = P.shape
    t = np.arange(T, dtype=np.float32) / HZ
    above = P > 2.0
    has = above.any(axis=1)
    arr_idx = np.where(has, above.argmax(axis=1), 0)
    tr_idx = np.clip((par["t_tr"] * HZ).astype(int), 0, T - 1)
    peak_idx = P.argmax(axis=1)
    peak = P[np.arange(n), peak_idx]
    p_tr = P[np.arange(n), tr_idx]
    cum = np.cumsum(P, axis=1) * dt
    fill_int = cum[np.arange(n), tr_idx] - cum[np.arange(n), arr_idx]
    # gate seal: first sample after the plateau settles where pressure falls below 95% of it
    settle = np.clip(tr_idx + (4 * par["tau_p"] * HZ).astype(int), 0, T - 1)
    plateau = P[np.arange(n), settle]
    after = (np.arange(T)[None, :] > settle[:, None]) & (P < 0.95 * plateau[:, None])
    gs_idx = np.where(after.any(axis=1), after.argmax(axis=1), T - 1)
    pack_int = cum[np.arange(n), gs_idx] - cum[np.arange(n), tr_idx]
    tail = P[:, -1] * tail_tau               # exponential tail beyond the window
    cycle_int = cum[:, -1] + tail
    # cooling rate: slope of ln(P) over the first decay time constant after gate seal
    i1 = np.clip(gs_idx + (tail_tau * HZ).astype(int), 0, T - 1)
    p1 = np.maximum(P[np.arange(n), i1], 1e-3)
    p0 = np.maximum(P[np.arange(n), gs_idx], 1e-3)
    cooling = (np.log(p0) - np.log(p1)) / np.maximum((i1 - gs_idx) * dt, dt)
    out = dict(fill_time_to_sensor_s=arr_idx * dt, peak_pressure_bar=peak, time_to_peak_s=peak_idx * dt,
               pressure_at_transfer_bar=p_tr, fill_integral=fill_int, pack_integral=pack_int,
               cycle_integral=cycle_int, cooling_rate=cooling)
    if eof:
        out["end_of_fill_pressure_bar"] = p_tr
        out["gate_seal_time_s"] = np.full(n, np.nan)
    else:
        out["end_of_fill_pressure_bar"] = np.full(n, np.nan)
        out["gate_seal_time_s"] = gs_idx * dt
    return out


def ar1(rng, n, rho, sd):
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════════

def build_cell(sink, rng_seed_offset=0, max_runs=None, run_stride=1):
    """sink(kind, df, mold_id, ts) receives the large per-shot tables in parts, written as
    Parquet by the caller: 'curves', 'summary', 'machine' and 'state'."""
    rng = np.random.default_rng(RANDOM_SEED + rng_seed_offset)
    runs = build_schedule(rng)
    lots = build_lots(rng)
    tcu_services, ring_changes = build_equipment(rng)
    for press, dates in ring_changes.items():
        moved = []
        for c in dates:
            nxt = [r["set_ts"] for r in runs if r["press_id"] == press and r["set_ts"] >= c]
            moved.append(min(nxt) - pd.Timedelta(hours=1) if nxt and c >= pd.Timestamp(START_DATE) else c)
        ring_changes[press] = moved
    tech_start = {t: pd.Timestamp(d) for t, d in TECHS}

    # Per-mold counters carried across runs
    vent_since = {m: int(rng.integers(0, C["vent_clean_shots"][m])) for m in MOLDS}
    tip_since = {m: int(rng.integers(0, C["tip_change_shots"])) for m in MOLDS}
    mold_shots = {m: int(rng.integers(200000, 900000)) for m in MOLDS}
    cycle_counter = {p: int(rng.integers(1_000_000, 3_000_000)) for p in CELL_PRESSES}
    lot_left = {}                     # resin -> (lot_id, kg left)
    lot_queue = {r: lots[lots["resin_id"] == r].sort_values("received_date").reset_index(drop=True) for r in RESINS}
    lot_ptr = {r: 0 for r in RESINS}
    tech_mold_first = {}
    cavity_offsets = {m: dict(weight=rng.normal(0, 0.0025, MOLDS[m]["cavities"]),
                              dim=rng.normal(0, 0.20, MOLDS[m]["cavities"]),
                              dev=rng.normal(0, 0.15, MOLDS[m]["cavities"])) for m in MOLDS}
    sensor_gain = {}
    sensor_offset = {}
    for m, mm in MOLDS.items():
        for k, (pos, cav) in enumerate(mm["sensors"]):
            sid = f"{m}-S{k + 1}"
            sensor_gain[sid] = 1 + rng.choice([-1, 1]) * rng.uniform(*C["sensor_gain_err"])
            sensor_offset[sid] = rng.uniform(*C["sensor_offset_bar"])

    # G10 events: time, press, type, duration
    g10 = []
    n_ev = int(round(C["g10_events_per_year"] * (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365))
    ev_runs = rng.choice(len(runs), size=min(n_ev, len(runs)), replace=False)
    for ri in ev_runs:
        g10.append(dict(run=int(ri), kind=str(rng.choice(["ring_failing", "heater_zone", "nozzle_drool", "wrong_material"])),
                        start_frac=float(rng.uniform(0.2, 0.7)), hours=float(rng.uniform(2.5, 7.0))))
    g10_by_run = {e["run"]: e for e in g10}
    if run_stride > 1:
        runs = runs[::run_stride]
    if max_runs:
        runs = runs[:max_runs]

    out = dict(shots=[], summaries=[], machine=[], defects=[], templates=[], approvals=[], loads=[],
               changes=[], maint=[], audits_truth=[], audit_plan=[], state=[])
    load_counter, change_counter, maint_counter, template_counter = 1, 1, 1, 1
    shot_counter = 1
    last_job_on_press = {}

    for r in runs:
        m_id, press, job = r["mold_id"], r["press_id"], r["job_id"]
        m = MOLDS[m_id]
        resin = m["resin"]
        R = RESINS[resin]
        cav = m["cavities"]

        # ── Maintenance between runs: vent cleaning on the shot count, tip changes, PM ──
        ts0 = r["set_ts"]
        if vent_since[m_id] >= C["vent_clean_shots"][m_id] * float(rng.uniform(0.75, 0.95)):
            out["maint"].append(dict(mold_id=m_id, event_ts=ts0 - pd.Timedelta(hours=2), event_type="vent_cleaning",
                                     shots_at_event=mold_shots[m_id], technician_id=str(rng.choice([t for t, _ in TECHS])),
                                     notes="Vents cleaned at the scheduled shot count"))
            vent_since[m_id] = 0
        if m_id in ("M-2041", "M-2043") and tip_since[m_id] >= C["tip_change_shots"]:
            out["maint"].append(dict(mold_id=m_id, event_ts=ts0 - pd.Timedelta(hours=3), event_type="hot_runner_tip",
                                     shots_at_event=mold_shots[m_id], technician_id="TC-01", notes="Hot-runner tips replaced"))
            tip_since[m_id] = 0
        if rng.random() < 0.3:
            out["maint"].append(dict(mold_id=m_id, event_ts=ts0 - pd.Timedelta(hours=4), event_type="PM",
                                     shots_at_event=mold_shots[m_id], technician_id=str(rng.choice([t for t, _ in TECHS])),
                                     notes="Preventive maintenance before run"))

        # ── Shot timeline ──
        tech = str(rng.choice(TECH_SHIFT[shift_of(ts0)]))
        # technicians hired well before the window have run every mold; newer ones learn it on the job
        first_on_mold = tech_mold_first.setdefault(
            (tech, m_id), tech_start[tech] if tech_start[tech] < pd.Timestamp(START_DATE) - pd.Timedelta(days=180) else ts0)
        tenure = max(0.0, (ts0 - max(tech_start[tech], first_on_mold)).days)
        n_setup = int(rng.integers(*C["approval_shots"]))
        cyc = m["cycle_s"]
        times = []
        t = ts0
        total_s = r["prod_days"] * 86400 * C["uptime"]
        elapsed = 0.0
        while elapsed < total_s:
            if not in_production(t):
                t = next_production(t)
            times.append(t)
            t = t + pd.Timedelta(seconds=cyc * float(1 + rng.normal(0, 0.004)))
            elapsed += cyc
            if rng.random() < cyc / 3600 * 0.35:
                t = t + pd.Timedelta(minutes=float(rng.uniform(*C["stop_minutes"])))
        n = len(times)
        ts = np.array(times, dtype="datetime64[ns]")
        idx = np.arange(n)
        approval_i = n_setup
        approval_ts = pd.Timestamp(ts[approval_i])

        # ── Loads and lots ──
        load_idx = [0]
        hours = (ts - ts[0]) / np.timedelta64(1, "h")
        h_next = float(rng.uniform(*C["load_every_h"]))
        for i in range(1, n):
            if hours[i] >= h_next:
                load_idx.append(i)
                h_next = hours[i] + float(rng.uniform(*C["load_every_h"]))
        load_of = np.searchsorted(np.array(load_idx), idx, side="right") - 1
        kg_per_shot = m["weight_g"] * cav * 1.08 / 1000
        load_lot, load_res, load_regrind, load_rows = [], [], [], []
        dryer = C["dryer_ids"][CELL_PRESSES.index(press)] if R["hygroscopic"] else None
        for li, s0 in enumerate(load_idx):
            s1 = load_idx[li + 1] if li + 1 < len(load_idx) else n
            need = (s1 - s0) * kg_per_shot
            lid, left = lot_left.get(resin, (None, 0.0))
            fresh = False
            if lid is None or left < need * 0.5:
                q = lot_queue[resin]
                while lot_ptr[resin] < len(q) - 1 and q.loc[lot_ptr[resin] + 1, "received_date"] <= pd.Timestamp(ts[s0]):
                    if lid is None or left < need * 0.5:
                        lot_ptr[resin] += 1
                        break
                    break
                row = q.loc[lot_ptr[resin]]
                if lid == row["lot_id"]:
                    lot_ptr[resin] = min(lot_ptr[resin] + 1, len(q) - 1)
                    row = q.loc[lot_ptr[resin]]
                lid, left, fresh = row["lot_id"], float(row["qty_kg"]), True
            lot_left[resin] = (lid, left - need)
            if R["hygroscopic"]:
                short = rng.random() < (0.35 if fresh else 0.08)
                res_h = R["min_residence_h"] * (rng.uniform(0.25, 0.8) if short else rng.uniform(1.1, 2.5))
            else:
                res_h = 0.0
            regrind = float(rng.uniform(*C["regrind_pct"])) if rng.random() < 0.6 else 0.0
            load_lot.append(lid)
            load_res.append(res_h)
            load_regrind.append(regrind)
            lts = pd.Timestamp(ts[s0])
            late = rng.random() < 0.125
            load_rows.append(dict(load_id=f"LD-{load_counter}", press_id=press, job_id=job,
                                  load_ts=lts + (pd.Timedelta(minutes=float(rng.uniform(5, 90))) if late else pd.Timedelta(0)),
                                  resin_lot_id=lid, colorant_lot_id=f"CL-{int(rng.integers(300, 360))}",
                                  regrind_pct=round(regrind, 1), dryer_id=dryer, residence_h=res_h,
                                  operator_id=f"OP{int(rng.integers(1, 40)):03d}", actual_ts=lts))
            load_counter += 1
        out["loads"].extend(load_rows)
        lot_true = lots.set_index("lot_id")["true_mfi"]
        mfi = np.array([lot_true[l] for l in load_lot])[load_of]
        lv_g1 = C["g1_visc_exp"] * np.log(R["mfi"] / mfi)
        # moisture: short residence on a hygroscopic grade, worse through the faulty dew-point sensor period
        res_h = np.array(load_res)[load_of]
        moist = np.zeros(n)
        if R["hygroscopic"]:
            hrs_into_load = hours - hours[np.array(load_idx)[load_of]]
            eff_res = res_h + hrs_into_load * 0.6
            moist = np.clip(1 - eff_res / R["min_residence_h"], 0, None)
            if dryer == C["faulty_dryer"]:
                in_fault = (ts >= np.datetime64(C["faulty_from"])) & (ts <= np.datetime64(C["faulty_to"]))
                moist = moist + in_fault * 0.55
        lv_g5 = C["moisture_visc_coef"] * 0.15 * moist
        regrind = np.array(load_regrind)[load_of]

        # ── Mold temperature (G6) ──
        svc = tcu_services[press]
        last_svc = svc[np.searchsorted(svc, ts, side="right") - 1]
        days_svc = (ts - last_svc) / np.timedelta64(1, "D")
        drift = (C["tcu_drift_c_per_day"] * days_svc) if press == "IM-11" else 0.1 * days_svc * 0.0
        dT = season_c(ts) + 0.5 * drift + ar1(rng, n, 0.98, 0.3)
        dT_ab = drift + ar1(rng, n, 0.98, 0.3)

        # ── Check ring (G4) ──
        rc = [c for c in ring_changes[press] if c <= pd.Timestamp(ts[0])]
        last_ring = max(rc)
        days_ring = (ts - np.datetime64(last_ring)) / np.timedelta64(1, "D")
        leak = (C["ring_leak_rate_per_day"] if press == "IM-12" else C["ring_leak_rate_per_day"] * 0.15) * days_ring

        # ── Vent (G2) and tip (G7) counters ──
        # vent restriction builds slowly, then sharply toward the cleaning count, and levels off past it
        S_v = C["vent_clean_shots"][m_id]
        vent_shots = vent_since[m_id] + idx * 1.0
        u_v = vent_shots / S_v
        vent = C["vent_max"] * np.where(u_v <= 1, u_v ** C["vent_exp"], 1 + 0.6 * np.tanh(1.5 * (u_v - 1)))
        tipw = (tip_since[m_id] + idx) / C["tip_change_shots"] if m_id in ("M-2041", "M-2043") else np.zeros(n)

        # ── Setup convergence (G3) ──
        tau = C["g3_tau_shots"][0] + (C["g3_tau_shots"][1] - C["g3_tau_shots"][0]) * np.exp(-tenure / C["g3_tenure_days"])
        sgn = 1 if rng.random() < 0.5 else -1
        newness = np.exp(-tenure / C["g3_tenure_days"])
        g3 = np.where(idx < approval_i, 2.5, np.exp(-(idx - approval_i) / tau)) * C["g3_offset"] * (1 + 1.2 * newness) * sgn

        # ── Setpoint changes (G8) ──
        dlog_vel = np.zeros(n)
        sw_delay = np.zeros(n)
        hold = np.zeros(n)
        melt = np.zeros(n)
        moldset = np.zeros(n)
        g8_fill = np.zeros(n)
        g8_pack = np.zeros(n)
        change_rows = [dict(change_id=f"SC-{change_counter}", press_id=press, job_id=job, change_ts=approval_ts,
                            parameter="hold_pressure_bar", old_value=None, new_value=None, technician_id=tech,
                            reason_code="first_shot_approval", approved_by="QE-01", is_tamper=False)]
        change_counter += 1
        # Technician actions in time order: scheduled checks (drift corrections, and tampering,
        # changes made to noise) and responses to a resin lot step. Each action sees the process
        # as it stands, including every earlier change.
        r0 = min(n - 60, approval_i + int(3 * tau))
        events = []
        for _ in range(int(rng.integers(*C["changes_per_run"]))):
            ci = int(rng.integers(min(n - 2, r0 + 120), max(min(n - 1, r0 + 121), n - 50)))
            kind = "tamper" if rng.random() < C["tamper_share"] else "check"
            events.append((ci, kind))
            if kind == "tamper":
                if rng.random() < 0.45:
                    # the next check finds the change and takes it back out
                    events.append((min(n - 1, ci + int(rng.uniform(1.0, 3.0) * 3600 / cyc)), "revert", ci))
                else:
                    # the change shows up in the parts and is corrected as a deviation
                    events.append((min(n - 1, ci + int(rng.uniform(3.0, 8.0) * 3600 / cyc)), "check"))
        for li in range(1, len(load_idx)):
            if load_lot[li] != load_lot[li - 1] and load_idx[li] > r0:
                first = min(n - 1, load_idx[li] + int(rng.uniform(20, 90) * 60 / cyc))
                events.append((first, "lot_response", load_idx[li]))
                events.append((min(n - 1, first + int(rng.uniform(60, 150) * 60 / cyc)), "lot_response", first))
        events = [e if len(e) == 3 else (e[0], e[1], None) for e in events]
        events.sort(key=lambda e: e[0])

        def provisional(i, since=None):
            lvx = lv_g1 + lv_g5 - 0.012 * melt
            fl = 0.8 * lvx + g8_fill - 2.5 * leak + 0.94 * g3
            pk = -0.25 * lvx + hold + g3
            rec = slice(max(r0, i - 200, since or 0), i)
            if rec.stop - rec.start < 10 or r0 < 0:
                return None                       # too few shots since the event to judge
            return (float(np.mean(fl[rec]) - np.mean(fl[r0:r0 + 60])),
                    float(np.mean(pk[rec]) - np.mean(pk[r0:r0 + 60])))

        tamper_done = {}
        for ci, kind, since in events:
            tamper = kind == "tamper"
            revert = kind == "revert"
            if revert:
                if since not in tamper_done or ci <= since:
                    continue
                param, d0 = tamper_done.pop(since)
            else:
                prov = provisional(ci, None if tamper else since)
                if prov is None:
                    continue
                prov_fill, prov_pack = prov
            if revert:
                pass
            elif tamper:
                param = str(rng.choice(["hold_pressure_bar", "injection_velocity_mm_s", "switchover_position_mm",
                                        "melt_temp_c", "mold_temp_c"], p=[0.35, 0.2, 0.15, 0.15, 0.15]))
            else:
                if max(abs(prov_fill) / 0.06, abs(prov_pack) / 0.04) < 0.3:
                    continue                      # nothing to correct
                param = "melt_temp_c" if abs(prov_fill) / 0.06 >= abs(prov_pack) / 0.04 else "hold_pressure_bar"
            size = rng.uniform(0.6, 1.4)
            direction = rng.choice([-1, 1])
            if revert:
                # the next check finds the change and takes it back out
                d = -d0
                if param == "hold_pressure_bar":
                    hold[ci:] += d; g8_pack[ci:] += d
                elif param == "injection_velocity_mm_s":
                    dlog_vel[ci:] += d; g8_fill[ci:] += 0.25 * d
                elif param == "switchover_position_mm":
                    sw_delay[ci:] += d; g8_fill[ci:] += -0.3 * d
                elif param == "melt_temp_c":
                    melt[ci:] += d
                else:
                    moldset[ci:] += d
            elif param == "hold_pressure_bar":
                d = (-prov_pack + rng.normal(0, 0.004)) if not tamper else direction * 0.03 * size
                hold[ci:] += d; g8_pack[ci:] += d
            elif param == "injection_velocity_mm_s":
                d = direction * 0.05 * size; dlog_vel[ci:] += d; g8_fill[ci:] += 0.25 * d
            elif param == "switchover_position_mm":
                d = direction * 0.03 * size; sw_delay[ci:] += d; g8_fill[ci:] += -0.3 * d
            elif param == "melt_temp_c":
                d = (float(np.clip(prov_fill / 0.0096, -12, 12)) + rng.normal(0, 0.4)) if not tamper else direction * 3.0 * size
                melt[ci:] += d
            else:
                d = direction * 1.0 * size; moldset[ci:] += d
            if tamper:
                tamper_done[ci] = (param, d)
                reason = "unknown" if rng.random() > 0.68 else str(rng.choice(["defect_response", "drift_correction"]))
            else:
                reason = "unknown" if rng.random() > 0.68 else ("defect_response" if kind == "lot_response" else "drift_correction")
            change_rows.append(dict(change_id=f"SC-{change_counter}", press_id=press, job_id=job,
                                    change_ts=pd.Timestamp(ts[ci]), parameter=param, old_value=0.0, new_value=float(d),
                                    technician_id=str(rng.choice(TECH_SHIFT[shift_of(pd.Timestamp(ts[ci]))])),
                                    reason_code=reason, approved_by=None if rng.random() < 0.10 else "QE-01",
                                    is_tamper=bool(tamper), shot_index=ci))
            change_counter += 1
        out["changes"].extend(change_rows)
        lv_g8 = -0.012 * melt
        dT = dT + moldset

        # ── Cavity blocking (G9) ──
        blocked = np.zeros(n, dtype=int)
        block_cav = None
        bi = n
        if rng.random() < C["block_prob_per_run"] and cav > 2:
            bi = int(rng.integers(approval_i + 500, max(approval_i + 501, n - 100)))
            block_cav = int(rng.integers(1, cav + 1))
            sensed = {c for _, c in m["sensors"]}
            while block_cav in sensed:
                block_cav = int(rng.integers(1, cav + 1))
            blocked[bi:] = 1
            out["maint"].append(dict(mold_id=m_id, event_ts=pd.Timestamp(ts[bi]), event_type="cavity_block",
                                     shots_at_event=mold_shots[m_id] + bi, technician_id=tech,
                                     notes=f"Cavity {block_cav} blocked, damaged core pin"))
            out["maint"].append(dict(mold_id=m_id, event_ts=pd.Timestamp(ts[-1]) + pd.Timedelta(hours=6),
                                     event_type="cavity_unblock", shots_at_event=mold_shots[m_id] + n,
                                     technician_id="TC-01", notes=f"Cavity {block_cav} repaired and unblocked"))
        g9_step = blocked * 0.9 / cav

        # ── Novel signatures (G10) ──
        g10_mask = np.zeros(n, bool)
        g10_kind = None
        g10_fill = np.zeros(n)
        g10_leak = np.zeros(n)
        drool = np.zeros(n)
        if r["run_no"] in g10_by_run:
            e = g10_by_run[r["run_no"]]
            s0 = int(approval_i + e["start_frac"] * (n - approval_i))
            s1 = min(n, s0 + int(e["hours"] * 3600 / cyc))
            g10_mask[s0:s1] = True
            g10_kind = e["kind"]
            ramp = np.clip((idx - s0) / max(1, s1 - s0), 0, 1) * g10_mask
            if g10_kind == "ring_failing":
                g10_leak = 0.06 * ramp + rng.exponential(0.02, n) * g10_mask
            elif g10_kind == "heater_zone":
                g10_fill = 0.10 * ramp
            elif g10_kind == "nozzle_drool":
                drool = g10_mask * rng.uniform(0.2, 1.0, n)
            else:
                g10_fill = -0.08 * g10_mask
        leak_eff = leak + g10_leak

        # ── Effective state ──
        base_noise_fill = ar1(rng, n, float(rng.uniform(*C["ar_integrals"])), C["noise_integral_frac"])
        pack_sd = C["noise_integral_frac"] * (1 + C["ring_var_coef"] * leak_eff)
        base_noise_pack = ar1(rng, n, float(rng.uniform(*C["ar_integrals"])), 1.0) * pack_sd
        lv = lv_g1 + lv_g5 + lv_g8 + g10_fill + 0.3 * g3 + np.log1p(regrind / 100 * -0.15) + base_noise_fill * 0.5
        st = dict(
            lv=lv, leak=leak_eff, vent=vent, dlog_vel=dlog_vel, sw_delay=sw_delay,
            fill_level=0.7 * g3 + base_noise_fill + g9_step * 0.5,
            pack_level=hold + g3 - 0.25 * (lv - base_noise_fill * 0.5) - 0.03 * tipw + base_noise_pack + g9_step,
            gs_level=0.4 * hold + 0.4 * g3 + ar1(rng, n, float(rng.uniform(*C["ar_timings"])), C["noise_timing_frac"]),
            dT=dT, dT_ab=dT_ab,
        )

        bad = [kk for kk, v in st.items() if not np.all(np.isfinite(v))]
        if bad:
            raise ValueError(f"non-finite state {bad} in run {r['run_no']} ({job}, {m_id}, {press})")

        # ── Curves and summaries per sensor ──
        retain = (idx % C["curve_retain_every"] == 0)
        sens_sum = {}
        for k, (pos, sc) in enumerate(m["sensors"]):
            sid = f"{m_id}-S{k + 1}"
            par = curve_params(m, pos, st)
            eof = pos == "end_of_fill"
            win = float(np.max(par["t_gs"]) + 3.2 * np.max(par["tau_d"]))
            T = int(win * HZ) + 1
            res = {key: np.empty(n, np.float32) for key in ["fill_time_to_sensor_s", "peak_pressure_bar", "time_to_peak_s",
                                                           "pressure_at_transfer_bar", "fill_integral", "pack_integral",
                                                           "cycle_integral", "cooling_rate", "end_of_fill_pressure_bar",
                                                           "gate_seal_time_s"]}
            for c0 in range(0, n, 400):
                c1 = min(n, c0 + 400)
                sub = {kk: v[c0:c1] for kk, v in par.items()}
                P = construct(sub, T, drool[c0:c1] if drool.any() else None) * sensor_gain[sid]
                s = summarize(P, sub, eof, par["tau_d"][c0:c1])
                for kk, v in s.items():
                    res[kk][c0:c1] = v
            sens_sum[sid] = (pos, sc, par, res, T)

        # ── Template (re-established at approval from the approved process) ──
        # Template versions: established at the first-shot approval from the converged process,
        # and re-established after a documented cavity block
        seg_starts = [(approval_i, min(n - 60, approval_i + int(3 * tau)))]
        if block_cav is not None:
            seg_starts.append((min(n - 1, bi + 40), min(n - 60, bi + 60)))
        seg_of = np.searchsorted(np.array([a0 for a0, _ in seg_starts]), idx, side="right") - 1
        seg_of = np.clip(seg_of, 0, None)
        tmpl_rows = []
        widened = rng.random() < C["widen_prob"]
        seg_ids = []
        for si, (a0, r0_) in enumerate(seg_starts):
            tid = f"TP-{template_counter}"
            template_counter += 1
            seg_ids.append(tid)
            for sid, (pos, sc, par, res, T) in sens_sum.items():
                for key in MONITORED[pos]:
                    win_vals = res[key][max(0, r0_):max(1, r0_) + 60]
                    if np.all(np.isnan(win_vals)):
                        continue
                    tv_ = float(np.nanmedian(win_vals))
                    band = C["alarm_band"][key] * (C["widen_factor"] if widened and key == "pack_integral" else 1.0)
                    tmpl_rows.append(dict(template_id=tid, segment=si, mold_id=m_id, press_id=press, sensor_id=sid,
                                          summary_value=key, template_value=tv_,
                                          warning_low=tv_ * (1 - band * C["warning_frac"]),
                                          warning_high=tv_ * (1 + band * C["warning_frac"]),
                                          alarm_low=tv_ * (1 - band), alarm_high=tv_ * (1 + band),
                                          match_score_threshold=C["match_score_threshold"],
                                          effective_from=pd.Timestamp(ts[a0]), established_by=tech,
                                          approved_by="QE-01" if not widened else "QE-02",
                                          source_run_job_id=job, widened=widened,
                                          reason="first_shot_approval" if si == 0 else "cavity_block"))
        out["templates"].extend(tmpl_rows)
        template_id = seg_ids[0]
        shot_template = np.array(seg_ids, dtype=object)[seg_of]
        tdf = pd.DataFrame(tmpl_rows)

        # ── Sensor layer and alarm evaluation ──
        shot_ids = np.arange(shot_counter, shot_counter + n)
        shot_counter += n
        cyc_no = cycle_counter[press] + idx
        cycle_counter[press] += n
        dev_band = {}                  # per sensor: deviation of each summary as a fraction of its alarm band
        alarm_any = np.zeros(n, bool)
        warn_any = np.zeros(n, bool)
        spike_any = np.zeros(n, bool)
        for sid, (pos, sc, par, res, T) in sens_sum.items():
            drop = rng.random(n) < C["dropout_rate"]
            spk = rng.random(n) < C["spike_rate"]
            res["peak_pressure_bar"] = np.where(spk, res["peak_pressure_bar"] * rng.uniform(1.3, 1.8, n), res["peak_pressure_bar"])
            spike_any |= spk
            devs = {}
            for key in SUMMARY_KEYS:
                rows_k = tdf[(tdf["sensor_id"] == sid) & (tdf["summary_value"] == key)].set_index("segment")
                if rows_k.empty:
                    continue
                tv, lo, hi = (rows_k["template_value"].to_numpy()[seg_of], rows_k["alarm_low"].to_numpy()[seg_of],
                              rows_k["alarm_high"].to_numpy()[seg_of])
                wlo, whi = rows_k["warning_low"].to_numpy()[seg_of], rows_k["warning_high"].to_numpy()[seg_of]
                v = res[key]
                devs[key] = (v - tv) / (hi - tv)
                a = ((v < lo) | (v > hi)) & ~drop
                w = ((v < wlo) | (v > whi)) & ~drop
                alarm_any |= a
                warn_any |= w
            # match score: similarity of this curve's summary shape to the template
            dvals = np.vstack([np.abs(devs[k]) for k in devs])
            match = np.clip(1 - 0.18 * np.sqrt(np.mean(dvals ** 2, axis=0)) - 0.25 * (drool > 0) - 0.35 * spk, 0, 1)
            if g10_kind in ("nozzle_drool", "ring_failing"):
                match = np.where(g10_mask, np.clip(match - rng.uniform(0.05, 0.25, n), 0, 1), match)
            alarm_any |= (match < C["match_score_threshold"]) & ~drop
            dev_band[sid] = devs
            sens_sum[sid] = (pos, sc, par, res, T, drop, spk, match, devs)

        alarm_state = np.where(alarm_any, "alarm", np.where(warn_any, "warning", "none"))
        sort = alarm_any & (idx >= approval_i)
        # the unit's recorded state disagrees with the recomputation on a handful of shots at template changes
        recorded_alarm = alarm_state.copy()
        flip = rng.random(n) < 0.004
        recorded_alarm = np.where(flip & (alarm_state == "warning"), "none", recorded_alarm)
        job_field = np.array([job] * n, dtype=object)
        prev_job = last_job_on_press.get(press)
        last_job_on_press[press] = job
        if prev_job is not None:
            k_lag = int(n * C["job_field_lag_share"] * rng.uniform(0.6, 1.4))
            job_field[:k_lag] = prev_job
        retain = retain | sort | (alarm_state == "alarm")

        shots = pd.DataFrame(dict(shot_id=shot_ids, press_id=press, unit_id=f"CPU-{press[-2:]}", mold_id=m_id,
                                  job_id=job_field, true_job_id=job, shot_ts=ts, cycle_no=cyc_no,
                                  alarm_state=recorded_alarm, recomputed_alarm_state=alarm_state,
                                  sort_signal=np.where(sort, "reject", "pass"), curve_retained=retain,
                                  in_production=idx >= approval_i, template_id=shot_template,
                                  active_cavities=cav - blocked))
        out["shots"].append(shots)

        # ── Summary rows ──
        for sid, (pos, sc, par, res, T, drop, spk, match, devs) in sens_sum.items():
            alarm_vals = np.full(n, "", dtype=object)
            for key, d in devs.items():
                alarm_vals = np.where(np.abs(d) > 1, np.where(alarm_vals == "", key, alarm_vals + ";" + key), alarm_vals)
            summ = pd.DataFrame(dict(shot_id=shot_ids, press_id=press, unit_id=f"CPU-{press[-2:]}", mold_id=m_id,
                                     sensor_id=sid, sensor_position=pos, cavity_no=sc, job_id=job_field, shot_ts=ts,
                                     cycle_no=cyc_no, template_match_score=np.round(match, 3),
                                     alarm_state=recorded_alarm, alarm_values=np.where(alarm_vals == "", None, alarm_vals),
                                     sort_signal=np.where(sort, "reject", "pass"), curve_retained=retain))
            for key in ["fill_time_to_sensor_s", "peak_pressure_bar", "time_to_peak_s", "pressure_at_transfer_bar",
                        "fill_integral", "pack_integral", "cycle_integral", "end_of_fill_pressure_bar",
                        "gate_seal_time_s", "cooling_rate"]:
                v = res[key].astype(float)
                summ[key] = np.where(drop, np.nan, np.round(v, 4))
            sink("summary", summ, m_id, pd.Timestamp(ts[0]))

            # retained curves at 100 Hz for the full cycle, with the sensor's baseline offset
            if sink is not None:
                ri = np.where(retain)[0]
                n_full = int(cyc * 100)
                for c0 in range(0, len(ri), 300):
                    sel = ri[c0:c0 + 300]
                    sub = {kk: v[sel] for kk, v in par.items()}
                    P = construct(sub, T, drool[sel] if drool.any() else None) * sensor_gain[sid]
                    P = P[:, ::5]
                    full = np.zeros((len(sel), n_full), np.float32)
                    w_ = min(n_full, P.shape[1])
                    full[:, :w_] = P[:, :w_]
                    off = sensor_offset[sid] + 0.002 * (ts[sel] - np.datetime64(START_DATE)) / np.timedelta64(1, "D")
                    full = full + off[:, None].astype(np.float32) + rng.normal(0, 0.6, full.shape).astype(np.float32)
                    full = np.where(drop[sel][:, None], np.nan, full)
                    sink("curves", pd.DataFrame(dict(
                        shot_id=np.repeat(shot_ids[sel], n_full).astype(np.int64),
                        sensor_id=sid,
                        t_ms=np.tile(np.arange(n_full, dtype=np.int32) * 10, len(sel)),
                        pressure_bar=np.round(full.ravel(), 1).astype(np.float32))), m_id, pd.Timestamp(ts[0]))

        # ── Machine-side record ──
        pg = next(v for v in sens_sum.values() if v[0] == "post_gate")
        ppar, pres = pg[2], pg[3]
        mach = pd.DataFrame(dict(
            shot_id=shot_ids, press_id=press, job_id=job, shot_ts=ts,
            cycle_time_s=np.round(cyc * (1 + rng.normal(0, 0.004, n)), 2),
            fill_time_s=np.round(ppar["t_tr"] * (1 + rng.normal(0, 0.01, n)), 3),
            switchover_position_mm=np.round(12.0 * (1 - sw_delay) + rng.normal(0, 0.05, n), 2),
            switchover_pressure_bar=np.round(ppar["p_tr"] * 2.1 * (1 + rng.normal(0, 0.02, n)), 1),
            peak_injection_pressure_bar=np.round(ppar["p_tr"] * 2.3 * (1 + ppar["spike"]) * (1 + rng.normal(0, 0.02, n)), 1),
            cushion_mm=np.round(np.clip(5.0 - 14 * leak_eff + rng.normal(0, 0.08, n) * (1 + 20 * leak_eff), 0.2, None), 2),
            hold_pressure_bar=np.round(m["pack_bar"] * 1.9 * np.exp(hold), 1),
            hold_time_s=np.round(m["seal_s"] * 1.3, 2),
            recovery_time_s=np.round(cyc * 0.32 * np.exp(0.4 * lv) * (1 + regrind / 100 * 0.5) * (1 + rng.normal(0, 0.02, n)), 2),
            back_pressure_bar=np.round(8 + rng.normal(0, 0.2, n), 2),
            screw_rpm=np.round(110 + rng.normal(0, 1.0, n), 1),
            barrel_zone_1_actual_c=np.round(250 + melt * 0.3 + rng.normal(0, 0.4, n), 1),
            barrel_zone_2_actual_c=np.round(265 + melt * 0.6 + rng.normal(0, 0.4, n), 1),
            barrel_zone_3_actual_c=np.round(275 + melt + (-25 * (g10_fill > 0.02)) + rng.normal(0, 0.4, n), 1),
            barrel_zone_4_actual_c=np.round(280 + melt + rng.normal(0, 0.4, n), 1),
            nozzle_actual_c=np.round(282 + melt + rng.normal(0, 0.5, n), 1),
            mold_temp_a_c=np.round(80 + dT - dT_ab / 2 + rng.normal(0, 0.2, n), 1),
            mold_temp_b_c=np.round(80 + dT + dT_ab / 2 + rng.normal(0, 0.2, n), 1),
            clamp_tonnage=np.round(180 + rng.normal(0, 0.8, n), 1),
            press_alarm_code=np.where(rng.random(n) < 0.001, "E-114", None),
        ))
        sink("machine", mach, m_id, pd.Timestamp(ts[0]))

        # ── Piece quality: weight, critical dimension, defects ──
        dv = pg[8]
        fill_d = np.nan_to_num(dv.get("fill_integral", np.zeros(n)))
        pack_d = np.nan_to_num(dv.get("pack_integral", np.zeros(n)))
        peak_d = np.nan_to_num(dv.get("peak_pressure_bar", np.zeros(n)))
        gs_d = np.nan_to_num(dv.get("gate_seal_time_s", np.zeros(n)))
        eof_s = next((v for v in sens_sum.values() if v[0] == "end_of_fill"), None)
        eof_d = np.nan_to_num(eof_s[8].get("end_of_fill_pressure_bar", fill_d)) if eof_s else fill_d
        # the defect drivers exclude the blocking step, which moves the values without moving quality
        band_pack = C["alarm_band"]["pack_integral"]
        # only shots still judged against the pre-block template carry the step in their deviation
        g9_unref = g9_step * (seg_of == 0)
        pack_true = pack_d - g9_unref / band_pack
        fill_true = fill_d - 0.5 * g9_unref / C["alarm_band"]["fill_integral"]
        eof_true = eof_d
        # mechanism contributions to the drivers, in band units
        contrib_fill = {"G1": 0.8 * lv_g1 / C["alarm_band"]["fill_integral"], "G5": 0.8 * lv_g5 / C["alarm_band"]["fill_integral"],
                        "G3": (0.7 * g3 + 0.24 * g3) / C["alarm_band"]["fill_integral"],
                        "G4": -2.5 * leak / C["alarm_band"]["fill_integral"],
                        "G8": (g8_fill + 0.8 * lv_g8) / C["alarm_band"]["fill_integral"],
                        "G10": (0.8 * g10_fill - 2.5 * g10_leak) / C["alarm_band"]["fill_integral"]}
        contrib_pack = {"G1": -0.25 * lv_g1 / band_pack, "G5": -0.25 * lv_g5 / band_pack, "G3": g3 / band_pack,
                        "G4": (base_noise_pack - base_noise_pack / (1 + C["ring_var_coef"] * leak_eff)) / band_pack,
                        "G7": -0.03 * tipw / band_pack, "G8": (g8_pack - 0.25 * lv_g8) / band_pack,
                        "G10": (-0.25 * g10_fill) / band_pack - 2.0 * g10_leak / band_pack}
        contrib_eof = {"G2": -C["vent_eof_coef"] * vent / C["alarm_band"]["end_of_fill_pressure_bar"],
                       **{kk: v * C["alarm_band"]["fill_integral"] / C["alarm_band"]["end_of_fill_pressure_bar"]
                          for kk, v in contrib_fill.items()}}

        prod = idx >= approval_i
        density_dev = rng.normal(0, 0.0006) + np.zeros(n)      # lot-level density, small
        # shots the hourly audit samples from, and the first-shot approval shot
        hrs = (ts - ts[0]) / np.timedelta64(1, "h")
        audit_start = [approval_i] + [int(i) for i in np.searchsorted(hrs, np.arange(np.ceil(hrs[approval_i]), hrs[-1], LABELS["audit_every_h"]))]
        audit_mask = np.zeros(n, bool)
        audit_plan = []
        for a0 in audit_start:
            if a0 >= n:
                continue
            size = int(rng.integers(*LABELS["audit_pieces"]))
            n_sh = int(np.ceil(size / cav))
            audit_mask[a0:min(n, a0 + n_sh)] = True
            audit_plan.append((a0, size))
        out["audit_plan"].append(pd.DataFrame(dict(shot_id=[int(shot_ids[a0]) for a0, _ in audit_plan],
                                                   sample_size=[s_ for _, s_ in audit_plan],
                                                   is_approval=[a0 == approval_i for a0, _ in audit_plan],
                                                   job_id=job, press_id=press, mold_id=m_id)))
        cav_dev = cavity_offsets[m_id]["dev"]
        sensed = {c for _, c in m["sensors"]}
        dim_nom, tol = m["dim_mm"], m["tol_mm"]
        pieces_defects = []
        for c in range(1, cav + 1):
            active = ~((blocked == 1) & (c == block_cav)) & prod
            off = 0.0 if c in sensed else cav_dev[c - 1]
            fd, pd_, ed = fill_true + off, pack_true + off, eof_true + off
            p = {}
            p["short_shot"] = 0.9 / (1 + np.exp((np.minimum(fd, ed) + 1.3) / 0.10))
            xf = pd_ + 0.5 * peak_d
            p["flash"] = 0.06 / (1 + np.exp(-(xf - 1.2) / 0.30)) + 0.45 / (1 + np.exp(-(xf - 1.7) / 0.12)) * (1 + mold_shots[m_id] / 2e6)
            thick = 3.5 if m_id == "M-2301" else (1.3 if m["housing"] else 1.0)
            p["sink"] = 0.012 * thick / (1 + np.exp((pd_ + 1.0) / 0.40)) + 0.35 / (1 + np.exp((pd_ + 1.5) / 0.12))
            p["void"] = 0.35 * p["sink"]
            p["weld_line"] = (0.030 * (2.0 if m["class_a"] else 1.0) / (1 + np.exp((ed + 0.85) / 0.30)) * (1 + np.clip(-melt / 5, 0, 2))
                              if m["weld_line"] else np.zeros(n))
            p["splay"] = 0.011 / (1 + np.exp(-(moist - 0.55) / 0.08)) if R["hygroscopic"] else np.zeros(n)
            p["burn"] = 0.009 / (1 + np.exp(-(vent - 0.020) / 0.003)) * np.exp(dlog_vel)
            p["warp"] = 0.035 / (1 + np.exp(-(np.abs(dT_ab) - 3.2) / 0.6)) if m["housing"] else np.zeros(n)
            p["black_specks"] = regrind / 100 * 0.004 + (lv > 0.12) * 0.002
            p["contamination"] = np.zeros(n)
            p["gate_vestige"] = 0.005 * tipw ** 2 if m_id in ("M-2041", "M-2043") else np.zeros(n)
            p["other"] = np.zeros(n)
            p["dimensional"] = np.zeros(n)
            if g10_kind == "nozzle_drool":
                p["other"] = p["other"] + 0.08 * g10_mask
            if g10_kind == "wrong_material":
                p["other"] = p["other"] + 0.15 * g10_mask

            # continuous quality: weight and critical dimension
            # weight: nominal x (1 + a x pack integral deviation from template + b x fill deviation)
            wt_true = (m["weight_g"] * (1 + 0.12 * pd_ * band_pack + 0.03 * fd * C["alarm_band"]["fill_integral"] + density_dev)
                       + m["weight_g"] * cavity_offsets[m_id]["weight"][c - 1] + rng.normal(0, m["weight_g"] * 0.0014, n))
            # critical dimension: shrinkage falls with pack integral and a later gate seal, rises with mold temperature
            dim_true = (dim_nom + tol * (0.37 * pd_ + 0.17 * gs_d - 0.035 * dT)
                        + tol * cavity_offsets[m_id]["dim"][c - 1] + rng.normal(0, tol * 0.062, n))
            dim_bad = np.abs(dim_true - dim_nom) > tol

            u = rng.random((n, len(CODES)))
            probs = np.vstack([p[k] + DEFECTS["base_rate"][k] for k in CODES]).T
            fired = (u < probs) & active[:, None]
            fired[:, CODES.index("dimensional")] |= dim_bad & active
            any_bad = fired.any(axis=1)
            for i in np.where(any_bad)[0]:
                ks = np.where(fired[i])[0]
                k = ks[0] if len(ks) == 1 else int(rng.choice(ks, p=probs[i, ks] / probs[i, ks].sum()))
                code = CODES[k]
                base = DEFECTS["base_rate"][code]
                total = probs[i, k] if code != "dimensional" else 1.0
                # attribute to a mechanism
                if code != "dimensional" and rng.random() < base / max(total, 1e-12):
                    cause = "G0"
                else:
                    if code in ("short_shot",):
                        cc = contrib_fill if fd[i] <= ed[i] else contrib_eof
                        w = {kk: max(0.0, -v[i]) for kk, v in cc.items()}
                    elif code in ("flash",):
                        w = {kk: max(0.0, v[i]) for kk, v in contrib_pack.items()}
                    elif code in ("sink", "void"):
                        w = {kk: max(0.0, -v[i]) for kk, v in contrib_pack.items()}
                    elif code == "dimensional":
                        sgn = np.sign(dim_true[i] - dim_nom)
                        w = {kk: max(0.0, sgn * 0.37 * v[i]) for kk, v in contrib_pack.items()}
                        w["G6"] = max(0.0, -sgn * 0.035 * dT[i])
                    elif code == "weld_line":
                        w = {kk: max(0.0, -v[i]) for kk, v in contrib_eof.items()}
                        w["G8"] = w.get("G8", 0) + max(0.0, -melt[i] / 20)
                    elif code == "splay":
                        w = {"G5": 1.0}
                    elif code == "burn":
                        w = {"G2": 1.0}
                    elif code == "warp":
                        w = {"G6": 1.0}
                    elif code == "gate_vestige":
                        w = {"G7": 1.0}
                    elif code == "other" and g10_mask[i]:
                        w = {"G10": 1.0}
                    else:
                        w = {}
                    if g10_mask[i] and code in ("short_shot", "sink", "void", "flash", "splay"):
                        w["G10"] = w.get("G10", 0) + 2.0
                    tot = sum(w.values())
                    if tot <= 1e-9:
                        cause = "G0"
                    else:
                        keys = list(w)
                        cause = keys[int(rng.choice(len(keys), p=np.array([w[kk] for kk in keys]) / tot))]
                pieces_defects.append((int(shot_ids[i]), c, code, cause))

            # audit sample truth: weight and dimension for every piece (audits sample from these)
            keep = audit_mask & active
            out["audits_truth"].append(pd.DataFrame(dict(shot_id=shot_ids[keep], cavity=c,
                                                         weight_true=wt_true[keep].astype(np.float32),
                                                         dim_true=dim_true[keep].astype(np.float32),
                                                         defect_code=np.array([None] * int(keep.sum()), dtype=object))))
        if pieces_defects:
            out["defects"].append(pd.DataFrame(pieces_defects, columns=["shot_id", "cavity", "defect_code", "root_cause_code"]))

        # state history for the reference register and checks
        sink("state", pd.DataFrame(dict(shot_id=shot_ids, run_no=r["run_no"], lv_g1=lv_g1.astype(np.float32),
                                              moist=moist.astype(np.float32), vent=vent.astype(np.float32),
                                              leak=leak_eff.astype(np.float32), dT=dT.astype(np.float32),
                                              dT_ab=dT_ab.astype(np.float32), g3=g3.astype(np.float32),
                                              tipw=tipw.astype(np.float32), g10=g10_mask, blocked=blocked.astype(np.int8),
                                              dlog_vel=dlog_vel.astype(np.float32), sw_delay=sw_delay.astype(np.float32),
                                              melt=melt.astype(np.float32), hold=hold.astype(np.float32),
                                              regrind=regrind.astype(np.float32),
                                              fill_dev=fill_d.astype(np.float32), pack_dev=pack_d.astype(np.float32),
                                              eof_dev=eof_d.astype(np.float32), gs_dev=gs_d.astype(np.float32))), m_id, pd.Timestamp(ts[0]))

        # approval record
        out["approvals"].append(dict(job_id=job, press_id=press, mold_id=m_id, approval_ts=approval_ts,
                                     technician_id=tech, tenure_days=tenure, shots_to_approval=n_setup,
                                     template_id_established=template_id, set_ts=ts0, run_no=r["run_no"],
                                     last_shot_ts=pd.Timestamp(ts[-1]), g10_kind=g10_kind, blocked_cavity=block_cav))

        vent_since[m_id] += n
        tip_since[m_id] += n
        mold_shots[m_id] += n

    # sensor recalibrations: yearly per mold
    for m_id in MOLDS:
        for d in ["2025-04-14", "2026-02-09"]:
            out["maint"].append(dict(mold_id=m_id, event_ts=pd.Timestamp(d) + pd.Timedelta(hours=7), event_type="sensor_recalibration",
                                     shots_at_event=None, technician_id="TC-01", notes="Cavity pressure sensors recalibrated"))
    return dict(runs=pd.DataFrame(runs), lots=lots, tcu=tcu_services, ring_changes=ring_changes, **out)
