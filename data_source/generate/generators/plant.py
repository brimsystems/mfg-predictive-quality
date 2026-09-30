"""
Generator: presses, part_attributes, process_sheets, technicians, inspectors,
operators, resin_lots, material_loads, dryer_log, mold_maintenance,
setpoint_changes, templates, production_orders, inspection_records,
scrap_events
Source systems: MES, ERP, HR, receiving, dryer controllers, tool room / mold
maintenance log, press controllers and the QMS change log, monitoring units.

The ERP, MES and QMS job records cover all 24 presses; the cell's own records
(IM-11 and IM-12) come from the cell build and are shaped here into each
system's extract.
"""
import numpy as np
import pandas as pd

from ..config import (CELL as C, CELL_PRESSES, END_DATE, LABELS, MOLDS, N_PRESSES, PLANT, PRESS_TONNAGE, RANDOM_SEED,
                      RESINS, SNAPSHOT_DATE, START_DATE)
from .molding_cell import TECHS

OTHER_CODES = ["short_shot", "flash", "sink", "splay", "warp", "black_specks", "contamination", "dimensional", "other"]


def presses():
    rng = np.random.default_rng(RANDOM_SEED + 41)
    rows = []
    for i in range(1, N_PRESSES + 1):
        pid = f"IM-{i:02d}"
        cell = pid in CELL_PRESSES
        rows.append(dict(press_id=pid, press_type="electric" if cell or rng.random() < 0.4 else "hydraulic",
                         tonnage=200 if cell else int(rng.choice(PRESS_TONNAGE)),
                         install_year=2020 if cell else int(rng.integers(2006, 2023)),
                         opc_ua=bool(cell or rng.random() < 0.3),
                         cavity_pressure_unit=f"CPU-{pid[-2:]}" if cell else None,
                         cell="instrumented" if cell else None))
    return pd.DataFrame(rows)


def part_attributes():
    rows = []
    for m_id, m in MOLDS.items():
        rows.append(dict(part_id=m["part"], mold_id=m_id, description=m["desc"], customer_id=m["customer"],
                         program=m["program"], resin=m["resin"], cavities=m["cavities"], nominal_weight_g=m["weight_g"],
                         critical_dimension_mm=m["dim_mm"], dimension_tolerance_mm=m["tol_mm"],
                         nominal_wall_mm=m["thick"], class_a_cosmetic=m["class_a"], has_weld_line=m["weld_line"],
                         flat_housing=m["housing"],
                         gauge_sd_weight_g=round(m["weight_g"] * 0.0018, 4),
                         gauge_sd_dimension_mm=round(m["tol_mm"] * 2 * LABELS["gauge_rr_frac"] / 6, 4),
                         standard_cost=round(m["weight_g"] * 0.012 + 0.18 * (m["cycle_s"] / m["cavities"]), 3)))
    return pd.DataFrame(rows)


def process_sheets():
    rows = []
    for m_id, m in MOLDS.items():
        for p in CELL_PRESSES:
            rows.append(dict(mold_id=m_id, press_id=p, revision="C" if p == "IM-11" else "B",
                             injection_velocity_mm_s=round(60 / m["fill_s"], 1), switchover_position_mm=12.0,
                             hold_pressure_bar=round(m["pack_bar"] * 1.9, 0), hold_time_s=round(m["seal_s"] * 1.3, 1),
                             cooling_time_s=round(m["cycle_s"] * 0.45, 1), melt_temp_c=282.0, mold_temp_c=80.0,
                             cycle_time_s=m["cycle_s"],
                             window_cushion_mm="3.5-6.5", window_fill_time_pct=5.0, window_hold_pressure_pct=3.0,
                             window_mold_temp_c=3.0, window_melt_temp_c=5.0, validated_on="2023-04-17"))
    return pd.DataFrame(rows)


def people():
    tech = pd.DataFrame([dict(technician_id=t, hire_date=d, shift={"TC-01": "A", "TC-04": "A", "TC-02": "B", "TC-05": "B",
                                                                   "TC-03": "C", "TC-06": "C"}[t]) for t, d in TECHS])
    insp = pd.DataFrame([dict(inspector_id=f"QC-0{i}", shift="ABCABC"[i - 1],
                              hire_date=["2014-05-12", "2017-02-06", "2019-08-19", "2021-10-04", "2024-09-16", "2025-01-13"][i - 1])
                         for i in range(1, 7)])
    rng = np.random.default_rng(RANDOM_SEED + 43)
    ops = pd.DataFrame([dict(operator_id=f"OP{i:03d}", shift=str(rng.choice(["A", "B", "C"])),
                             hire_date=str((pd.Timestamp("2025-06-01") - pd.Timedelta(days=int(rng.integers(90, 5000)))).date()))
                        for i in range(1, 60)])
    return tech, insp, ops


def dryer_log():
    rng = np.random.default_rng(RANDOM_SEED + 45)
    ts = pd.date_range(START_DATE, END_DATE + pd.Timedelta(days=1), freq="15min")
    rows = []
    for d in C["dryer_ids"]:
        n = len(ts)
        dew_actual = -40 + rng.normal(0, 1.5, n)
        dew_read = dew_actual.copy()
        if d == C["faulty_dryer"]:
            f = (ts >= C["faulty_from"]) & (ts <= C["faulty_to"])
            dew_actual = np.where(f, dew_actual + 14, dew_actual)          # the dryer was running wet
            dew_read = np.where(f, dew_actual - C["faulty_dew_offset_c"], dew_actual)
        rows.append(pd.DataFrame(dict(dryer_id=d, ts=ts, setpoint_temp_c=80.0 if d != "DR-3" else 120.0,
                                      actual_temp_c=np.round((80.0 if d != "DR-3" else 120.0) + rng.normal(0, 0.8, n), 1),
                                      dew_point_c=np.round(dew_read, 1),
                                      hopper_level_pct=np.round(np.clip(60 + 30 * np.sin(np.arange(n) / 17) + rng.normal(0, 5, n), 5, 100), 0),
                                      alarm_code=np.where(rng.random(n) < 0.0008, "D-21", None))))
    return pd.concat(rows, ignore_index=True)


def plant_jobs(cell_runs: pd.DataFrame, cell_scrap: pd.DataFrame, parts: pd.DataFrame):
    """Work orders, final inspection and scrap for all 24 presses, job level."""
    rng = np.random.default_rng(RANDOM_SEED + 47)
    orders, insp, scrap = [], [], []
    wo = 60000
    customers = [f"Customer {c}" for c in "ABCDEFGHJKLNPQ"]
    for i in range(1, N_PRESSES + 1):
        pid = f"IM-{i:02d}"
        if pid in CELL_PRESSES:
            continue
        d = pd.Timestamp(START_DATE)
        while d < pd.Timestamp(END_DATE):
            d += pd.Timedelta(days=float(rng.exponential(7 / PLANT["jobs_per_press_week"])))
            qty = int(rng.integers(*PLANT["qty"]))
            mold = f"M-{int(rng.integers(1000, 1999))}"
            rate = PLANT["scrap_rate"] * float(rng.gamma(2.0, 0.5))
            failed = int(rng.binomial(qty, min(rate, 0.3)))
            wid = f"WO-{wo}"
            wo += 1
            orders.append(dict(work_order_id=wid, job_id=wid, part_number=f"P-{mold[2:]}1", mold_id=mold,
                               customer=str(rng.choice(customers)), press_id=pid, quantity_ordered=qty,
                               order_date=(d - pd.Timedelta(days=int(rng.integers(5, 30)))).date(),
                               scheduled_start=d.floor("h"), actual_start=(d + pd.Timedelta(hours=float(rng.uniform(0, 8)))).floor("min"),
                               resin=str(rng.choice(list(RESINS))), program=str(rng.choice(["industrial", "consumer", "automotive"])),
                               instrumented=False))
            insp.append(dict(inspection_id=f"IN-{wo}", work_order_id=wid,
                             inspection_date=(d + pd.Timedelta(days=float(rng.uniform(1, 5)))).floor("min"),
                             inspector_id=f"QC-0{int(rng.integers(1, 7))}", quantity_inspected=qty,
                             quantity_passed=qty - failed, quantity_failed=failed,
                             defect_code_clean=str(rng.choice(OTHER_CODES)) if failed else "none",
                             disposition="Scrap" if failed else "Pass"))
            for code in rng.choice(OTHER_CODES, size=min(3, max(1, failed // 50)), replace=False) if failed else []:
                q = max(1, failed // 3)
                unit = float(rng.uniform(0.08, 1.4))
                scrap.append(dict(work_order_id=wid, press_id=pid, scrap_date=(d + pd.Timedelta(days=1)).date(),
                                  defect_code=str(code), quantity_scrapped=q, unit_cost=round(unit, 3),
                                  material_cost=round(q * unit * 0.6, 2), labor_cost=round(q * unit * 0.4, 2),
                                  total_scrap_cost=round(q * unit, 2)))
    # cell jobs and the non-instrumented jobs between them on IM-11 and IM-12
    std = parts.set_index("mold_id")["standard_cost"]
    prev_end = {}
    for r in cell_runs.sort_values("set_ts").itertuples():
        m = MOLDS[r.mold_id]
        orders.append(dict(work_order_id=r.job_id, job_id=r.job_id, part_number=m["part"], mold_id=r.mold_id,
                           customer=m["customer"], press_id=r.press_id,
                           quantity_ordered=int(r.prod_days * 86400 * C["uptime"] / m["cycle_s"] * m["cavities"] / 100) * 100,
                           order_date=(r.set_ts - pd.Timedelta(days=int(rng.integers(5, 30)))).date(),
                           scheduled_start=r.set_ts.floor("h"), actual_start=r.set_ts.floor("min"), resin=m["resin"],
                           program=m["program"], instrumented=True))
        # between cell runs the press runs one or two non-instrumented jobs
        gap0 = prev_end.get(r.press_id, r.set_ts - pd.Timedelta(days=2))
        prev_end[r.press_id] = r.last_shot_ts
        k = int(rng.integers(1, 3))
        for q in range(k):
            wid = f"WO-{wo}"
            wo += 1
            d = gap0 + (r.set_ts - gap0) * (q + 0.1) / (k + 0.2)
            qty = int(rng.integers(2000, 20000))
            orders.append(dict(work_order_id=wid, job_id=wid, part_number=f"P-{int(rng.integers(1000, 1999))}1",
                               mold_id=f"M-{int(rng.integers(1000, 1999))}", customer=str(rng.choice(customers)),
                               press_id=r.press_id, quantity_ordered=qty, order_date=(d - pd.Timedelta(days=10)).date(),
                               scheduled_start=d.floor("h"), actual_start=d.floor("min"), resin=str(rng.choice(list(RESINS))),
                               program="industrial", instrumented=False))
    orders = pd.DataFrame(orders)
    cell_scrap = cell_scrap.copy()
    cell_scrap["unit_cost"] = cell_scrap["mold_id"].map(std)
    cs = (cell_scrap.groupby(["job_id", "press_id", "mold_id", "defect_code"])
          .agg(quantity_scrapped=("qty", "sum"), scrap_date=("date", "max"), unit_cost=("unit_cost", "first")).reset_index())
    cs["material_cost"] = (cs["quantity_scrapped"] * cs["unit_cost"] * 0.6).round(2)
    cs["labor_cost"] = (cs["quantity_scrapped"] * cs["unit_cost"] * 0.4).round(2)
    cs["total_scrap_cost"] = (cs["quantity_scrapped"] * cs["unit_cost"]).round(2)
    cs = cs.rename(columns={"job_id": "work_order_id"}).drop(columns=["mold_id"])
    scrap = pd.concat([pd.DataFrame(scrap), cs], ignore_index=True)
    scrap.insert(0, "scrap_id", [f"SC-{i + 1}" for i in range(len(scrap))])
    tot = cs.groupby("work_order_id")["quantity_scrapped"].sum()
    top = cs.sort_values("quantity_scrapped").groupby("work_order_id")["defect_code"].last()
    for r in orders[orders["instrumented"]].itertuples():
        f = int(tot.get(r.work_order_id, 0))
        insp.append(dict(inspection_id=f"IN-{r.work_order_id}", work_order_id=r.work_order_id,
                         inspection_date=r.actual_start + pd.Timedelta(days=4), inspector_id="QC-01",
                         quantity_inspected=r.quantity_ordered, quantity_passed=r.quantity_ordered - f,
                         quantity_failed=f, defect_code_clean=top.get(r.work_order_id, "none"),
                         disposition="Scrap" if f else "Pass"))
    return orders, pd.DataFrame(insp), scrap
