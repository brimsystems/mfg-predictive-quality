"""
Writes every source-system extract.

The per-shot tables stream to Parquet as the cell build produces them, so the
full fifteen months never sit in memory at once:

  raw/monitoring/cavity_curves/month=YYYY-MM/mold=M-xxxx/part-N.parquet
  raw/monitoring/cavity_shot_summary/month=YYYY-MM/part-N.parquet
  raw/mes/machine_shot_data/month=YYYY-MM/part-N.parquet

Everything else is CSV under raw/<system>/. Quality engineering's root-cause
register and the cell's state history go to reference/, outside the load path.

Usage:  python -m data_source.generate.run_generator [--runs N]
"""
import argparse
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CELL_PRESSES, MOLDS, RAW_DIR, REFERENCE_DIR, SAMPLE_SIZE, SAMPLES_DIR
from .dirty.transformations import dirty_defect, dirty_part
from .generators import plant
from .generators.inspections import build_labels
from .generators.molding_cell import build_cell

SYSTEM = {
    "cavity_shot_summary": "monitoring", "cavity_curves": "monitoring", "templates": "monitoring",
    "machine_shot_data": "mes", "presses": "mes",
    "production_orders": "erp", "part_attributes": "erp",
    "setpoint_changes": "qms", "first_shot_approvals": "qms", "qc_audits": "qms", "qc_audit_pieces": "qms",
    "sort_dispositions": "qms", "scrap_tallies": "qms", "customer_returns": "qms",
    "inspection_records": "qms", "scrap_events": "qms",
    "resin_lots": "materials", "material_loads": "materials", "dryer_log": "materials",
    "mold_maintenance": "toolroom", "equipment_service": "toolroom", "process_sheets": "engineering",
    "technicians": "hr", "inspectors": "hr", "operators": "hr",
}
FLUSH_ROWS = {"curves": 6_000_000, "summary": 400_000, "machine": 400_000, "state": 400_000}


class ParquetSink:
    """Buffers the cell's streamed parts by partition and writes zstd Parquet files."""

    def __init__(self):
        self.buf = defaultdict(list)
        self.rows = defaultdict(int)
        self.parts = defaultdict(int)
        self.total = defaultdict(int)
        self.samples = {}
        self.shot_month = None

    def _dir(self, kind, month, mold):
        if kind == "curves":
            return RAW_DIR / "monitoring" / "cavity_curves" / f"month={month}" / f"mold={mold}"
        if kind == "summary":
            return RAW_DIR / "monitoring" / "cavity_shot_summary" / f"month={month}"
        if kind == "machine":
            return RAW_DIR / "mes" / "machine_shot_data" / f"month={month}"
        return REFERENCE_DIR / "shot_state" / f"month={month}"

    def __call__(self, kind, df, mold_id, ts):
        if kind in ("summary", "machine"):
            month = pd.to_datetime(df["shot_ts"]).dt.strftime("%Y-%m")
            if kind == "summary":
                self.shot_month = pd.Series(month.to_numpy(), index=df["shot_id"].to_numpy()).groupby(level=0).first()
        else:
            # curves and state carry no timestamp; the shot's month comes from its summary row
            month = df["shot_id"].map(self.shot_month) if self.shot_month is not None else None
            if month is None or month.isna().any():
                month = pd.Series(pd.Timestamp(ts).strftime("%Y-%m"), index=df.index)
        if kind not in self.samples:
            self.samples[kind] = df.head(SAMPLE_SIZE)
        for mo, part in df.groupby(month.to_numpy(), sort=False):
            key = (kind, mo, mold_id if kind == "curves" else None)
            self.buf[key].append(part)
            self.rows[key] += len(part)
            if self.rows[key] >= FLUSH_ROWS[kind]:
                self._flush(key)

    def _flush(self, key):
        if not self.buf[key]:
            return
        kind, month, mold = key
        d = self._dir(kind, month, mold)
        d.mkdir(parents=True, exist_ok=True)
        df = pd.concat(self.buf[key], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / f"part-{self.parts[key]}.parquet",
                       compression="zstd", row_group_size=1_000_000)
        self.parts[key] += 1
        self.total[kind] += len(df)
        self.buf[key], self.rows[key] = [], 0

    def close(self):
        for key in list(self.buf):
            self._flush(key)


def _save(df, name, sample=True):
    d = RAW_DIR / SYSTEM[name]
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv", index=False)
    if sample:
        _sample(df, name)
    print(f"  [{SYSTEM[name]:>11}]  {name:<24} {len(df):>10,} rows")


def _sample(df, name):
    d = SAMPLES_DIR / SYSTEM[name]
    d.mkdir(parents=True, exist_ok=True)
    df.head(SAMPLE_SIZE).to_csv(d / f"{name}_sample.csv", index=False)


def _ref(df, name):
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REFERENCE_DIR / f"{name}.parquet", index=False, compression="zstd")
    print(f"  [  reference]  {name:<24} {len(df):>10,} rows")


def _setpoint_changes(changes, runs, sheets, rng):
    """Values in engineering units: the press logs old and new setpoints, not deltas."""
    base = sheets.set_index(["mold_id", "press_id"])
    job_mold = runs.set_index("job_id")["mold_id"]
    rows = []
    for job, g in changes.sort_values("change_ts").groupby("job_id", sort=False):
        sh = base.loc[(job_mold[job], g["press_id"].iat[0])]
        cur = {"hold_pressure_bar": sh["hold_pressure_bar"], "injection_velocity_mm_s": sh["injection_velocity_mm_s"],
               "switchover_position_mm": sh["switchover_position_mm"], "melt_temp_c": sh["melt_temp_c"],
               "mold_temp_c": sh["mold_temp_c"]}
        for c in g.itertuples():
            old = cur[c.parameter]
            if pd.isna(c.new_value):
                new = old                         # sign-off at approval: the setpoint confirmed, not moved
            elif c.parameter in ("hold_pressure_bar", "injection_velocity_mm_s"):
                new = old * np.exp(c.new_value)
            elif c.parameter == "switchover_position_mm":
                new = old - 12.0 * c.new_value
            else:
                new = old + c.new_value
            new = round(float(new), 1)
            cur[c.parameter] = new
            rows.append(dict(change_id=c.change_id, press_id=c.press_id, job_id=c.job_id, change_ts=c.change_ts.floor("s"),
                             parameter=c.parameter, old_value=round(float(old), 1), new_value=new,
                             technician_id=c.technician_id, reason_code=c.reason_code, approved_by=c.approved_by))
    # the process sheet load at each mold set, and the approval sign-off
    n = len(rows)
    for r in runs.itertuples():
        sh = base.loc[(r.mold_id, r.press_id)]
        for p in ("hold_pressure_bar", "melt_temp_c", "cycle_time_s"):
            n += 1
            rows.append(dict(change_id=f"SC-C{n}", press_id=r.press_id, job_id=r.job_id,
                             change_ts=(r.set_ts + pd.Timedelta(minutes=float(rng.uniform(5, 40)))).floor("s"),
                             parameter=p, old_value=None, new_value=float(sh[p]), technician_id=r.technician_id,
                             reason_code="changeover" if rng.random() < 0.7 else "unknown",
                             approved_by=None if rng.random() < 0.10 else "QE-01"))
    return pd.DataFrame(rows).sort_values("change_ts").reset_index(drop=True)


def _templates_wide(tmpl, shots):
    ref = (shots[shots["curve_retained"] & shots["in_production"]].groupby("template_id")["shot_id"].first())
    t = tmpl.copy()
    keys = ["template_id", "mold_id", "press_id", "sensor_id", "effective_from", "established_by", "approved_by",
            "source_run_job_id", "match_score_threshold", "widened", "reason"]
    w = t.pivot_table(index=keys, columns="summary_value",
                      values=["template_value", "warning_low", "warning_high", "alarm_low", "alarm_high"],
                      aggfunc="first")
    w.columns = [f"{s}_{v}" for v, s in w.columns]
    w = w.reset_index().rename(columns={"widened": "band_widened"})
    w.insert(8, "template_curve_ref", w["template_id"].map(ref))
    num = w.select_dtypes("number").columns
    w[num] = w[num].round(4)
    return w


def run(max_runs=None, run_stride=1):
    t0 = time.time()
    for d in (RAW_DIR, REFERENCE_DIR, SAMPLES_DIR):
        if d.exists():
            shutil.rmtree(d)
    rng = np.random.default_rng(11)

    print("Cell build (IM-11, IM-12): curves, summaries, machine data, state")
    sink = ParquetSink()
    cell = build_cell(sink, max_runs=max_runs, run_stride=run_stride)
    sink.close()
    for kind, name in [("summary", "cavity_shot_summary"), ("machine", "machine_shot_data"), ("curves", "cavity_curves")]:
        print(f"  [{SYSTEM[name]:>11}]  {name:<24} {sink.total[kind]:>10,} rows (Parquet)")
        _sample(sink.samples[kind], name)
    print(f"  ({time.time() - t0:,.0f} s)")

    for k in ("shots", "defects", "audit_plan", "audits_truth"):
        cell[k] = pd.concat(cell[k], ignore_index=True)
    cell["approvals"] = pd.DataFrame(cell["approvals"])
    runs = cell["runs"].merge(cell["approvals"][["job_id", "approval_ts", "technician_id", "last_shot_ts"]], on="job_id")

    parts = plant.part_attributes()
    sheets = plant.process_sheets()
    print("Labels: sort review, audits, approvals, tallies, returns")
    lab = build_labels(cell, parts)

    # ── Monitoring ──
    _save(_templates_wide(pd.DataFrame(cell["templates"]), cell["shots"]), "templates")

    # ── MES / ERP ──
    _save(plant.presses(), "presses")
    _save(parts, "part_attributes")
    _save(sheets, "process_sheets")
    sd = lab["sort_dispositions"].merge(cell["shots"][["shot_id", "true_job_id", "press_id", "mold_id"]], on="shot_id")
    sort_scrap = []
    for r in sd[sd["pieces_confirmed_defective"] > 0].itertuples():
        for code in str(r.defect_codes).split(";"):
            sort_scrap.append(dict(job_id=r.true_job_id, press_id=r.press_id, mold_id=r.mold_id, defect_code=code,
                                   qty=r.pieces_confirmed_defective / len(str(r.defect_codes).split(";")),
                                   date=r.reviewed_ts.date()))
    job_mold = runs.set_index("job_id")["mold_id"]
    tl = lab["scrap_tallies"]
    cell_scrap = pd.concat([pd.DataFrame(sort_scrap),
                            pd.DataFrame(dict(job_id=tl["job_id"], press_id=tl["press_id"],
                                              mold_id=tl["job_id"].map(job_mold), defect_code=tl["defect_code"],
                                              qty=tl["qty"], date=pd.to_datetime(tl["shift_date"]).dt.date))])
    cell_scrap["qty"] = np.ceil(cell_scrap["qty"]).astype(int)
    orders, insp, scrap = plant.plant_jobs(runs, cell_scrap, parts)
    orders["part_number"] = [dirty_part(p) if not i else p for p, i in zip(orders["part_number"], orders["instrumented"])]
    orders = orders.drop(columns=["instrumented"])
    insp["defect_code_raw"] = [dirty_defect(c) for c in insp["defect_code_clean"]]
    scrap["defect_code"] = [dirty_defect(c) for c in scrap["defect_code"]]
    _save(orders, "production_orders")
    _save(insp, "inspection_records")
    _save(scrap, "scrap_events")

    # ── QMS ──
    ch = pd.DataFrame(cell["changes"])
    _save(_setpoint_changes(ch, runs, sheets, rng), "setpoint_changes")
    for name in ("first_shot_approvals", "qc_audits", "qc_audit_pieces", "sort_dispositions", "scrap_tallies",
                 "customer_returns"):
        _save(lab[name], name)

    # ── Materials, tool room, HR ──
    lots = cell["lots"]
    _save(lots.drop(columns=["true_mfi"]), "resin_lots")
    loads = pd.DataFrame(cell["loads"])
    _save(loads.drop(columns=["actual_ts", "residence_h"]).assign(load_ts=lambda d: d["load_ts"].dt.floor("min")),
          "material_loads")
    _save(plant.dryer_log(), "dryer_log")
    maint = pd.DataFrame(cell["maint"]).sort_values("event_ts").reset_index(drop=True)
    _save(maint, "mold_maintenance")
    eq = [dict(press_id=p, equipment="mold_temperature_controller", event_ts=pd.Timestamp(t) + pd.Timedelta(hours=7),
               event_type="service") for p, ts in cell["tcu"].items() for t in ts]
    eq += [dict(press_id=p, equipment="check_ring", event_ts=pd.Timestamp(t).floor("min"), event_type="replace")
           for p, dates in cell["ring_changes"].items() for t in dates]
    _save(pd.DataFrame(eq).sort_values("event_ts"), "equipment_service")
    tech, inspectors, ops = plant.people()
    _save(tech, "technicians")
    _save(inspectors, "inspectors")
    _save(ops, "operators")

    # ── Reference (outside the load path) ──
    print("Reference")
    _ref(lab["register"], "root_cause_register")
    _ref(cell["shots"], "shot_truth")
    _ref(cell["audits_truth"], "audit_piece_truth")
    _ref(lab["audit_piece_links"], "audit_piece_links")
    _ref(lots[["lot_id", "true_mfi"]], "lot_truth")
    _ref(loads[["load_id", "actual_ts", "residence_h"]], "load_truth")
    _ref(ch, "change_truth")
    _ref(cell["approvals"], "run_log")
    print(f"Done in {time.time() - t0:,.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None, help="limit the cell build to the first N runs")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth run (tuning passes)")
    a = ap.parse_args()
    run(a.runs, a.stride)
