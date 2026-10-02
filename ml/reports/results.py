"""
Every number the reports quote, read from the current generation run: the warehouse
(dbt marts), the model results in ml/data/results, and the realism checks. Reports
import this module so a figure in one report is the same figure in the others.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.features import DATA_DIR, ROOT, TRAIN_END, VALID_END, connect

RES = DATA_DIR / "results"
REF = ROOT / "data_source" / "reference"


def q(sql):
    con = connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


@lru_cache(maxsize=None)
def load():
    R = {}
    R["checks"] = json.loads((REF / "checks.json").read_text())
    R["M"] = json.loads((RES / "measures.json").read_text())
    R["DM"] = json.loads((RES / "drift_mechanisms.json").read_text())
    R["ae"] = json.loads((RES / "autoencoder_summary.json").read_text())
    R["anomaly_events"] = pd.read_csv(RES / "measures_anomaly_events.csv", parse_dates=["start"])
    R["anomaly_thresholds"] = pd.read_csv(RES / "rolling_anomaly_thresholds.csv", parse_dates=["month"])
    R["vm_month"] = pd.read_csv(RES / "rolling_vm_metrics.csv", parse_dates=["month"])
    R["vm_pred"] = pd.read_parquet(RES / "rolling_vm_predictions.parquet")
    R["defect"] = pd.read_csv(RES / "defect_metrics_full.csv")
    R["defect_abl"] = pd.read_csv(RES / "defect_ablations.csv")
    R["audit_only"] = json.loads((RES / "defect_audit_only.json").read_text())["mean"]

    s = q("""
        select count(*) as shots, count(*) filter (where after_approval) as prod_shots,
               sum(active_cavities) filter (where after_approval) as pieces,
               count(*) filter (where after_approval and unit_sorted) as sorted_shots,
               count(*) filter (where after_approval and unit_alarm_state = 'alarm') as alarm_shots,
               avg((unit_alarm_state = recomputed_alarm_state)::int) filter (where after_approval) as unit_agreement,
               count(distinct job_id) as jobs, min(shot_ts) as first_ts, max(shot_ts) as last_ts,
               count(distinct date_trunc('hour', shot_ts)) as hours
        from fct_shot""").iloc[0].to_dict()
    R["shots"] = s
    R["sort"] = q("""
        select count(*) filter (where review_mode = 'indexed_tray') as tray_reviews,
               count(*) filter (where review_mode = 'per_shift') as shift_reviews,
               avg((pieces_confirmed_defective = 0)::int) filter (where review_mode = 'indexed_tray') as false_reject_share,
               sum(pieces_good) / sum(pieces_reviewed) as pieces_found_good_share,
               sum(pieces_confirmed_defective) as confirmed_pieces, sum(pieces_reviewed) as reviewed_pieces
        from stg_qms__sort_dispositions""").iloc[0].to_dict()
    R["confirmed"] = q("""
        select source, sum(qty) as qty, sum(case when shot_id is not null then qty else 0 end) as linked
        from fct_confirmed_defects group by 1 order by 2 desc""")
    R["codes"] = q("""select defect_code, source, sum(qty) as qty from fct_confirmed_defects group by 1, 2""")
    R["audits"] = q("""select count(distinct audit_id) as audits, count(*) as pieces, avg(linked_to_shot::int) as linked
                       from fct_audit_piece""").iloc[0].to_dict()
    R["spc_points"] = q("""select avg(we1::int) we1, avg(we2::int) we2, avg(we4::int) we4, avg(we5::int) we5, avg(mr_rule::int) mr,
                                  count(*) as points from spc_chart_points where is_monitoring""").iloc[0].to_dict()
    R["spc_alarms"] = q("select rule, count(*) as n from spc_alarms group by 1 order by 2 desc")
    R["drift"] = q("select detector, count(*) as n, count(distinct job_id) as jobs from drift_signals group by 1 order by 2 desc")
    R["lot_steps"] = q("""select l.*, r.supplier_id from lot_step_tests l
                          left join stg_materials__resin_lots r on r.lot_id = l.resin_lot_id""")
    R["jobs"] = q("select * from fct_job")
    R["parts"] = q("select * from stg_erp__part_attributes")
    R["events"] = q("select * from stg_toolroom__equipment_service")
    R["maint"] = q("select * from stg_toolroom__mold_maintenance")

    # quality engineering's register: mechanism shares on confirmed defects (reference, for the technical report)
    reg = pd.read_parquet(REF / "root_cause_register.parquet")
    R["register_share"] = reg[reg["caught_by"].notna()]["root_cause_code"].value_counts(normalize=True)
    R["register_n"] = int(reg["caught_by"].notna().sum())
    R["register_all"] = len(reg)
    R["register_caught"] = reg["caught_by"].fillna("escaped").value_counts()
    return R


def check(R, name_part):
    for c in R["checks"]:
        if name_part in c["name"]:
            return c
    raise KeyError(name_part)


WARM = ("2025-06-01", "2025-10-01")


def vm_overall(R, variant="full"):
    """Rolling-origin virtual metrology, medians over mold and press of the monthly per-cell metrics: overall, warm months
    (June to September 2025) and the other months."""
    v = R["vm_month"][R["vm_month"]["variant"] == variant].copy()
    v["warm"] = (v["month"] >= WARM[0]) & (v["month"] < WARM[1])
    cols = ["rmse_over_gauge", "r2", "calibration_slope"]
    out = {}
    for t, g in v.groupby("target"):
        out[t] = dict(all=g[cols].median().to_dict(), warm=g[g["warm"]][cols].median().to_dict(), other=g[~g["warm"]][cols].median().to_dict())
    return out


def vm_by_cell(R):
    """Per mold and press, pooled over the eleven evaluation months: error over gauge, R2 and calibration."""
    p = R["vm_pred"].copy()
    gauge = R["vm_month"][R["vm_month"]["variant"] == "full"].groupby(["target", "cell"])["gauge_sd"].first()
    rows = []
    for (t, cell), g in p.groupby(["target", "cell"]):
        err = g["measured"] - g["predicted"]
        ss = ((g["measured"] - g["measured"].mean()) ** 2).sum()
        rmse = float(np.sqrt((err ** 2).mean()))
        rows.append(dict(target=t, cell=cell, n=len(g), rmse=rmse, gauge_sd=float(gauge.get((t, cell), np.nan)),
                         rmse_over_gauge=rmse / float(gauge.get((t, cell), np.nan)), r2=float(1 - (err ** 2).sum() / ss),
                         calibration_slope=float(np.polyfit(g["predicted"], g["measured"], 1)[0])))
    return pd.DataFrame(rows)


def defect_table(R, tag_df=None):
    d = R["defect"] if tag_df is None else tag_df
    d = d[d["usable"] == True]
    g = d.groupby("target").agg(n_train_pos=("n_train_pos", "first"), n_test_pos=("n_test_pos", "first"),
                                recall_model=("recall_model", "mean"), recall_model_sd=("recall_model", "std"),
                                recall_template=("recall_template", "mean"), recall_spc=("recall_spc", "mean"),
                                ap_model=("ap_model", "mean"), ap_template=("ap_template", "mean"),
                                alarm_rate=("alarm_rate", "first")).reset_index()
    g["code"] = g["target"].str.replace("y_", "", regex=False).str.replace("_", " ")
    return g


def vm_shots(shot_ids):
    """Predicted weight (fraction of nominal) and dimension (tolerance units) per shot and cavity, from the rolling-origin
    model in force that month."""
    ids = pd.Index(shot_ids).unique()
    p = pd.read_parquet(RES / "rolling_vm_shot.parquet", filters=[("shot_id", "in", list(map(int, ids)))])
    return p


def anomaly_shots(shot_ids=None):
    """Anomaly score, raw flag and alarm state per monitored shot, from the rolling-origin model in force that month."""
    a = pd.read_parquet(RES / "rolling_anomaly.parquet")
    return a if shot_ids is None else a[a["shot_id"].isin(set(shot_ids))]
