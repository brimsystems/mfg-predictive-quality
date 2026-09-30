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
    R["layers"] = json.loads((RES / "layers.json").read_text())["layers"]
    R["anomaly"] = json.loads((RES / "anomaly_summary.json").read_text())
    R["anomaly_events"] = pd.read_csv(RES / "anomaly_events.csv", parse_dates=["start"])
    R["anomaly_im12"] = pd.read_csv(RES / "anomaly_im12_by_month.csv")
    R["vm"] = pd.read_csv(RES / "vm_metrics_full.csv")
    R["vm_abl"] = pd.read_csv(RES / "vm_ablations.csv")
    R["vm_pred"] = pd.read_parquet(RES / "vm_test_predictions_full.parquet")
    R["defect"] = pd.read_csv(RES / "defect_metrics_full.csv")
    R["defect_abl"] = pd.read_csv(RES / "defect_ablations.csv")

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
        select count(*) as reviewed, avg((pieces_confirmed_defective = 0)::int) as false_reject_share,
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


def vm_summary(R, design="pooled"):
    v = R["vm"][R["vm"]["design"] == design]
    return v.groupby(["target", "cell"])[["rmse", "gauge_sd", "rmse_over_gauge", "r2", "calibration_slope", "residual_lag1"]].mean().reset_index()


def vm_headline(R):
    out = {}
    for design in ("pooled", "per_cell"):
        v = R["vm"][R["vm"]["design"] == design]
        per_seed = v.groupby(["target", "seed"])[["rmse_over_gauge", "r2"]].median()
        for t in ("weight", "dimension"):
            ps = per_seed.loc[t]
            out[(design, t)] = dict(ratio=float(ps["rmse_over_gauge"].mean()), ratio_sd=float(ps["rmse_over_gauge"].std()),
                                    r2=float(ps["r2"].mean()), r2_sd=float(ps["r2"].std()))
    return out


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


def layer_table(R):
    L = R["layers"]["share"]
    order = [("template_sort", "Template alarms and sort"), ("rules", "Control-chart rules on every shot"),
             ("drift", "Drift detection"), ("virtual_metrology", "Virtual metrology (dimensional codes)"),
             ("anomaly", "Anomaly model"), ("none", "Caught by none")]
    rows = []
    for k, label in order:
        rows.append(dict(layer=label, share=L.get(k, 0.0),
                         alarms_per_shift=R["layers"]["alarms_per_shift"].get(k, np.nan)))
    return pd.DataFrame(rows)


def linkage(R):
    c = R["confirmed"]
    c = c[c["source"].isin(["sort", "audit", "tally", "return"])]
    return float(c["linked"].sum() / c["qty"].sum())
