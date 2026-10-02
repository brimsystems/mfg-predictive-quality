"""
Linkage and detection figures for one generation run, saved under a tag so two runs can be
compared: linkage coverage, shot-linked confirmed defects by mold, supervised results (all
shot-linked labels and the audit-only re-test), shot-level and job-hour detection by layer,
and the not-detected share.

Usage: python -m ml.src.linkage_compare <tag>
"""
import json
import sys

import pandas as pd

from . import defect
from .features import DATA_DIR, VALID_END, connect, load_shots
from .run_all import VM_FLAG, vm_shot_dimension, window_signals

RESULTS = DATA_DIR / "results"


def main(tag):
    con = connect()
    link = con.execute("""
        select source, mold_id, sum(qty) as qty, sum(case when shot_id is not null then qty else 0 end) as linked
        from fct_confirmed_defects where source in ('sort', 'audit', 'tally', 'return') group by 1, 2""").df()
    con.close()
    out = {"linkage_coverage": float(link["linked"].sum() / link["qty"].sum())}
    out["linked_by_mold"] = link.groupby("mold_id")["linked"].sum().astype(int).to_dict()
    out["linked_by_mold_source"] = link.pivot_table(index="mold_id", columns="source", values="linked", aggfunc="sum",
                                                    fill_value=0).astype(int).to_dict(orient="index")

    d = pd.read_csv(RESULTS / "defect_metrics_full.csv")
    d = d[d["target"] == "y_any"]
    out["supervised_all"] = d[["n_train_pos", "n_test_pos", "recall_model", "recall_template", "ap_model", "ap_template"]].mean().to_dict()
    X = json.loads((RESULTS / "overview_extras.json").read_text())
    out["supervised_audit_only"] = X["audit_only"]

    # shot-level detection by layer on every shot-linked confirmed defect (sort review and audit), test period
    df = load_shots()
    an = pd.read_parquet(RESULTS / "anomaly_if_scores.parquet")
    an = an[an["seed"] == an["seed"].min()]
    end = pd.Timestamp("2026-04-01")
    te = df[(df["shot_ts"] >= VALID_END) & (df["shot_ts"] < end) & df["after_approval"]]
    te, sig = window_signals(df, an, VALID_END, end, vm_shot_dimension(te))
    con = connect()
    lk = con.execute(f"""select source, shot_id, mold_id, defect_code, qty from fct_confirmed_defects
                         where source in ('sort', 'audit') and shot_id is not null
                         and hour_ts >= '{VALID_END.date()}' and hour_ts < '{end.date()}'""").df()
    con.close()
    lk = lk.merge(te[["shot_id", "unit_alarm_state", "spc_we1", "spc_we2", "drift_any_signal", "vm_dim"]], on="shot_id")
    an_ids = set(sig["anomaly"]["shot_id"])
    flags = {"template_sort": lk["unit_alarm_state"] == "alarm", "rules": lk["spc_we1"] | lk["spc_we2"],
             "drift": lk["drift_any_signal"].astype(bool), "anomaly": lk["shot_id"].isin(an_ids),
             "virtual_metrology": (lk["vm_dim"] > VM_FLAG) & (lk["defect_code"] == "dimensional")}
    any_flag = pd.concat(flags, axis=1).any(axis=1)
    out["shot_level"] = {k: float((v * lk["qty"]).sum() / lk["qty"].sum()) for k, v in flags.items()}
    out["shot_level"]["any_layer"] = float((any_flag * lk["qty"]).sum() / lk["qty"].sum())
    out["shot_level_n"] = int(lk["qty"].sum())
    out["shot_level_by_source_n"] = lk.groupby("source")["qty"].sum().astype(int).to_dict()
    out["shot_level_medical_share"] = float(lk.loc[lk["mold_id"].isin(["M-2118", "M-2119"]), "qty"].sum() / lk["qty"].sum())

    L = json.loads((RESULTS / "layers.json").read_text())["layers"]
    out["job_hour"] = L["share"]
    out["not_detected"] = L["share"].get("none", 0.0)
    out["alarms_per_shift"] = L["alarms_per_shift"]
    (RESULTS / f"linkage_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "current")
