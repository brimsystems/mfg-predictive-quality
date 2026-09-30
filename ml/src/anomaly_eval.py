"""
Anomaly evaluation.

  novel signatures   each event window from quality engineering's event log (reference,
                     outside the load path): detected or not, and lead time from the first
                     anomaly flag to the first confirmed defect or technician change
  check ring onset   the anomaly flag rate on IM-12 by month before the ring replacement
  false positives    flag rate on validated test shots
  overlap            share of anomaly flags on shots the template passed
"""
import numpy as np
import pandas as pd

from .anomaly import validated
from .features import ROOT, VALID_END, connect

REF = ROOT / "data_source" / "reference"


def events():
    runs = pd.read_parquet(REF / "run_log.parquet")
    st = pd.read_parquet(REF / "shot_state", columns=["shot_id", "run_no", "g10"])
    g = st[st["g10"]].groupby("run_no")["shot_id"].agg(["min", "max"]).reset_index()
    return g.merge(runs[["run_no", "job_id", "mold_id", "press_id", "g10_kind"]], on="run_no")


def evaluate(df, flags, ae=None):
    """df: fct_shot rows (all periods); flags: Series of bool indexed like df (isolation forest flags)."""
    d = df[["shot_id", "job_id", "press_id", "mold_id", "shot_ts", "after_approval", "unit_alarm_state", "split",
            "shots_since_approval", "pieces_confirmed_defective", "audit_rejects"]].copy()
    d["flag"] = flags.values
    ev = events()
    con = connect()
    # the shop's first response: a sorted shot reviewed, a tally entered, or a setpoint change
    conf = con.execute("""
        select s.job_id, r.reviewed_ts as ts from stg_qms__sort_dispositions r join fct_shot s using (shot_id)
        where r.pieces_confirmed_defective > 0
        union all
        select job_id, entered_ts from stg_qms__scrap_tallies""").df()
    ch = con.execute("select job_id, change_ts from stg_qms__setpoint_changes where old_value is distinct from new_value").df()
    con.close()
    rows = []
    for e in ev.itertuples():
        w = d[(d["shot_id"] >= e.min) & (d["shot_id"] <= e.max)].sort_values("shot_ts")
        if w.empty:
            continue
        start = w["shot_ts"].iloc[0]
        f1 = w.loc[w["flag"], "shot_ts"].min()
        a1 = w.loc[w["unit_alarm_state"] == "alarm", "shot_ts"].min()
        c1 = conf[(conf["job_id"] == e.job_id) & (conf["ts"] >= start)]["ts"].min()
        t1 = ch[(ch["job_id"] == e.job_id) & (ch["change_ts"] >= start)]["change_ts"].min()
        response = min([x for x in (c1, t1) if pd.notna(x)], default=pd.NaT)
        rows.append(dict(job_id=e.job_id, kind=e.g10_kind, mold_id=e.mold_id, press_id=e.press_id, start=start,
                         shots=len(w), flag_rate=float(w["flag"].mean()), detected=bool(pd.notna(f1)),
                         template_alarm_rate=float((w["unit_alarm_state"] == "alarm").mean()),
                         minutes_to_flag=(f1 - start).total_seconds() / 60 if pd.notna(f1) else np.nan,
                         minutes_to_template_alarm=(a1 - start).total_seconds() / 60 if pd.notna(a1) else np.nan,
                         lead_over_response_min=(response - f1).total_seconds() / 60 if pd.notna(f1) and pd.notna(response) else np.nan,
                         split=w["split"].iloc[0]))
    evr = pd.DataFrame(rows)
    te = d[(d["shot_ts"] >= VALID_END) & d["after_approval"]]
    val = te[validated(df.loc[te.index])]
    im12 = d[(d["press_id"] == "IM-12") & d["after_approval"]].copy()
    onset = im12.groupby(im12["shot_ts"].dt.to_period("M"))["flag"].mean()
    fl = te[te["flag"]]
    summary = dict(events=len(evr), detected=int(evr["detected"].sum()) if len(evr) else 0,
                   median_minutes_to_flag=float(evr["minutes_to_flag"].median()) if len(evr) else np.nan,
                   median_lead_over_response_min=float(evr["lead_over_response_min"].median()) if len(evr) else np.nan,
                   false_positive_rate=float(val["flag"].mean()), test_flag_rate=float(te["flag"].mean()),
                   share_on_template_pass=float((fl["unit_alarm_state"] != "alarm").mean()) if len(fl) else np.nan)
    if ae is not None and len(ae):
        a = ae.merge(d[["shot_id", "shot_ts", "after_approval", "unit_alarm_state"]], on="shot_id")
        ev_ids = set()
        for e in ev.itertuples():
            ev_ids |= set(range(int(e.min), int(e.max) + 1))
        a["in_event"] = a["shot_id"].isin(ev_ids)
        at = a[a["shot_ts"] >= VALID_END]
        summary.update(ae_flag_rate_test=float(at["ae_flag"].mean()),
                       ae_event_flag_rate=float(a.loc[a["in_event"], "ae_flag"].mean()) if a["in_event"].any() else np.nan,
                       ae_nonevent_flag_rate=float(a.loc[~a["in_event"], "ae_flag"].mean()))
        ifr = d.merge(a[["shot_id", "in_event"]], on="shot_id")
        summary.update(if_event_flag_rate_retained=float(ifr.loc[ifr["in_event"], "flag"].mean()) if ifr["in_event"].any() else np.nan)
    return evr, onset, summary
