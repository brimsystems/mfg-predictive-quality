"""
Layer comparison at the job-hour grain, and the alarm budget.

For each confirmed defect (sort review, audit or packing tally) in a job-hour, the
layer that alarmed first in time on the same job, within its look-back, is credited
(ties go to the earlier layer in this list):

  1. template alarms and sort: the unit alarmed on the job in the hour or the two before
  2. control-chart rules on every shot, at the deployed rule set (rules 1 and 2)
  3. drift detection: a signal onset (EWMA, CUSUM, lot step) in the eight hours before,
     since a drift signal stays open until its reset
  4. virtual metrology: a predicted dimension beyond 80% of tolerance (dimensional codes only)
  5. the anomaly model: two flags in ten consecutive shots
  6. none

An alarm is the first firing of a layer on a job in an hour; repeated firings inside the
hour are one investigation. The budget is what one technician per shift can investigate.
"""
import pandas as pd

from .features import connect

SHIFT_H = 8
LOOKBACK = {"template_sort": 2, "rules": 2, "drift": 8, "virtual_metrology": 2, "anomaly": 2}


def confirmed(test_start):
    con = connect()
    d = con.execute(f"""
        select source, job_id, press_id, mold_id, hour_ts, defect_code, qty
        from fct_confirmed_defects
        where source in ('sort', 'audit', 'tally') and hour_ts >= '{test_start}' and job_id is not null
    """).df()
    con.close()
    return d


def hours(frame, ts="shot_ts"):
    return set(zip(frame["job_id"], pd.to_datetime(frame[ts]).dt.floor("h")))


def alarms_per_shift(signal_hours, n_hours):
    return len(signal_hours) / max(n_hours / SHIFT_H, 1)


def compare(defects, signals):
    """signals: ordered list of (layer, set of (job_id, hour)). Each defect is credited to the layer whose alarm came
    first in time within its look-back; ties go to the earlier layer in the list."""
    credit = []
    for r in defects.itertuples():
        best, best_k = "none", -1
        for name, sig in signals:
            if name == "virtual_metrology" and r.defect_code != "dimensional":
                continue
            for k in range(LOOKBACK[name], -1, -1):          # earliest hour first
                if (r.job_id, r.hour_ts - pd.Timedelta(hours=k)) in sig:
                    if k > best_k:
                        best, best_k = name, k
                    break
        credit.append(best)
    d = defects.assign(layer=credit)
    share = d.groupby("layer")["qty"].sum() / d["qty"].sum()
    dim = d[d["defect_code"] == "dimensional"]
    dim_share = dim.groupby("layer")["qty"].sum() / max(dim["qty"].sum(), 1)
    return d, share, dim_share
