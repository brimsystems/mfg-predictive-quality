-- One row per rule firing, after the baseline window, with the technician's disposition where
-- a change was logged within the hour that followed.
with p as (
    select * from {{ ref('spc_chart_points') }}
    where is_monitoring and (we1 or we2 or we4 or we5 or mr_rule)
),
fired as (
    select p.*,
           case when we1 then 'WE1' when we2 then 'WE2' when we4 then 'WE4' when we5 then 'WE5' else 'MR' end as rule,
           -- a new alarm only when the same chart did not fire on the previous point
           lag(point_no) over (partition by job_id, sensor_id, metric order by point_no) as prev_fired_point
    from p
),
alarms as (
    select * from fired where prev_fired_point is null or point_no - prev_fired_point > 1
),
changes as (
    select press_id, change_ts, change_id, parameter, reason_code from {{ ref('stg_qms__setpoint_changes') }}
    where old_value is distinct from new_value
)
select a.shot_id, a.sensor_id, a.mold_id, a.press_id, a.job_id, a.shot_ts, a.metric, a.rule, a.value, a.center, a.sigma, a.z,
       case when c.change_ts <= a.shot_ts + interval 1 hour then c.change_id end as disposition_change_id,
       case when c.change_ts <= a.shot_ts + interval 1 hour then c.parameter end as disposition_parameter,
       case when c.change_ts <= a.shot_ts + interval 1 hour then c.reason_code end as disposition_reason
from alarms a
asof left join changes c on a.press_id = c.press_id and a.shot_ts <= c.change_ts
