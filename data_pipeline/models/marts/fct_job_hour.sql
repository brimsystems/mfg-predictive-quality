-- Job x hour: shots, what each detection layer raised, and the confirmed defects the shop
-- recorded for that hour (sort, audit and packing tally).
with s as (
    select job_id, press_id, mold_id, date_trunc('hour', shot_ts) as hour_ts,
           count(*) as shots,
           sum(active_cavities) as pieces,
           count(*) filter (where after_approval) as production_shots,
           count(*) filter (where unit_alarm_state = 'alarm') as unit_alarm_shots,
           count(*) filter (where unit_sorted and after_approval) as sorted_shots,
           count(*) filter (where spc_any) as spc_rule_shots,
           count(*) filter (where drift_any_signal) as drift_signal_shots,
           avg(pg_pack_dev) as mean_pack_dev, avg(pg_fill_dev) as mean_fill_dev, avg(eof_pressure_dev) as mean_eof_dev
    from {{ ref('fct_shot') }}
    group by 1, 2, 3, 4
),
d as (
    select job_id, hour_ts,
           sum(qty) filter (where source = 'sort') as confirmed_sort,
           sum(qty) filter (where source = 'audit') as confirmed_audit,
           sum(qty) filter (where source = 'tally') as confirmed_tally,
           sum(qty) filter (where source in ('sort', 'audit', 'tally')) as confirmed_total
    from {{ ref('fct_confirmed_defects') }}
    where hour_ts is not null
    group by 1, 2
),
alarms as (
    select job_id, date_trunc('hour', shot_ts) as hour_ts, count(*) as spc_alarms
    from {{ ref('spc_alarms') }} group by 1, 2
),
drift as (
    select job_id, date_trunc('hour', shot_ts) as hour_ts, count(*) as drift_signals
    from {{ ref('drift_signals') }} group by 1, 2
)
select s.*,
       coalesce(a.spc_alarms, 0) as spc_alarms, coalesce(dr.drift_signals, 0) as drift_signals,
       coalesce(d.confirmed_sort, 0) as confirmed_sort, coalesce(d.confirmed_audit, 0) as confirmed_audit,
       coalesce(d.confirmed_tally, 0) as confirmed_tally, coalesce(d.confirmed_total, 0) as confirmed_total
from s
left join d using (job_id, hour_ts)
left join alarms a using (job_id, hour_ts)
left join drift dr using (job_id, hour_ts)
