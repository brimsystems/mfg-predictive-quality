-- One row per instrumented job: run window, approval, alarms and sort, confirmed defects, maturity.
with s as (
    select job_id, count(*) as shots, sum(active_cavities) filter (where after_approval) as pieces,
           count(*) filter (where after_approval and unit_sorted) as sorted_shots,
           count(*) filter (where after_approval and unit_alarm_state = 'alarm') as alarm_shots,
           avg((unit_alarm_state = recomputed_alarm_state)::int) filter (where after_approval) as unit_state_agreement,
           avg(unit_job_field_stale::int) as unit_job_field_stale_share
    from {{ ref('fct_shot') }} group by 1
),
d as (
    select job_id,
           sum(qty) filter (where source = 'sort') as confirmed_sort,
           sum(qty) filter (where source = 'audit') as confirmed_audit,
           sum(qty) filter (where source = 'tally') as confirmed_tally,
           sum(qty) filter (where source = 'return') as returned
    from {{ ref('fct_confirmed_defects') }} group by 1
)
select j.*, s.shots, s.pieces, s.sorted_shots, s.alarm_shots, s.unit_state_agreement, s.unit_job_field_stale_share,
       coalesce(d.confirmed_sort, 0) as confirmed_sort, coalesce(d.confirmed_audit, 0) as confirmed_audit,
       coalesce(d.confirmed_tally, 0) as confirmed_tally, coalesce(d.returned, 0) as returned
from {{ ref('int_cell_jobs') }} j
left join s using (job_id)
left join d using (job_id)
