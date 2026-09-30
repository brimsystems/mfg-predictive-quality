-- Every shot on the cell aligned to its MES job. The unit's job field is set at mold set and
-- lags on some shots after a changeover, so the job comes from the ERP order for this mold on
-- this press that started most recently before the shot.
with shots as (
    select shot_id, any_value(press_id) as press_id, any_value(mold_id) as mold_id, any_value(unit_job_id) as unit_job_id,
           any_value(shot_ts) as shot_ts, any_value(cycle_no) as cycle_no
    from {{ ref('stg_monitoring__shot_summary') }}
    group by shot_id
),
orders as (
    select job_id, press_id, mold_id, actual_start
    from {{ ref('stg_erp__production_orders') }}
    where press_id in ('IM-11', 'IM-12')
)
select s.shot_id, s.press_id, s.mold_id, s.shot_ts, s.cycle_no, s.unit_job_id, o.job_id,
       s.unit_job_id <> o.job_id as unit_job_field_stale
from shots s
asof left join orders o
  on s.press_id = o.press_id and s.mold_id = o.mold_id and s.shot_ts >= o.actual_start
