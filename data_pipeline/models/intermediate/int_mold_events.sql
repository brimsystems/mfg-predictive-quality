-- Maintenance and equipment events on a mold or press, with the mold shot counter at the event.
with mold_counter as (
    select shot_id, mold_id, press_id, shot_ts,
           row_number() over (partition by mold_id order by shot_ts) as mold_shot_no,
           row_number() over (partition by press_id order by shot_ts) as press_shot_no
    from {{ ref('int_shot_jobs') }}
)
select m.mold_id, null as press_id, m.event_type, m.event_ts, coalesce(c.mold_shot_no, 0) as shot_no_at_event
from {{ ref('stg_toolroom__mold_maintenance') }} m
asof left join mold_counter c on m.mold_id = c.mold_id and m.event_ts >= c.shot_ts
union all
select null as mold_id, e.press_id, e.equipment || '_' || e.event_type as event_type, e.event_ts,
       coalesce(c.press_shot_no, 0) as shot_no_at_event
from {{ ref('stg_toolroom__equipment_service') }} e
asof left join (select press_id, shot_ts, press_shot_no from mold_counter) c
  on e.press_id = c.press_id and e.event_ts >= c.shot_ts
