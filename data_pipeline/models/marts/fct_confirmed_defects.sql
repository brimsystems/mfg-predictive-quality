-- Every confirmed defect with where it was found. Indexed-tray reviews (medical molds) and robot-linked audits
-- carry a shot; per-shift bin reviews carry a job and shift, and their pieces are spread over the shift's hours
-- in proportion to the shots sorted in each hour; packing tallies carry a job and hour; returns a job on most.
with sort_codes as (
    select r.review_mode, r.shot_id, r.job_id, r.press_id, r.mold_id, r.shift_date, r.shift,
           split_part(c.code, ':', 1) as defect_code, cast(split_part(c.code, ':', 2) as double) as qty
    from {{ ref('stg_qms__sort_dispositions') }} r, unnest(string_split(r.defect_codes, ';')) as c(code)
    where r.pieces_confirmed_defective > 0
),
sorted_hours as (
    select job_id, cast(date_trunc('day', shot_ts - interval 6 hour) as date) as shift_date,
           case when hour(shot_ts) >= 6 and hour(shot_ts) < 14 then 'A' when hour(shot_ts) >= 14 and hour(shot_ts) < 22 then 'B' else 'C' end as shift,
           date_trunc('hour', shot_ts) as hour_ts, count(*) as n
    from {{ ref('fct_shot') }} where unit_sorted and after_approval
    group by all
),
hour_share as (
    select *, n / sum(n) over (partition by job_id, shift_date, shift) as share from sorted_hours
)
select 'sort' as source, sc.job_id, sc.press_id, sc.mold_id, sc.shot_id, date_trunc('hour', s.shot_ts) as hour_ts,
       null::integer as cavity_id, sc.defect_code, sc.qty
from sort_codes sc join {{ ref('int_shot_context') }} s using (shot_id)
where sc.review_mode = 'indexed_tray'
union all
select 'sort', sc.job_id, sc.press_id, sc.mold_id, null, h.hour_ts, null::integer, sc.defect_code, sc.qty * h.share
from sort_codes sc join hour_share h using (job_id, shift_date, shift)
where sc.review_mode = 'per_shift'
union all
select 'audit', p.job_id, p.press_id, p.mold_id, p.shot_id, date_trunc('hour', coalesce(s.shot_ts, p.audit_ts)),
       p.cavity_id, p.defect_code, 1.0
from {{ ref('int_audit_pieces_linked') }} p left join {{ ref('int_shot_context') }} s using (shot_id)
where p.defect_code is not null
union all
select 'tally', t.job_id, t.press_id, j.mold_id, null, t.tally_hour, t.cavity_id, t.defect_code, t.qty
from {{ ref('stg_qms__scrap_tallies') }} t left join {{ ref('int_cell_jobs') }} j using (job_id)
union all
select 'return', r.attributed_job_id, j.press_id, j.mold_id, null, null, r.attributed_cavity_id, r.defect_code, r.qty
from {{ ref('stg_qms__customer_returns') }} r left join {{ ref('int_cell_jobs') }} j on j.job_id = r.attributed_job_id
