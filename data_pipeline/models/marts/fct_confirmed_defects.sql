-- Every confirmed defect with where it was found: the sort review and audits link to a shot,
-- packing tallies to a job and hour (cavity on most), returns to a job on most.
with sort_codes as (
    select r.shot_id, unnest(string_split(r.defect_codes, ';')) as defect_code, r.pieces_confirmed_defective,
           len(string_split(r.defect_codes, ';')) as n_codes
    from {{ ref('stg_qms__sort_dispositions') }} r where r.pieces_confirmed_defective > 0
)
select 'sort' as source, s.job_id, s.press_id, s.mold_id, s.shot_id, date_trunc('hour', s.shot_ts) as hour_ts,
       null::integer as cavity_id, sc.defect_code, ceil(sc.pieces_confirmed_defective / sc.n_codes)::integer as qty
from sort_codes sc join {{ ref('int_shot_context') }} s using (shot_id)
union all
select 'audit', p.job_id, p.press_id, p.mold_id, p.shot_id, date_trunc('hour', coalesce(s.shot_ts, p.audit_ts)),
       p.cavity_id, p.defect_code, 1
from {{ ref('int_audit_pieces_linked') }} p left join {{ ref('int_shot_context') }} s using (shot_id)
where p.defect_code is not null
union all
select 'tally', t.job_id, t.press_id, j.mold_id, null, t.tally_hour, t.cavity_id, t.defect_code, t.qty
from {{ ref('stg_qms__scrap_tallies') }} t left join {{ ref('int_cell_jobs') }} j using (job_id)
union all
select 'return', r.attributed_job_id, j.press_id, j.mold_id, null, null, r.attributed_cavity_id, r.defect_code, r.qty
from {{ ref('stg_qms__customer_returns') }} r left join {{ ref('int_cell_jobs') }} j on j.job_id = r.attributed_job_id
