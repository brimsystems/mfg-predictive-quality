-- Audit pieces linked to their shot through the robot's recorded cycle and the tray row.
-- Pieces pulled by hand have no recorded cycle and stay unlinked.
select
    p.audit_id, p.tray_row, p.cavity_id, p.part_weight_g, p.dimension_1, p.dimension_2, p.dimension_3,
    p.visual_result, p.defect_code,
    a.job_id, a.press_id, a.mold_id, a.audit_ts, a.inspector_id, a.sampled_shot_cycle_no,
    j.shot_id
from {{ ref('stg_qms__qc_audit_pieces') }} p
join {{ ref('stg_qms__qc_audits') }} a using (audit_id)
left join {{ ref('int_shot_jobs') }} j
  on j.press_id = a.press_id and j.cycle_no = a.sampled_shot_cycle_no + p.tray_row
