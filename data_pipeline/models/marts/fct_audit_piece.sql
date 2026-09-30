-- Audit pieces with their shot's features where the robot recorded the cycle. This is the
-- virtual metrology training set.
select
    p.audit_id, p.tray_row, p.cavity_id, p.part_weight_g, p.dimension_1 as critical_dimension_mm,
    p.visual_result, p.defect_code, p.inspector_id, p.audit_ts,
    pa.nominal_weight_g, pa.critical_dimension_mm as nominal_dimension_mm, pa.dimension_tolerance_mm,
    pa.gauge_sd_weight_g, pa.gauge_sd_dimension_mm, pa.standard_cost,
    p.shot_id is not null as linked_to_shot,
    s.*
from {{ ref('int_audit_pieces_linked') }} p
left join {{ ref('stg_erp__part_attributes') }} pa on pa.mold_id = p.mold_id
left join {{ ref('fct_shot') }} s using (shot_id)
