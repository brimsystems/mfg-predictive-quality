select
    job_id, press_id, mold_id, cast(approval_ts as timestamp) as approval_ts, technician_id, inspector_id,
    cast(shots_to_approval as integer) as shots_to_approval, part_weight_g, dimension_values, visual_result,
    approved = 'True' as approved, template_id_established
from {{ source('qms', 'first_shot_approvals') }}
