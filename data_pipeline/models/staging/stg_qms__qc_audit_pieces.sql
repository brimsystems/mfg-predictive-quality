select
    audit_id, tray_row, cavity_id, part_weight_g, dimension_1, dimension_2, dimension_3,
    visual_result, defect_codes as defect_code
from {{ source('qms', 'qc_audit_pieces') }}
