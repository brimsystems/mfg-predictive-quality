select inspection_id, work_order_id, cast(inspection_date as timestamp) as inspection_date, inspector_id,
       quantity_inspected, quantity_passed, quantity_failed,
       {{ normalize_defect_code('defect_code_raw') }} as defect_code, disposition
from {{ source('qms', 'inspection_records') }}
