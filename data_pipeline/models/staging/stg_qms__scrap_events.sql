-- Plant-wide scrap entries; free-text codes normalized to the QMS code list.
select scrap_id, work_order_id, press_id, cast(scrap_date as date) as scrap_date,
       {{ normalize_defect_code('defect_code') }} as defect_code,
       quantity_scrapped, unit_cost, material_cost, labor_cost, total_scrap_cost
from {{ source('qms', 'scrap_events') }}
