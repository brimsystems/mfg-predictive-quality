select load_id, press_id, job_id, cast(load_ts as timestamp) as load_ts, resin_lot_id, colorant_lot_id,
       regrind_pct, dryer_id, operator_id
from {{ source('materials', 'material_loads') }}
