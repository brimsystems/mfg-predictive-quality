select lot_id, resin_id, grade, supplier_id, cast(received_date as date) as received_date,
       cert_melt_flow_index, cert_moisture_pct, cert_density, qty_kg
from {{ source('materials', 'resin_lots') }}
