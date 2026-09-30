select return_id, customer_id, part_id, cast(date_code as varchar) as date_code, defect_code, qty,
       cast(received_date as date) as received_date, attributed_job_id,
       cast(attributed_cavity_id as integer) as attributed_cavity_id
from {{ source('qms', 'customer_returns') }}
