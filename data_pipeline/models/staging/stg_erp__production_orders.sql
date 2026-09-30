-- Work orders for all 24 presses. Part numbers typed from travelers are normalized to P-nnnnn.
select
    work_order_id, job_id,
    'P-' || regexp_replace(upper(part_number), '^(PART|PN|P)[- ]?', '') as part_number,
    mold_id, customer, press_id, quantity_ordered,
    cast(order_date as date) as order_date,
    cast(scheduled_start as timestamp) as scheduled_start,
    cast(actual_start as timestamp) as actual_start,
    resin, program
from {{ source('erp', 'production_orders') }}
