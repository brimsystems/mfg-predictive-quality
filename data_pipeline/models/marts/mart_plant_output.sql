-- Plant-wide work orders and final inspection by press and month (the scrap-rate denominator).
select
    o.press_id, date_trunc('month', o.actual_start) as month,
    count(*) as work_orders,
    sum(i.quantity_inspected) as quantity_inspected,
    sum(i.quantity_failed) as quantity_failed
from {{ ref('stg_erp__production_orders') }} o
left join {{ ref('stg_qms__inspection_records') }} i using (work_order_id)
group by all
