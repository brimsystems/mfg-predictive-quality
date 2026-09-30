-- Plant-wide scrap by press, mold, code and month, from the job-level QMS scrap entries.
select
    o.press_id, o.mold_id, o.part_number, o.customer, o.program,
    date_trunc('month', e.scrap_date) as month,
    e.defect_code,
    o.press_id in ('IM-11', 'IM-12') as on_cell,
    count(distinct e.work_order_id) as work_orders,
    sum(e.quantity_scrapped) as quantity_scrapped,
    sum(e.total_scrap_cost) as scrap_cost
from {{ ref('stg_qms__scrap_events') }} e
join {{ ref('stg_erp__production_orders') }} o using (work_order_id)
group by all
