-- Instrumented jobs on the cell: the ERP order, the first-shot approval and the run window.
with orders as (
    select o.*
    from {{ ref('stg_erp__production_orders') }} o
    where o.press_id in ('IM-11', 'IM-12')
      and o.mold_id in (select mold_id from {{ ref('stg_erp__part_attributes') }})
),
shots as (
    select a.job_id, min(s.shot_ts) as first_shot_ts, max(s.shot_ts) as last_shot_ts
    from {{ ref('int_shot_jobs') }} s join orders a on s.job_id = a.job_id
    group by 1
)
select
    o.job_id, o.press_id, o.mold_id, o.part_number, o.customer, o.resin, o.program, o.quantity_ordered,
    o.actual_start as set_ts,
    f.approval_ts, f.technician_id, f.inspector_id, f.shots_to_approval, f.template_id_established,
    s.first_shot_ts, s.last_shot_ts,
    s.last_shot_ts + to_days({{ var('maturity_days') }}) as matures_on,
    s.last_shot_ts + to_days({{ var('maturity_days') }}) <= cast('{{ var("snapshot_date") }}' as timestamp) as is_matured,
    -- the approving technician's prior approvals on this mold (experience on the mold)
    count(*) over (partition by o.mold_id, f.technician_id order by f.approval_ts
                   rows between unbounded preceding and 1 preceding) as tech_prior_runs_on_mold,
    datediff('day', t.hire_date, f.approval_ts) as tech_tenure_days
from orders o
left join {{ ref('stg_qms__first_shot_approvals') }} f using (job_id)
left join shots s using (job_id)
left join {{ ref('stg_hr__technicians') }} t on t.technician_id = f.technician_id
