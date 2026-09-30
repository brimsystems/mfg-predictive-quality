-- The unit's recorded alarm state must agree with the platform's recomputation from summary
-- values and templates on at least 99% of production shots.
select avg((unit_alarm_state = recomputed_alarm_state)::int) as agreement
from {{ ref('fct_shot') }}
where after_approval
having avg((unit_alarm_state = recomputed_alarm_state)::int) < 0.99
