-- Audit pieces link to a shot on 80-95% of pieces (the robot records the cycle on most audits).
select avg(linked_to_shot::int) as linkage
from {{ ref('fct_audit_piece') }}
having avg(linked_to_shot::int) not between 0.80 and 0.95
