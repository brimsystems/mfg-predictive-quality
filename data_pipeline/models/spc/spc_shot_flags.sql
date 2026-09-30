-- Shot grain: which control-chart rules fired on the shot, any sensor, any value.
select
    shot_id,
    bool_or(is_monitoring and we1) as spc_we1,
    bool_or(is_monitoring and we2) as spc_we2,
    bool_or(is_monitoring and we4) as spc_we4,
    bool_or(is_monitoring and we5) as spc_we5,
    bool_or(is_monitoring and mr_rule) as spc_mr,
    bool_or(is_monitoring and (we1 or we2 or we4 or we5 or mr_rule)) as spc_any,
    max(abs(z)) as spc_max_abs_z
from {{ ref('spc_chart_points') }}
group by shot_id
