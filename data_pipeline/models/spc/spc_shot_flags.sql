-- Shot grain: which control-chart rules fired on the shot, on any sensor and value, after the baseline window.
-- The deployed charts plot AR(1) residuals; the textbook and baseline versions on raw values are kept for comparison.
select
    shot_id,
    bool_or(is_monitoring and we1) as spc_we1,
    bool_or(is_monitoring and we2) as spc_we2,
    bool_or(is_monitoring and we4) as spc_we4,
    bool_or(is_monitoring and we5) as spc_we5,
    bool_or(is_monitoring and mr_rule) as spc_mr,
    bool_or(is_monitoring and (we1 or we2 or we4 or we5 or mr_rule)) as spc_any,
    bool_or(is_monitoring and (we1_textbook or we2_textbook)) as spc_rule12_textbook,
    bool_or(is_monitoring and (we1_baseline or we2_baseline)) as spc_rule12_baseline,
    count(*) as charts,
    max(abs(z)) as spc_max_abs_z
from {{ ref('spc_chart_points') }}
group by shot_id
