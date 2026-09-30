{% macro normalize_defect_code(col) %}
    case
        when {{ col }} is null or trim({{ col }}) = '' then 'other'
        when regexp_matches(lower({{ col }}), 'short|non.?fill') then 'short_shot'
        when regexp_matches(lower({{ col }}), 'flash|flsh') then 'flash'
        when regexp_matches(lower({{ col }}), 'sink') then 'sink'
        when regexp_matches(lower({{ col }}), 'void|bubble') then 'void'
        when regexp_matches(lower({{ col }}), 'splay|silver') then 'splay'
        when regexp_matches(lower({{ col }}), 'burn|diesel') then 'burn'
        when regexp_matches(lower({{ col }}), 'weld|knit') then 'weld_line'
        when regexp_matches(lower({{ col }}), 'warp') then 'warp'
        when regexp_matches(lower({{ col }}), '^dim|oot|out of tol') then 'dimensional'
        when regexp_matches(lower({{ col }}), 'speck|blk') then 'black_specks'
        when regexp_matches(lower({{ col }}), 'contam|foreign') then 'contamination'
        when regexp_matches(lower({{ col }}), 'gate|vestige') then 'gate_vestige'
        when regexp_matches(lower({{ col }}), '^(none|ok|pass)$') then 'none'
        else 'other'
    end
{% endmacro %}
