-- Each sensor's summary values against the template in force at the shot, as a fraction of the
-- alarm band (0 on template, +/-1 at the alarm limit), and the unit's alarm logic recomputed.
with s as (
    select * from {{ ref('stg_monitoring__shot_summary') }}
),
t as (
    select * from {{ ref('stg_monitoring__templates') }}
),
j as (
    select s.*, t.template_id, t.match_score_threshold,
    {%- for v in ['fill_integral', 'pack_integral', 'peak_pressure_bar', 'gate_seal_time_s', 'end_of_fill_pressure_bar'] %}
        (s.{{ v }} - t.{{ v }}_template_value) / nullif(
            case when s.{{ v }} >= t.{{ v }}_template_value then t.{{ v }}_alarm_high - t.{{ v }}_template_value
                 else t.{{ v }}_template_value - t.{{ v }}_alarm_low end, 0) as {{ v }}_dev,
        s.{{ v }} < t.{{ v }}_alarm_low or s.{{ v }} > t.{{ v }}_alarm_high as {{ v }}_alarm,
        s.{{ v }} < t.{{ v }}_warning_low or s.{{ v }} > t.{{ v }}_warning_high as {{ v }}_warning,
    {%- endfor %}
        t.effective_from as template_from
    from s
    asof left join t
      on s.mold_id = t.mold_id and s.press_id = t.press_id and s.sensor_id = t.sensor_id and s.shot_ts >= t.effective_from
)
select
    *,
    case
        when sensor_dropout then 'none'
        when coalesce(fill_integral_alarm, false) or coalesce(pack_integral_alarm, false) or coalesce(peak_pressure_bar_alarm, false)
          or coalesce(gate_seal_time_s_alarm, false) or coalesce(end_of_fill_pressure_bar_alarm, false)
          or template_match_score < match_score_threshold then 'alarm'
        when coalesce(fill_integral_warning, false) or coalesce(pack_integral_warning, false) or coalesce(peak_pressure_bar_warning, false)
          or coalesce(gate_seal_time_s_warning, false) or coalesce(end_of_fill_pressure_bar_warning, false) then 'warning'
        else 'none'
    end as sensor_state_recomputed
from j
