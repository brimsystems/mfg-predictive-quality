-- Shot grain: post-gate and end-of-fill values (mean over the sensed cavities), their template
-- deviations, and the unit's recorded and recomputed alarm state.
with n as (
    select * from {{ ref('int_sensor_shot_normalized') }}
)
select
    shot_id,
    any_value(template_id) as template_id,
    count(*) as n_sensors,
    sum(sensor_dropout::int) as n_sensor_dropouts,
    -- post-gate
    avg(case when sensor_position = 'post_gate' then fill_integral end) as pg_fill_integral,
    avg(case when sensor_position = 'post_gate' then pack_integral end) as pg_pack_integral,
    avg(case when sensor_position = 'post_gate' then cycle_integral end) as pg_cycle_integral,
    avg(case when sensor_position = 'post_gate' then peak_pressure_bar end) as pg_peak_pressure_bar,
    avg(case when sensor_position = 'post_gate' then gate_seal_time_s end) as pg_gate_seal_time_s,
    avg(case when sensor_position = 'post_gate' then fill_time_to_sensor_s end) as pg_fill_time_to_sensor_s,
    avg(case when sensor_position = 'post_gate' then time_to_peak_s end) as pg_time_to_peak_s,
    avg(case when sensor_position = 'post_gate' then pressure_at_transfer_bar end) as pg_pressure_at_transfer_bar,
    avg(case when sensor_position = 'post_gate' then cooling_rate end) as pg_cooling_rate,
    avg(case when sensor_position = 'post_gate' then fill_integral_dev end) as pg_fill_dev,
    avg(case when sensor_position = 'post_gate' then pack_integral_dev end) as pg_pack_dev,
    avg(case when sensor_position = 'post_gate' then peak_pressure_bar_dev end) as pg_peak_dev,
    avg(case when sensor_position = 'post_gate' then gate_seal_time_s_dev end) as pg_gate_seal_dev,
    -- end of fill
    avg(case when sensor_position = 'end_of_fill' then end_of_fill_pressure_bar end) as eof_pressure_bar,
    avg(case when sensor_position = 'end_of_fill' then pack_integral end) as eof_pack_integral,
    avg(case when sensor_position = 'end_of_fill' then fill_time_to_sensor_s end) as eof_fill_time_to_sensor_s,
    avg(case when sensor_position = 'end_of_fill' then end_of_fill_pressure_bar_dev end) as eof_pressure_dev,
    avg(case when sensor_position = 'end_of_fill' then pack_integral_dev end) as eof_pack_dev,
    -- cavity spread between sensed cavities (post-gate pack integral)
    max(case when sensor_position = 'post_gate' then pack_integral_dev end)
      - min(case when sensor_position = 'post_gate' then pack_integral_dev end) as pg_pack_dev_spread,
    min(template_match_score) as match_score_min,
    max(greatest(abs(coalesce(fill_integral_dev, 0)), abs(coalesce(pack_integral_dev, 0)), abs(coalesce(peak_pressure_bar_dev, 0)),
                 abs(coalesce(gate_seal_time_s_dev, 0)), abs(coalesce(end_of_fill_pressure_bar_dev, 0)))) as max_abs_dev,
    -- the unit's recorded state and the platform's recomputation
    max(case alarm_state when 'alarm' then 2 when 'warning' then 1 else 0 end) as unit_state_code,
    max(case sensor_state_recomputed when 'alarm' then 2 when 'warning' then 1 else 0 end) as recomputed_state_code,
    string_agg(distinct alarm_values, ';') as alarm_values,
    bool_or(sort_signal = 'reject') as unit_sorted,
    bool_or(curve_retained) as curve_retained
from n
group by shot_id
