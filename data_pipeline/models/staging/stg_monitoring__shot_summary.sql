-- One row per shot per sensor, as the monitoring unit recorded it.
select
    shot_id, press_id, unit_id, mold_id, sensor_id, sensor_position, cavity_no,
    job_id as unit_job_id,
    cast(shot_ts as timestamp) as shot_ts,
    cycle_no,
    fill_time_to_sensor_s, peak_pressure_bar, time_to_peak_s, pressure_at_transfer_bar,
    fill_integral, pack_integral, cycle_integral, end_of_fill_pressure_bar, gate_seal_time_s, cooling_rate,
    template_match_score, alarm_state, alarm_values, sort_signal, curve_retained,
    pack_integral is null as sensor_dropout
from {{ source('monitoring', 'cavity_shot_summary') }}
