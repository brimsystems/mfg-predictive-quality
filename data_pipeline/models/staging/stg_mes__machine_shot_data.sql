select
    shot_id, press_id, job_id as mes_job_id, cast(shot_ts as timestamp) as shot_ts,
    cycle_time_s, fill_time_s, switchover_position_mm, switchover_pressure_bar, peak_injection_pressure_bar,
    cushion_mm, hold_pressure_bar, hold_time_s, recovery_time_s, back_pressure_bar, screw_rpm,
    barrel_zone_1_actual_c, barrel_zone_2_actual_c, barrel_zone_3_actual_c, barrel_zone_4_actual_c,
    nozzle_actual_c, mold_temp_a_c, mold_temp_b_c, clamp_tonnage, press_alarm_code
from {{ source('mes', 'machine_shot_data') }}
