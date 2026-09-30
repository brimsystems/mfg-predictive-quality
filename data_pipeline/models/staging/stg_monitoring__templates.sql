-- One row per template version per sensor, with the bands for each monitored summary value.
select
    template_id, mold_id, press_id, sensor_id,
    cast(effective_from as timestamp) as effective_from,
    established_by, approved_by, source_run_job_id,
    template_curve_ref, match_score_threshold, band_widened, reason as template_reason,
    fill_integral_template_value, fill_integral_warning_low, fill_integral_warning_high, fill_integral_alarm_low, fill_integral_alarm_high,
    pack_integral_template_value, pack_integral_warning_low, pack_integral_warning_high, pack_integral_alarm_low, pack_integral_alarm_high,
    peak_pressure_bar_template_value, peak_pressure_bar_warning_low, peak_pressure_bar_warning_high, peak_pressure_bar_alarm_low, peak_pressure_bar_alarm_high,
    gate_seal_time_s_template_value, gate_seal_time_s_warning_low, gate_seal_time_s_warning_high, gate_seal_time_s_alarm_low, gate_seal_time_s_alarm_high,
    end_of_fill_pressure_bar_template_value, end_of_fill_pressure_bar_warning_low, end_of_fill_pressure_bar_warning_high, end_of_fill_pressure_bar_alarm_low, end_of_fill_pressure_bar_alarm_high
from {{ source('monitoring', 'templates') }}
