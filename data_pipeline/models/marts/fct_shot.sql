-- One row per shot on the cell: context, normalized summary values, the unit's alarm and sort,
-- control-chart and drift states, and the labels that link to the shot.
{% set rel = [('s.pg_fill_time_to_sensor_s', 'pg_fill_time_to_sensor_rel'), ('s.pg_time_to_peak_s', 'pg_time_to_peak_rel'),
               ('s.pg_cooling_rate', 'pg_cooling_rate_rel'), ('s.pg_cycle_integral', 'pg_cycle_integral_rel'),
               ('c.fill_time_s', 'fill_time_rel'), ('c.recovery_time_s', 'recovery_time_rel'),
               ('c.peak_injection_pressure_bar', 'peak_injection_pressure_rel'),
               ('c.switchover_pressure_bar', 'switchover_pressure_rel'), ('c.cycle_time_s', 'cycle_time_rel')] %}
with job_ref as (
    -- values with no template band are taken relative to the job's first 500 production shots
    select c.job_id,
    {%- for src, dst in rel %}
           median({{ src }}) as {{ dst }}_ref{{ "," if not loop.last }}
    {%- endfor %}
    from {{ ref('int_shot_context') }} c join {{ ref('int_shot_summary') }} s using (shot_id)
    where c.after_approval and c.shots_since_approval < {{ var('spc_baseline_shots') }}
    group by 1
),
sort_rev as (
    -- only the indexed-tray reviews (medical molds) are tied to a shot
    select shot_id, pieces_reviewed, pieces_confirmed_defective, defect_code_list as sort_defect_codes, reviewed_ts
    from {{ ref('stg_qms__sort_dispositions') }}
    where review_mode = 'indexed_tray'
),
audit as (
    select shot_id,
           count(*) as audit_pieces,
           count(*) filter (where visual_result = 'reject') as audit_rejects,
           string_agg(defect_code, ';') filter (where defect_code is not null) as audit_defect_codes
    from {{ ref('int_audit_pieces_linked') }}
    where shot_id is not null
    group by 1
)
select
    c.shot_id, c.press_id, c.mold_id, c.job_id, c.shot_ts, c.cycle_no, c.unit_job_field_stale,
    c.after_approval, c.shots_since_approval, c.approval_ts,
    j.technician_id, j.tech_prior_runs_on_mold, j.tech_tenure_days, j.is_matured,
    pa.cavities,
    pa.cavities - case when c.shots_since_cavity_block is not null
                        and (c.shots_since_cavity_unblock is null or c.shots_since_cavity_block < c.shots_since_cavity_unblock)
                       then 1 else 0 end as active_cavities,
    -- material
    c.resin_lot_id, c.resin_id, c.supplier_id, c.cert_melt_flow_index, c.cert_mfi_band_position, c.cert_moisture_pct,
    c.regrind_pct, c.dryer_id, c.dryer_dew_point_c, c.lot_residence_h, c.residence_below_min, c.hours_since_lot_change,
    -- maintenance
    c.shots_since_vent_cleaning, c.shots_since_hot_runner_tip, c.shots_since_pm, c.shots_since_sensor_recalibration,
    c.days_since_ring_replace, c.days_since_tcu_service,
    -- setpoints against the sheet
    c.setpoint_delta_hold_pressure_bar, c.setpoint_delta_injection_velocity_mm_s, c.setpoint_delta_switchover_position_mm,
    c.setpoint_delta_melt_temp_c, c.setpoint_delta_mold_temp_c,
    -- machine side
    c.cycle_time_s, c.fill_time_s, c.switchover_position_mm, c.switchover_pressure_bar, c.peak_injection_pressure_bar,
    c.cushion_mm, c.hold_pressure_bar, c.recovery_time_s, c.back_pressure_bar, c.screw_rpm, c.nozzle_actual_c,
    c.mold_temp_a_c, c.mold_temp_b_c, c.clamp_tonnage, c.press_alarm_code,
    c.cushion_window_pos, c.hold_pressure_window_pos, c.melt_temp_window_pos, c.mold_temp_window_pos, c.mold_temp_half_diff_c,
    c.shot_month, c.season_sin, c.season_cos,
    -- cavity pressure
    s.template_id, s.n_sensors, s.n_sensor_dropouts,
    s.pg_fill_integral, s.pg_pack_integral, s.pg_cycle_integral, s.pg_peak_pressure_bar, s.pg_gate_seal_time_s,
    s.pg_fill_time_to_sensor_s, s.pg_time_to_peak_s, s.pg_pressure_at_transfer_bar, s.pg_cooling_rate,
    s.pg_fill_dev, s.pg_pack_dev, s.pg_peak_dev, s.pg_gate_seal_dev, s.pg_pack_dev_spread,
    s.eof_pressure_bar, s.eof_pack_integral, s.eof_fill_time_to_sensor_s, s.eof_pressure_dev, s.eof_pack_dev,
    s.match_score_min, s.max_abs_dev,
    {%- for src, dst in rel %}
    {{ src }} / nullif(jr.{{ dst }}_ref, 0) - 1 as {{ dst }},
    {%- endfor %}
    case s.unit_state_code when 2 then 'alarm' when 1 then 'warning' else 'none' end as unit_alarm_state,
    case s.recomputed_state_code when 2 then 'alarm' when 1 then 'warning' else 'none' end as recomputed_alarm_state,
    s.alarm_values, s.unit_sorted, s.curve_retained,
    -- control charts and drift
    coalesce(f.spc_we1, false) as spc_we1, coalesce(f.spc_we2, false) as spc_we2, coalesce(f.spc_we4, false) as spc_we4,
    coalesce(f.spc_we5, false) as spc_we5, coalesce(f.spc_mr, false) as spc_mr, coalesce(f.spc_any, false) as spc_any,
    f.spc_max_abs_z,
    d.ewma_pack, d.ewma_fill, d.cusum_eof, d.cusum_gate_seal, d.cusum_pack_var,
    coalesce(d.drift_any_signal, false) as drift_any_signal, coalesce(d.drift_any_fire, false) as drift_any_fire,
    -- labels linked to the shot
    r.pieces_reviewed, r.pieces_confirmed_defective, r.sort_defect_codes,
    a.audit_pieces, a.audit_rejects, a.audit_defect_codes
from {{ ref('int_shot_context') }} c
join {{ ref('int_shot_summary') }} s using (shot_id)
left join {{ ref('int_cell_jobs') }} j using (job_id)
left join {{ ref('stg_erp__part_attributes') }} pa on pa.mold_id = c.mold_id
left join {{ ref('spc_shot_flags') }} f using (shot_id)
left join {{ ref('drift_shot_state') }} d using (shot_id)
left join job_ref jr on jr.job_id = c.job_id
left join sort_rev r using (shot_id)
left join audit a using (shot_id)
