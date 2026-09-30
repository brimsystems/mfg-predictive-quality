-- As-of context for every shot: load, lot and dryer; shots since each maintenance event;
-- the setpoints in force against the process sheet; the machine side against the process window.
-- Every join takes the latest record at or before the shot.
{% set mold_events = ['vent_cleaning', 'hot_runner_tip', 'PM', 'cavity_block', 'cavity_unblock', 'sensor_recalibration'] %}
{% set params = ['hold_pressure_bar', 'injection_velocity_mm_s', 'switchover_position_mm', 'melt_temp_c', 'mold_temp_c'] %}
with shots as (
    select j.*,
           row_number() over (partition by j.mold_id order by j.shot_ts) as mold_shot_no,
           row_number() over (partition by j.press_id order by j.shot_ts) as press_shot_no
    from {{ ref('int_shot_jobs') }} j
),
loads as (
    select l.*, r.resin_id, r.supplier_id, r.cert_melt_flow_index, r.cert_moisture_pct,
           (r.cert_melt_flow_index - g.mfi_band_low) / (g.mfi_band_high - g.mfi_band_low) as cert_mfi_band_position,
           g.hygroscopic, g.min_residence_h,
           -- the first load of this lot on this press starts its time in the dryer
           min(l.load_ts) over (partition by l.press_id, l.resin_lot_id) as lot_first_load_ts,
           lag(l.resin_lot_id) over (partition by l.press_id order by l.load_ts) as prev_lot_id
    from {{ ref('stg_materials__material_loads') }} l
    left join {{ ref('stg_materials__resin_lots') }} r on r.lot_id = l.resin_lot_id
    left join {{ ref('resin_grades') }} g on g.resin_id = r.resin_id
),
lot_changes as (
    select press_id, load_ts as lot_change_ts from loads where prev_lot_id is distinct from resin_lot_id
),
dryer as (
    select * from {{ ref('stg_materials__dryer_log') }}
),
{% for e in mold_events %}
ev_{{ e | lower }} as (
    select mold_id, event_ts, shot_no_at_event from {{ ref('int_mold_events') }} where event_type = '{{ e }}'
),
{% endfor %}
press_ev as (
    select press_id, event_type, event_ts, shot_no_at_event from {{ ref('int_mold_events') }} where press_id is not null
),
changes as (
    select c.job_id, c.parameter, c.change_ts, c.new_value
    from {{ ref('stg_qms__setpoint_changes') }} c
    where c.new_value is not null
),
drift_corrections as (
    select press_id, change_ts from {{ ref('stg_qms__setpoint_changes') }}
    where reason_code in ('drift_correction', 'defect_response') and old_value <> new_value
),
sheets as (
    select * from {{ ref('stg_engineering__process_sheets') }}
),
approvals as (
    select job_id, approval_ts from {{ ref('stg_qms__first_shot_approvals') }}
),
base as (
    select
        s.shot_id, s.press_id, s.mold_id, s.job_id, s.shot_ts, s.cycle_no, s.unit_job_id, s.unit_job_field_stale,
        s.mold_shot_no, s.press_shot_no,
        a.approval_ts,
        s.shot_ts >= a.approval_ts as after_approval,
        l.load_id, l.resin_lot_id, l.resin_id, l.supplier_id, l.cert_melt_flow_index, l.cert_moisture_pct,
        l.cert_mfi_band_position, l.regrind_pct, l.dryer_id, l.hygroscopic, l.min_residence_h,
        datediff('minute', l.lot_first_load_ts, s.shot_ts) / 60.0 as lot_residence_h,
        lc.lot_change_ts
    from shots s
    left join approvals a using (job_id)
    asof left join loads l on s.press_id = l.press_id and s.shot_ts >= l.load_ts
    asof left join lot_changes lc on s.press_id = lc.press_id and s.shot_ts >= lc.lot_change_ts
)
select
    b.*,
    case when b.after_approval
         then row_number() over (partition by b.job_id, b.after_approval order by b.shot_ts) - 1
         else -row_number() over (partition by b.job_id, b.after_approval order by b.shot_ts desc) end as shots_since_approval,
    d.dew_point_c as dryer_dew_point_c,
    d.actual_temp_c as dryer_temp_c,
    b.hygroscopic and b.lot_residence_h < b.min_residence_h as residence_below_min,
    {% for e in mold_events %}
    b.mold_shot_no - ev_{{ e | lower }}.shot_no_at_event as shots_since_{{ e | lower }},
    {% endfor %}
    datediff('hour', ring.event_ts, b.shot_ts) / 24.0 as days_since_ring_replace,
    datediff('hour', tcu.event_ts, b.shot_ts) / 24.0 as days_since_tcu_service,
    datediff('hour', b.lot_change_ts, b.shot_ts) as hours_since_lot_change,
    dc.change_ts as last_drift_correction_ts,
    {% for p in params %}
    coalesce(ch_{{ loop.index }}.new_value, sh.{{ p }}) - sh.{{ p }} as setpoint_delta_{{ p }},
    {% endfor %}
    -- machine side against the process window
    m.cycle_time_s, m.fill_time_s, m.switchover_position_mm, m.switchover_pressure_bar, m.peak_injection_pressure_bar,
    m.cushion_mm, m.hold_pressure_bar, m.recovery_time_s, m.back_pressure_bar, m.screw_rpm,
    m.barrel_zone_3_actual_c, m.nozzle_actual_c, m.mold_temp_a_c, m.mold_temp_b_c, m.clamp_tonnage, m.press_alarm_code,
    (m.cushion_mm - 5.0) / 1.5 as cushion_window_pos,
    (m.hold_pressure_bar / sh.hold_pressure_bar - 1) * 100 / sh.window_hold_pressure_pct as hold_pressure_window_pos,
    (m.nozzle_actual_c - sh.melt_temp_c) / sh.window_melt_temp_c as melt_temp_window_pos,
    ((m.mold_temp_a_c + m.mold_temp_b_c) / 2 - sh.mold_temp_c) / sh.window_mold_temp_c as mold_temp_window_pos,
    m.mold_temp_b_c - m.mold_temp_a_c as mold_temp_half_diff_c,
    -- season
    extract('month' from b.shot_ts) as shot_month,
    sin(2 * pi() * dayofyear(b.shot_ts) / 365.25) as season_sin,
    cos(2 * pi() * dayofyear(b.shot_ts) / 365.25) as season_cos
from base b
left join {{ ref('stg_mes__machine_shot_data') }} m using (shot_id)
left join sheets sh on sh.mold_id = b.mold_id and sh.press_id = b.press_id
asof left join dryer d on b.dryer_id = d.dryer_id and b.shot_ts >= d.ts
{% for e in mold_events %}
asof left join ev_{{ e | lower }} on b.mold_id = ev_{{ e | lower }}.mold_id and b.shot_ts >= ev_{{ e | lower }}.event_ts
{% endfor %}
asof left join (select * from press_ev where event_type = 'check_ring_replace') ring
  on b.press_id = ring.press_id and b.shot_ts >= ring.event_ts
asof left join (select * from press_ev where event_type = 'mold_temperature_controller_service') tcu
  on b.press_id = tcu.press_id and b.shot_ts >= tcu.event_ts
asof left join drift_corrections dc on b.press_id = dc.press_id and b.shot_ts >= dc.change_ts
{% for p in params %}
asof left join (select * from changes where parameter = '{{ p }}') ch_{{ loop.index }}
  on b.job_id = ch_{{ loop.index }}.job_id and b.shot_ts >= ch_{{ loop.index }}.change_ts
{% endfor %}
