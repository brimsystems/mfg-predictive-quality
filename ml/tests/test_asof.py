"""
As-of rule: every feature on a shot is computed from records at or before the shot.

Each test recomputes a context field for a sample of shots directly from the staged
source and checks the shot never sees a record stamped after it.
"""
import pytest

from ml.src.features import WAREHOUSE, connect

pytestmark = pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not built")

SAMPLE = "using sample 20000 rows (reservoir, 7)"


def _violations(sql):
    con = connect()
    n = con.execute(sql).fetchone()[0]
    con.close()
    return n


def test_load_is_at_or_before_shot():
    assert _violations(f"""
        with s as (select shot_id, shot_ts, load_id from int_shot_context {SAMPLE})
        select count(*) from s join stg_materials__material_loads l using (load_id) where l.load_ts > s.shot_ts
    """) == 0


def test_setpoints_in_force_at_shot():
    # the setpoint delta on a shot must equal the last change at or before it (or zero)
    assert _violations(f"""
        with s as (select shot_id, job_id, shot_ts, setpoint_delta_hold_pressure_bar as d, mold_id, press_id
                   from int_shot_context {SAMPLE}),
        last as (
            select s.shot_id, arg_max(c.new_value, c.change_ts) as v
            from s join stg_qms__setpoint_changes c
              on c.job_id = s.job_id and c.parameter = 'hold_pressure_bar' and c.change_ts <= s.shot_ts
              and c.new_value is not null
            group by 1)
        select count(*) from s left join last using (shot_id)
        join stg_engineering__process_sheets p on p.mold_id = s.mold_id and p.press_id = s.press_id
        where abs(coalesce(last.v, p.hold_pressure_bar) - p.hold_pressure_bar - s.d) > 1e-6
    """) == 0


def test_maintenance_counts_never_negative():
    assert _violations("""
        select count(*) from int_shot_context
        where shots_since_vent_cleaning < 0 or shots_since_pm < 0 or shots_since_hot_runner_tip < 0
           or days_since_ring_replace < 0 or days_since_tcu_service < 0
    """) == 0


def test_template_effective_before_shot():
    assert _violations(f"""
        with s as (select shot_id, sensor_id, shot_ts, template_from from int_sensor_shot_normalized {SAMPLE})
        select count(*) from s where template_from > shot_ts
    """) == 0


def test_job_baseline_uses_first_production_shots_only():
    # drift and chart baselines come from the first 500 shots after approval
    assert _violations("""
        select count(*) from spc_chart_points p join int_shot_context c using (shot_id)
        where p.point_no <= 500 and c.shots_since_approval >= 600
    """) == 0
