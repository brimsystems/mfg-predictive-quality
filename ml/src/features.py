"""
The shared feature pipeline.

Every feature is computed in dbt from records at or before the shot (as-of joins),
so a shot's features never see its own outcome or anything later. Feature groups
are named so the ablations can drop them:

  cavity    per-shot cavity pressure values normalized to the template
  machine   machine-side values normalized to the process window
  history   shots since approval and each maintenance event, lot and dryer
            context, setpoint deltas from the process sheet, season
  state     control-chart and drift states
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data_pipeline" / "molding.duckdb"
DATA_DIR = ROOT / "ml" / "data"

TRAIN_END = pd.Timestamp("2025-10-01")
VALID_END = pd.Timestamp("2025-12-01")
TEST_END = pd.Timestamp("2026-04-01")
SEEDS = [11, 23, 37, 51, 73]

CAVITY = ["pg_fill_dev", "pg_pack_dev", "pg_peak_dev", "pg_gate_seal_dev", "pg_pack_dev_spread",
          "eof_pressure_dev", "eof_pack_dev", "match_score_min", "max_abs_dev",
          "pg_fill_time_to_sensor_rel", "pg_time_to_peak_rel", "pg_cooling_rate_rel", "pg_cycle_integral_rel"]
POST_GATE_ONLY = [c for c in CAVITY if not c.startswith("eof_")]
MACHINE = ["cushion_window_pos", "hold_pressure_window_pos", "melt_temp_window_pos", "mold_temp_window_pos",
           "mold_temp_half_diff_c", "fill_time_rel", "recovery_time_rel", "peak_injection_pressure_rel",
           "switchover_pressure_rel", "cycle_time_rel"]
HISTORY = ["shots_since_approval", "shots_since_vent_cleaning", "shots_since_hot_runner_tip", "shots_since_pm",
           "days_since_ring_replace", "days_since_tcu_service", "cert_mfi_band_position", "cert_moisture_pct",
           "regrind_pct", "dryer_dew_point_c", "lot_residence_h", "hours_since_lot_change",
           "setpoint_delta_hold_pressure_bar", "setpoint_delta_injection_velocity_mm_s",
           "setpoint_delta_switchover_position_mm", "setpoint_delta_melt_temp_c", "setpoint_delta_mold_temp_c",
           "tech_prior_runs_on_mold", "season_sin", "season_cos", "active_cavities"]
STATE = ["ewma_pack", "ewma_fill", "cusum_eof", "cusum_gate_seal", "cusum_pack_var", "spc_max_abs_z"]
GROUPS = {"cavity": CAVITY, "machine": MACHINE, "history": HISTORY, "state": STATE}
ALL = CAVITY + MACHINE + HISTORY + STATE


def connect():
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    # staging views read the raw extracts by paths relative to the dbt project
    con.execute(f"set file_search_path = '{(ROOT / 'data_pipeline').as_posix()}'")
    return con


def split_of(ts: pd.Series) -> pd.Series:
    return pd.Series(np.where(ts < TRAIN_END, "train", np.where(ts < VALID_END, "valid", "test")), index=ts.index)


def load_shots(columns=None, where="true"):
    con = connect()
    cols = "*" if columns is None else ", ".join(columns)
    df = con.execute(f"select {cols} from fct_shot where {where} order by shot_ts").df()
    con.close()
    df["split"] = split_of(df["shot_ts"])
    df["cell"] = df["mold_id"] + " / " + df["press_id"]
    return df


def load_audit_pieces():
    con = connect()
    df = con.execute("select * from fct_audit_piece where linked_to_shot order by shot_ts").df()
    con.close()
    return df
