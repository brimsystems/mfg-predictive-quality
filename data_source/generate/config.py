"""
Central configuration for the source-system extracts.

Every rate, band and physical parameter the generators use lives here, so a
change in plant conditions is a change to this file only. The instrumented
cell (presses IM-11 and IM-12) is parameterized by mold in MOLDS and by
mechanism (G0 to G10) in CELL; the label processes are in LABELS.
"""
import os
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
MODULE_DIR    = Path(__file__).parent.parent.parent
RAW_DIR       = MODULE_DIR / "data_source" / "raw"
SAMPLES_DIR   = MODULE_DIR / "data_source" / "samples"
# Quality engineering's root-cause register and the cell's state history.
# Kept outside the load path; read only by the validation checks and the
# technical report's mechanism-recovery section.
REFERENCE_DIR = MODULE_DIR / "data_source" / "reference"

RANDOM_SEED = int(os.environ.get("EXTRACT_SEED", 7))
SAMPLE_SIZE = 200

# ── Window ────────────────────────────────────────────────────────────────
START_DATE    = datetime(2025, 1, 6)     # first Monday of 2025
END_DATE      = datetime(2026, 3, 28)    # last production day in the extract
SNAPSHOT_DATE = datetime(2026, 4, 30)    # QMS extract date
SHIFTS = {"A": (6, 14), "B": (14, 22), "C": (22, 30)}   # three shifts, Monday 06:00 to Saturday 06:00

# ── Plant ─────────────────────────────────────────────────────────────────
N_PRESSES = 24
CELL_PRESSES = ["IM-11", "IM-12"]
PRESS_TONNAGE = [55, 85, 110, 150, 200, 300, 450]

# ── Instrumented molds ────────────────────────────────────────────────────
# sensors: (position, cavity). flow_ratio: end-of-fill arrival over post-gate
# arrival. weight_g and dim_mm are nominal per piece; tol_mm is the drawing
# tolerance on the critical dimension.
MOLDS = {
    "M-2041": dict(part="P-40411", desc="Automotive connector housing", cavities=8, resin="PA66 GF30",
                   cycle_s=24.0, runs_per_year=18, sensors=[("post_gate", 3), ("end_of_fill", 3)],
                   flow_ratio=2.1, fill_s=0.85, transfer_bar=620, pack_bar=410, seal_s=4.2, decay_s=1.1,
                   eof_ratio=0.62, weight_g=6.40, dim_mm=24.60, tol_mm=0.08, thick=1.0, weld_line=False,
                   housing=False, class_a=False, customer="Customer D", program="automotive"),
    "M-2043": dict(part="P-40431", desc="Automotive connector housing, mating half", cavities=8, resin="PA66 GF30",
                   cycle_s=26.0, runs_per_year=18, sensors=[("post_gate", 3), ("end_of_fill", 3)],
                   flow_ratio=2.3, fill_s=0.95, transfer_bar=640, pack_bar=420, seal_s=4.5, decay_s=1.2,
                   eof_ratio=0.60, weight_g=7.10, dim_mm=26.20, tol_mm=0.08, thick=1.0, weld_line=False,
                   housing=False, class_a=False, customer="Customer D", program="automotive"),
    "M-2118": dict(part="P-21181", desc="Medical device housing, front", cavities=4, resin="PC/ABS",
                   cycle_s=38.0, runs_per_year=12, sensors=[("post_gate", 1), ("end_of_fill", 1),
                                                            ("post_gate", 4), ("end_of_fill", 4)],
                   flow_ratio=2.6, fill_s=1.40, transfer_bar=560, pack_bar=380, seal_s=7.0, decay_s=1.8,
                   eof_ratio=0.55, weight_g=38.5, dim_mm=112.40, tol_mm=0.15, thick=1.4, weld_line=True,
                   housing=True, class_a=True, customer="Customer M", program="medical"),
    "M-2119": dict(part="P-21191", desc="Medical device housing, rear", cavities=4, resin="PC/ABS",
                   cycle_s=41.0, runs_per_year=12, sensors=[("post_gate", 1), ("end_of_fill", 1),
                                                            ("post_gate", 4), ("end_of_fill", 4)],
                   flow_ratio=2.7, fill_s=1.50, transfer_bar=570, pack_bar=385, seal_s=7.4, decay_s=1.9,
                   eof_ratio=0.54, weight_g=41.2, dim_mm=112.40, tol_mm=0.15, thick=1.4, weld_line=True,
                   housing=True, class_a=True, customer="Customer M", program="medical"),
    "M-2260": dict(part="P-22601", desc="Automotive trim clip", cavities=16, resin="POM",
                   cycle_s=18.0, runs_per_year=14, sensors=[("post_gate", 8)],
                   flow_ratio=1.9, fill_s=0.60, transfer_bar=700, pack_bar=480, seal_s=3.2, decay_s=0.8,
                   eof_ratio=0.65, weight_g=2.15, dim_mm=18.00, tol_mm=0.10, thick=0.9, weld_line=False,
                   housing=False, class_a=False, customer="Customer R", program="automotive"),
    "M-2301": dict(part="P-23011", desc="Fluid-handling manifold", cavities=2, resin="PP copolymer",
                   cycle_s=55.0, runs_per_year=10, sensors=[("post_gate", 1), ("end_of_fill", 1),
                                                            ("post_gate", 2), ("end_of_fill", 2)],
                   flow_ratio=2.4, fill_s=1.80, transfer_bar=420, pack_bar=300, seal_s=12.0, decay_s=2.6,
                   eof_ratio=0.58, weight_g=88.0, dim_mm=140.00, tol_mm=0.30, thick=3.2, weld_line=True,
                   housing=False, class_a=False, customer="Customer F", program="industrial"),
}

# ── Resins ────────────────────────────────────────────────────────────────
# mfi: grade nominal melt flow index and its band; hygroscopic grades dry.
RESINS = {
    "PA66 GF30":    dict(mfi=12.0, band=(9.0, 15.0), hygroscopic=True,  min_residence_h=4.0, suppliers=["RS-1", "RS-2", "RS-3"]),
    "PC/ABS":       dict(mfi=18.0, band=(14.0, 22.0), hygroscopic=True, min_residence_h=3.0, suppliers=["RS-2", "RS-4"]),
    "POM":          dict(mfi=9.0,  band=(7.5, 10.5), hygroscopic=False, min_residence_h=0.0, suppliers=["RS-1", "RS-5"]),
    "PP copolymer": dict(mfi=20.0, band=(16.0, 24.0), hygroscopic=False, min_residence_h=0.0, suppliers=["RS-3", "RS-5"]),
}
LOW_FLOW_SUPPLIER = "RS-3"          # lots sit low in the flow band
LOT_MFI_SD_FRAC   = 0.065            # lot-to-lot true melt flow spread, fraction of nominal
CERT_MFI_ERR_FRAC = 0.05            # lot-level error between the cert and the true value
LOT_KG            = (1200, 3000)

# ── Cell schedule ─────────────────────────────────────────────────────────
CELL = dict(
    run_days=(2.0, 6.0),
    uptime=0.91,                      # share of scheduled time the press is cycling
    stop_minutes=(4, 55),             # unplanned stops within a run
    approval_shots=(20, 90),          # shots from mold set to first-shot approval
    load_every_h=(3.0, 7.0),          # hopper loads
    regrind_pct=(0, 15),
    curve_retain_every=25,            # one shot in 25 retained, plus every alarmed or sorted shot
    job_field_lag_share=0.015,        # shots still carrying the previous job after a changeover

    # ── Mechanisms ────────────────────────────────────────────────────────
    # G1 resin lot viscosity: log-viscosity per unit of log(MFI_nominal / MFI_true)
    g1_visc_exp=1.5,
    # G2 vent restriction: rises per shot since cleaning, per mold
    vent_clean_shots={"M-2041": 26000, "M-2043": 26000, "M-2118": 16000, "M-2119": 16000,
                      "M-2260": 40000, "M-2301": 10000},
    vent_max=0.04,                    # restriction reached at the cleaning count
    vent_exp=8.0,                     # restriction builds slowly, then sharply
    vent_eof_coef=1.6,                # end-of-fill pressure loss per unit of restriction
    # G3 setup convergence after approval
    g3_tau_shots=(100.0, 350.0),      # experienced to new to the mold
    g3_tenure_days=180.0,
    g3_offset=0.056,                   # initial log offset on the pack and fill levels
    # G4 check ring (IM-12 drifts up and is replaced in October 2025)
    ring_leak_rate_per_day=0.00007,
    ring_change_date="2025-10-14",
    ring_var_coef=35.0,                # pack-integral variance per unit of leakage
    # G5 moisture on hygroscopic grades
    dryer_ids=["DR-1", "DR-2", "DR-3"],
    faulty_dryer="DR-2",
    faulty_dew_offset_c=9.0,
    faulty_from="2025-06-02", faulty_to="2025-09-19",
    moisture_visc_coef=-0.15,
    # G6 mold temperature: one controller drifts between services; summer chiller load
    tcu_drift_c_per_day=0.045,
    tcu_service_days=(45, 90),
    summer_peak_c=3.5,
    # G7 hot-runner tip wear (M-2041, M-2043)
    tip_change_shots=260000,
    # G8 technician adjustments
    changes_per_run=(2, 7),
    tamper_share=0.12,
    # G9 cavity blocking
    block_prob_per_run=0.10,
    # G10 novel signatures
    g10_events_per_year=5,

    # ── Sensor layer ─────────────────────────────────────────────────────
    sensor_gain_err=(0.01, 0.03),
    sensor_offset_bar=(1.0, 6.0),
    ar_integrals=(0.78, 0.9),
    ar_timings=(0.3, 0.5),
    noise_integral_frac=0.007,
    noise_timing_frac=0.005,
    dropout_rate=0.003,
    spike_rate=0.0005,

    # ── Templates ────────────────────────────────────────────────────────
    # alarm band half-widths as a fraction of the template value (from launch DOE)
    alarm_band={"fill_integral": 0.075, "pack_integral": 0.065, "peak_pressure_bar": 0.08,
                "end_of_fill_pressure_bar": 0.11, "gate_seal_time_s": 0.10},
    warning_frac=0.6,
    match_score_threshold=0.80,
    widen_prob=0.12, widen_factor=1.35,
)

# ── Defect model: per-piece probability from the shot's deviations ─────────
# Each code: logistic in a driver built from summary deviations (as a fraction
# of the alarm band) and state; base is the no-signal residual rate (G0).
DEFECTS = dict(
    base_rate={"short_shot": 0.00006, "flash": 0.0002, "sink": 0.0003, "void": 0.0001, "splay": 0.0002,
               "burn": 0.00005, "weld_line": 0.0002, "warp": 0.0003, "dimensional": 0.0,
               "black_specks": 0.0009, "contamination": 0.0008, "gate_vestige": 0.0003, "other": 0.0008},
)

# ── Labels ────────────────────────────────────────────────────────────────
LABELS = dict(
    audit_every_h=1.0,
    audit_pieces=(6, 20),
    audit_robot_share=0.875,            # sampled cycle recorded by the robot
    gauge_rr_frac=0.15,                 # gauge R&R as a fraction of tolerance (1 sigma x 6)
    weight_gauge_g=0.012,               # relative gauge error on weight (1 sigma, fraction of nominal x 1e-2)
    inspector_strictness=(0.75, 1.30),
    pack_find_rate=0.72,                # visible defects packers catch
    tally_wrong_code=0.065, tally_wrong_job=0.04, tally_adjacent_hour=0.12,
    tally_cavity_share=0.55,
    return_rate=0.10,                   # escaped defective pieces that come back
    return_unattributed=0.175,
    review_miss=0.04,
    maturity_days=21,
)

# ── Plant-wide jobs (all 24 presses, job level) ───────────────────────────
PLANT = dict(jobs_per_press_week=1.6, qty=(2000, 60000), scrap_rate=0.021)
