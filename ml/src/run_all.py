"""
Runs every model and evaluation, logs the headline figures to MLflow, and writes ml/data/results/.

  supervised defect    any code and per code, five seeds, against the template limits, with ablations and the
                       audit-only re-test (fixed split: train through September 2025, test December to March)
  rolling origin       virtual metrology and anomaly detection retrained monthly on all prior data, May 2025 to
                       March 2026; the curve autoencoder as a comparison
  drift mechanisms     each drift detector against the mechanism it is built for
  measures             alarm budget, shot-level detection with the incremental column, anomaly per event, virtual
                       metrology by month, all at the deployed thresholds

Usage: python -m ml.src.run_all [defect] [rolling] [drift] [measures]
"""
import json
import time

import mlflow
import pandas as pd

from . import defect, drift_mechanisms, measures, rolling
from .features import ALL, CAVITY, DATA_DIR, HISTORY, MACHINE, POST_GATE_ONLY, ROOT, STATE, load_shots

RESULTS = DATA_DIR / "results"
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'ml' / 'mlruns.db').as_posix()}")
mlflow.set_experiment("molding-cell")

ABLATIONS = {
    "post_gate_only": [c for c in ALL if c not in CAVITY] + POST_GATE_ONLY,
    "no_machine_side": [c for c in ALL if c not in MACHINE],
    "no_cavity_signal": MACHINE + HISTORY,
    "no_history": CAVITY + MACHINE + STATE,
}


def log(name, params, metrics):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})


def run_defect(df):
    print("supervised defect", flush=True)
    d = defect.run(df, tag="full")
    for t, g in d[d["usable"] == True].groupby("target"):
        log(f"defect_{t}", dict(target=t), dict(recall_model=g["recall_model"].mean(), recall_template=g["recall_template"].mean(),
                                                ap_model=g["ap_model"].mean(), ap_template=g["ap_template"].mean()))
    dab = []
    for name in ("no_machine_side", "no_cavity_signal"):
        dab.append(defect.run(df, feats=ABLATIONS[name], tag=name))
    pd.concat(dab).to_csv(RESULTS / "defect_ablations.csv", index=False)
    ao = defect.evaluate_audit_only(defect.labeled(df))
    (RESULTS / "defect_audit_only.json").write_text(json.dumps(dict(mean=ao.mean(numeric_only=True).to_dict(),
                                                                    sd=ao.std(numeric_only=True).to_dict()), indent=1, default=float))


def main(stages=("defect", "rolling", "drift", "measures")):
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if "defect" in stages:
        run_defect(load_shots())
    if "rolling" in stages:
        rolling.main(("vm", "anomaly", "autoencoder"))
    if "drift" in stages:
        drift_mechanisms.main()
    if "measures" in stages:
        measures.main()
        m = json.loads((RESULTS / "measures.json").read_text())
        inc = m["shot_level"]["incremental"]
        log("measures", dict(period="2025-05 to 2026-03"),
            {f"episodes_{k}": v["episodes_per_shift"] for k, v in m["budget"].items() if isinstance(v, dict) and "episodes_per_shift" in v}
            | {f"incremental_{k}": inc[k]["All defects"]["above_chance"] for k in inc})
    print(f"done in {time.time() - t0:,.0f} s", flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("defect", "rolling", "drift", "measures"))
