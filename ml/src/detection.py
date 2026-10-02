"""
Detection with a chance baseline.

Shot-level, on audit pieces linked to their shot: for each layer and defect group, the share of defective audited
pieces whose shot the layer flagged, against the share of good audited pieces from the same audits whose shot it
flagged. The difference is detection above chance. Template, control-chart rules and drift involve no trained model
and use all fifteen months of audits; the anomaly model and the virtual metrology advisory use the test period only.
95% intervals come from a bootstrap that resamples whole audits.

Job-hour coverage (secondary): the share of confirmed defects credited to some layer under the window rules, against
the share of good audited pieces whose job-hour would have been credited under the same rules.

Usage: python -m ml.src.detection
"""
import json

import numpy as np
import pandas as pd

from . import vm
from .features import DATA_DIR, VALID_END, connect, load_shots
from .layers import LOOKBACK
from .run_all import VM_AUDIT, window_signals

RESULTS = DATA_DIR / "results"
GROUPS = {"Fill volume": ["short_shot", "flash"], "Packing and shrinkage": ["sink", "void", "dimensional", "warp"],
          "Flow front": ["weld_line", "burn"], "Gate and other": ["gate_vestige", "other"],
          "Material": ["splay", "black_specks", "contamination"]}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
LAYERS = ["template_sort", "rules", "drift", "spc_any", "virtual_metrology", "anomaly", "any_layer"]
TEST_ONLY = {"virtual_metrology", "anomaly", "any_layer"}
END = pd.Timestamp("2026-04-01")
B = 400
ORDER = ["template_sort", "rules", "drift", "anomaly", "virtual_metrology"]


def boot(flag, defect, audit, rng, key_mask):
    """Bootstrap over audits: rate among defective pieces, rate among good pieces, and the difference."""
    f = np.nan_to_num(flag.astype(float))
    dk = defect.astype(bool) & key_mask.astype(bool)
    gd = ~defect.astype(bool)
    d = pd.DataFrame({"a": audit, "fd": f * dk, "nd": dk.astype(float), "fg": f * gd, "ng": gd.astype(float)})
    A = d.groupby("a")[["fd", "nd", "fg", "ng"]].sum().to_numpy()
    tot = A.sum(0)
    rd, rg = tot[0] / max(tot[1], 1), tot[2] / max(tot[3], 1)
    idx = rng.integers(0, len(A), size=(B, len(A)))
    S = A[idx].sum(1)
    bd = S[:, 0] / np.maximum(S[:, 1], 1)
    bg = S[:, 2] / np.maximum(S[:, 3], 1)
    diff = bd - bg
    return dict(n_defective=int(tot[1]), n_good=int(tot[3]), rate_defective=float(rd), rate_good=float(rg), above_chance=float(rd - rg),
                ci_defective=[float(np.quantile(bd, .025)), float(np.quantile(bd, .975))],
                ci_good=[float(np.quantile(bg, .025)), float(np.quantile(bg, .975))],
                ci_above_chance=[float(np.quantile(diff, .025)), float(np.quantile(diff, .975))])


def shot_flags(df):
    """Per shot: each layer's flag."""
    an = pd.read_parquet(RESULTS / "anomaly_if_scores.parquet")
    an = an[an["seed"] == an["seed"].min()]
    te = df[(df["shot_ts"] >= VALID_END) & (df["shot_ts"] < END) & df["after_approval"]]
    _, sig = window_signals(df, an, VALID_END, END, pd.Series(dtype=float))
    unusual = set(sig["anomaly"]["shot_id"])
    f = df[["shot_id", "shot_ts", "job_id", "unit_alarm_state", "spc_we1", "spc_we2", "drift_any_signal"]].copy()
    f["template_sort"] = f["unit_alarm_state"] == "alarm"
    f["rules"] = f["spc_we1"] | f["spc_we2"]
    # drift counts on shots where a detector is signaling: the EWMA beyond its limit, a CUSUM crossing its limit
    f["drift"] = f["drift_any_signal"].astype(bool)
    f["spc_any"] = f["template_sort"] | f["rules"] | f["drift"]
    f["anomaly"] = f["shot_id"].isin(unusual)
    f["test"] = f["shot_id"].isin(set(te["shot_id"]))
    return f.set_index("shot_id"), sig


def main():
    rng = np.random.default_rng(7)
    df = load_shots()
    flags, sig = shot_flags(df)
    con = connect()
    ap = con.execute("""select audit_id, shot_id, cavity_id, defect_code, audit_ts, job_id from fct_audit_piece
                        where linked_to_shot and after_approval""").df()
    con.close()
    ap = ap.join(flags, on="shot_id", rsuffix="_s")
    ap["defective"] = ap["defect_code"].notna()
    ap["group"] = ap["defect_code"].map(GROUP_OF)
    # virtual metrology advisory on the piece's own cavity, test period
    tp = ap[ap["test"]]
    shots_te = df[df["shot_id"].isin(tp["shot_id"].unique())]
    pr = vm.predict_shots(shots_te, "dimension")
    pr = pr.set_index(["shot_id", "cavity_id"])["y"]
    ap["vm_dim"] = [pr.get((s, c), np.nan) for s, c in zip(ap["shot_id"], ap["cavity_id"])]
    ap["virtual_metrology"] = ap["vm_dim"].abs() > VM_AUDIT
    ap["any_layer"] = ap["spc_any"] | ap["anomaly"] | ap["virtual_metrology"]

    out = {"shot_level": {}, "notes": {"spc_period": "January 2025 to March 2026 (all audits)",
                                       "model_period": "December 2025 to March 2026 (test period)"}}
    targets = [("All defects", None)] + [(g, g) for g in GROUPS]
    for layer in LAYERS:
        sub = ap[ap["test"]] if layer in TEST_ONLY else ap
        res = {}
        for name, g in targets:
            mask = np.ones(len(sub), bool) if g is None else (sub["group"] == g).to_numpy()
            res[name] = boot(sub[layer].to_numpy(), sub["defective"].to_numpy(), sub["audit_id"].to_numpy(), rng, mask)
        out["shot_level"][layer] = res
    # incremental detection: each layer in order, pieces it flagged that no earlier layer flagged
    out["incremental"] = {}
    for k, layer in enumerate(ORDER):
        test_only = any(l in TEST_ONLY for l in ORDER[:k + 1])
        sub = ap[ap["test"]] if test_only else ap
        earlier = np.zeros(len(sub), bool)
        for l in ORDER[:k]:
            earlier |= sub[l].to_numpy(bool)
        inc = sub[layer].to_numpy(bool) & ~earlier
        res = {}
        for name, g in targets:
            mask = np.ones(len(sub), bool) if g is None else (sub["group"] == g).to_numpy()
            res[name] = boot(inc, sub["defective"].to_numpy(), sub["audit_id"].to_numpy(), rng, mask)
        res["period"] = "test" if test_only else "all"
        out["incremental"][layer] = res
    # good-piece flag rates and alarms per shift on the same definitions, for the alarm budget
    out["good_flag_rate"] = {l: out["shot_level"][l]["All defects"]["rate_good"] for l in ORDER}

    # virtual metrology on packing and shrinkage defects the template did not alarm on, test period
    t = ap[ap["test"] & ~ap["template_sort"]]
    mask = (t["group"] == "Packing and shrinkage").to_numpy()
    out["vm_packing_missed_by_template"] = boot(t["virtual_metrology"].to_numpy(), t["defective"].to_numpy(), t["audit_id"].to_numpy(), rng, mask)
    t2 = ap[~ap["template_sort"]]
    out["spc_packing_missed_by_template"] = boot((t2["rules"] | t2["drift"]).to_numpy(), t2["defective"].to_numpy(), t2["audit_id"].to_numpy(), rng,
                                                 (t2["group"] == "Packing and shrinkage").to_numpy())
    # by individual code (SPC layers and any_layer)
    out["by_code"] = {}
    for layer in ("template_sort", "rules", "drift", "spc_any", "virtual_metrology", "anomaly"):
        sub = ap[ap["test"]] if layer in TEST_ONLY else ap
        out["by_code"][layer] = {}
        for code in GROUP_OF:
            mask = (sub["defect_code"] == code).to_numpy()
            if mask.sum() < 5:
                continue
            r = boot(sub[layer].to_numpy(), sub["defective"].to_numpy(), sub["audit_id"].to_numpy(), rng, mask)
            out["by_code"][layer][code] = {k: r[k] for k in ("n_defective", "rate_defective", "rate_good", "above_chance")}

    # job-hour coverage and its chance level (test period)
    L = json.loads((RESULTS / "layers.json").read_text())["layers"]
    hours = {name: {(j, h) for j, h in zip(v["job_id"], pd.to_datetime(v["shot_ts"]).dt.floor("h"))} for name, v in sig.items()}
    good = ap[ap["test"] & ~ap["defective"]].copy()
    good["hour"] = pd.to_datetime(good["shot_ts"]).dt.floor("h")

    def credited(job, hour, layers):
        return any(any((job, hour - pd.Timedelta(hours=k)) in hours[l] for k in range(LOOKBACK[l] + 1)) for l in layers)

    jl = ["template_sort", "rules", "drift", "anomaly"]
    good_any = np.mean([credited(j, h, jl) for j, h in zip(good["job_id"], good["hour"])])
    good_by = {l: float(np.mean([credited(j, h, [l]) for j, h in zip(good["job_id"], good["hour"])])) for l in jl}
    out["job_hour"] = dict(detected=1 - L["share"].get("none", 0), share=L["share"], chance_any=float(good_any),
                           chance_by_layer=good_by, n_good=int(len(good)),
                           by_group={g: 1 - v for g, v in L.get("by_group_none", {}).items()})
    bc = pd.DataFrame(L["by_code"]).T.fillna(0)
    jg = {}
    for g, cs in GROUPS.items():
        s = bc.loc[[c for c in cs if c in bc.index]].sum()
        jg[g] = dict(n=float(s.sum()), detected=float(1 - s.get("none", 0) / s.sum()),
                     **{k: float(s.get(k, 0) / s.sum()) for k in ("template_sort", "rules", "drift", "virtual_metrology", "anomaly", "none")})
    out["job_hour"]["by_group"] = jg
    out["job_hour"]["by_code"] = {c: {k: float(v) for k, v in (r / r.sum()).items()} | {"n": float(r.sum())} for c, r in bc.iterrows()}
    (RESULTS / "detection.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", RESULTS / "detection.json")


if __name__ == "__main__":
    main()
