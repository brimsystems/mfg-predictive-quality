"""
Supervised defect prediction, attempted and reported as it lands.

Labeled shots are the ones a confirmed outcome links to: sorted shots the reject-bin
review confirmed or returned as good, and audited shots the robot recorded. The
target is a confirmed defective piece on the shot (any code, then per code). The
baselines are the template limits (the largest deviation as a fraction of the alarm
band) and the control-chart rules (the largest z on the shot's charts). Every
comparison is at the template's own alarm rate on the labeled set.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from .features import ALL, DATA_DIR, SEEDS

CODES = ["short_shot", "flash", "sink", "void", "dimensional", "splay", "weld_line", "burn", "warp",
         "gate_vestige", "black_specks", "contamination", "other"]


def labeled(df):
    lab = df[df["after_approval"] & (df["pieces_reviewed"].notna() | df["audit_pieces"].notna())].copy()
    codes = (lab["sort_defect_codes"].fillna("") + ";" + lab["audit_defect_codes"].fillna("")).str.strip(";")
    lab["y_any"] = ((lab["pieces_confirmed_defective"].fillna(0) > 0) | (lab["audit_rejects"].fillna(0) > 0)).astype(int)
    for c in CODES:
        lab[f"y_{c}"] = codes.str.contains(rf"(?:^|;){c}(?:;|$)").astype(int)
    lab["confirmed_pieces"] = lab["pieces_confirmed_defective"].fillna(0) + lab["audit_rejects"].fillna(0)
    return lab


def recall_at_rate(y, score, rate):
    if y.sum() == 0:
        return np.nan
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y)[top].sum() / y.sum())


def evaluate(lab, target, feats=ALL, seeds=SEEDS):
    tr, va, te = (lab[lab["split"] == s] for s in ("train", "valid", "test"))
    y_te = te[target].to_numpy()
    n_pos = {s: int(lab.loc[lab["split"] == s, target].sum()) for s in ("train", "valid", "test")}
    base = dict(target=target, n_train_pos=n_pos["train"], n_valid_pos=n_pos["valid"], n_test_pos=n_pos["test"],
                n_test=len(te))
    if n_pos["train"] < 30 or n_pos["test"] < 10:
        return [dict(base, seed=s, usable=False) for s in seeds]
    rate = float((te["unit_alarm_state"] == "alarm").mean())
    tmpl = te["max_abs_dev"].fillna(0).to_numpy()
    spc = te["spc_max_abs_z"].fillna(0).to_numpy()
    rows = []
    for s in seeds:
        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, scale_pos_weight=max(1.0, (len(tr) - n_pos["train"]) / n_pos["train"]),
                              eval_metric="aucpr", early_stopping_rounds=60, random_state=s, tree_method="hist")
        m.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], verbose=False)
        p = m.predict_proba(te[feats])[:, 1]
        k = max(1, int(round(rate * len(te))))
        top = np.argsort(-p)[:k]
        cost_model = float(te["confirmed_pieces"].to_numpy()[top].sum())
        top_t = np.argsort(-tmpl)[:k]
        cost_tmpl = float(te["confirmed_pieces"].to_numpy()[top_t].sum())
        rows.append(dict(base, seed=s, usable=True, alarm_rate=rate,
                         ap_model=float(average_precision_score(y_te, p)),
                         ap_template=float(average_precision_score(y_te, tmpl)),
                         ap_spc=float(average_precision_score(y_te, spc)),
                         recall_model=recall_at_rate(y_te, p, rate),
                         recall_template=recall_at_rate(y_te, tmpl, rate),
                         recall_spc=recall_at_rate(y_te, spc, rate),
                         pieces_captured_model=cost_model, pieces_captured_template=cost_tmpl,
                         pieces_total=float(te["confirmed_pieces"].sum())))
    return rows


def evaluate_audit_only(lab, feats=ALL, seeds=SEEDS):
    """Trained as usual; scored only on test shots the robot sampled at a recorded cycle, with the audit's
    own finding as the label. These pieces were pulled on a clock, not because the template sorted them."""
    tr, va = lab[lab["split"] == "train"], lab[lab["split"] == "valid"]
    te = lab[(lab["split"] == "test") & lab["audit_pieces"].notna()].copy()
    y = (te["audit_rejects"].fillna(0) > 0).astype(int).to_numpy()
    rate = float((te["unit_alarm_state"] == "alarm").mean())
    tmpl = te["max_abs_dev"].fillna(0).to_numpy()
    n_pos = int(lab.loc[lab["split"] == "train", "y_any"].sum())
    rows = []
    for s in seeds:
        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, scale_pos_weight=max(1.0, (len(tr) - n_pos) / n_pos),
                              eval_metric="aucpr", early_stopping_rounds=60, random_state=s, tree_method="hist")
        m.fit(tr[feats], tr["y_any"], eval_set=[(va[feats], va["y_any"])], verbose=False)
        p = m.predict_proba(te[feats])[:, 1]
        rows.append(dict(seed=s, n_test=len(te), n_test_pos=int(y.sum()), alarm_rate=rate,
                         recall_model=recall_at_rate(y, p, rate), recall_template=recall_at_rate(y, tmpl, rate),
                         ap_model=float(average_precision_score(y, p)), ap_template=float(average_precision_score(y, tmpl))))
    return pd.DataFrame(rows)


def run(df, feats=ALL, tag="full"):
    lab = labeled(df)
    rows = []
    for target in ["y_any"] + [f"y_{c}" for c in CODES]:
        rows += evaluate(lab, target, feats)
    res = pd.DataFrame(rows)
    res["tag"] = tag
    d = DATA_DIR / "results"
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / f"defect_metrics_{tag}.csv", index=False)
    return res
