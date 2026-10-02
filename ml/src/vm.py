"""
Virtual metrology: audit part weight and critical dimension from the shot's features.

Two designs, both reported: one model per mold x press, and a pooled model with the
mold x press as a feature. Targets are taken relative to nominal (weight as a
fraction of nominal, dimension in tolerance units) so the pooled model shares
one scale; errors are reported back in grams and millimetres against each part's
gauge R&R. Five seeds; the test period is December 2025 to March 2026.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import ALL, DATA_DIR, ROOT, SEEDS, load_audit_pieces, split_of

MODELS = ROOT / "ml" / "models"

TARGETS = {"weight": ("part_weight_g", "nominal_weight_g", "gauge_sd_weight_g"),
           "dimension": ("critical_dimension_mm", "nominal_dimension_mm", "gauge_sd_dimension_mm")}
PARAMS = dict(n_estimators=1200, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0, tree_method="hist", early_stopping_rounds=60)


def prepare(df):
    df = df.copy()
    df["split"] = split_of(df["shot_ts"])
    df["cell"] = df["mold_id"] + " / " + df["press_id"]
    df["y_weight"] = df["part_weight_g"] / df["nominal_weight_g"] - 1
    df["y_dimension"] = (df["critical_dimension_mm"] - df["nominal_dimension_mm"]) / df["dimension_tolerance_mm"]
    df["cell_code"] = df["cell"].astype("category").cat.codes.astype("int64")   # int8 codes would overflow the cavity key
    df["cavity_key"] = (df["cell_code"] * 32 + df["cavity_id"]).astype(int)
    return df


def _fit(tr, va, feats, target, seed):
    m = xgb.XGBRegressor(random_state=seed, **PARAMS)
    m.fit(tr[feats], tr[target], eval_set=[(va[feats], va[target])], verbose=False)
    return m


def predict_shots(shots, kind, model=None, meta=None):
    """Predicted value per shot and cavity (target scale: weight as a fraction of nominal, dimension in
    tolerance units) from the saved pooled model, or from the model given."""
    if model is None:
        meta = json.loads((MODELS / f"vm_{kind}_pooled.meta.json").read_text())
        model = xgb.XGBRegressor()
        model.load_model(MODELS / f"vm_{kind}_pooled.json")
    m = model
    rows = []
    for cav in range(1, int(shots["cavities"].max()) + 1):
        g = shots[shots["cavities"] >= cav].copy()
        g["cavity_id"] = cav
        g["cell"] = g["mold_id"] + " / " + g["press_id"]
        g["cell_code"] = g["cell"].map(meta["cell_codes"])
        g["cavity_key"] = (g["cell_code"] * 32 + cav).astype(int)
        rows.append(pd.DataFrame(dict(shot_id=g["shot_id"], cavity_id=cav, y=m.predict(g[meta["features"]]))))
    return pd.concat(rows, ignore_index=True)


def backtest(kind, train_end, eval_start, eval_end, seed=11, feats=None):
    """Pooled model trained on pieces before train_end (its last month held out for early stopping),
    evaluated on pieces in [eval_start, eval_end). Returns metrics, the model and its meta."""
    df = prepare(load_audit_pieces())
    feats = feats or ALL
    f = feats + ["cell_code", "cavity_key"]
    stop = pd.Timestamp(train_end) - pd.DateOffset(months=1)
    tr = df[df["shot_ts"] < stop]
    va = df[(df["shot_ts"] >= stop) & (df["shot_ts"] < pd.Timestamp(train_end))]
    te = df[(df["shot_ts"] >= pd.Timestamp(eval_start)) & (df["shot_ts"] < pd.Timestamp(eval_end))]
    m = _fit(tr, va, f, f"y_{kind}", seed)
    p = _unscale(te, pd.Series(m.predict(te[f]), index=te.index), kind)
    cats = dict(zip(df["cell"].astype("category").cat.categories, range(len(df["cell"].astype("category").cat.categories))))
    return metrics(te, p, kind), m, dict(features=f, cell_codes=cats)


def _unscale(df, pred, kind):
    if kind == "weight":
        return (1 + pred) * df["nominal_weight_g"]
    return df["nominal_dimension_mm"] + pred * df["dimension_tolerance_mm"]


def fit_predict(df, feats, kind, seed, design="pooled"):
    """Returns test-period predictions in engineering units."""
    target = f"y_{kind}"
    tr, va, te = (df[df["split"] == s] for s in ("train", "valid", "test"))
    out = pd.Series(np.nan, index=te.index)
    if design == "pooled":
        f = feats + ["cell_code", "cavity_key"]
        m = _fit(tr, va, f, target, seed)
        out[:] = m.predict(te[f])
        model = m
    else:
        model = {}
        for cell, g in tr.groupby("cell"):
            v, t = va[va["cell"] == cell], te[te["cell"] == cell]
            if len(g) < 300 or len(v) < 50 or len(t) == 0:
                continue
            f = feats + ["cavity_id"]
            m = _fit(g, v, f, target, seed)
            out[t.index] = m.predict(t[f])
            model[cell] = m
    return _unscale(te, out, kind), model


def metrics(te, pred, kind):
    col, _, gauge = TARGETS[kind]
    err = te[col] - pred
    rows = []
    for cell, g in te.assign(err=err, pred=pred).dropna(subset=["pred"]).groupby("cell"):
        rmse = float(np.sqrt(np.mean(g["err"] ** 2)))
        ss = float(np.sum((g[col] - g[col].mean()) ** 2))
        g = g.sort_values("shot_ts")
        slope = float(np.polyfit(g["pred"], g[col], 1)[0]) if g["pred"].std() > 0 else np.nan
        rows.append(dict(cell=cell, n=len(g), rmse=rmse, gauge_sd=float(g[gauge].iloc[0]),
                         rmse_over_gauge=rmse / float(g[gauge].iloc[0]),
                         r2=1 - float(np.sum(g["err"] ** 2)) / ss if ss > 0 else np.nan,
                         calibration_slope=slope,
                         residual_lag1=float(pd.Series(g["err"].to_numpy()).autocorr(1))))
    return pd.DataFrame(rows)


def run(feature_set=None, designs=("pooled", "per_cell"), seeds=SEEDS, tag="full", save_predictions=True):
    df = prepare(load_audit_pieces())
    feats = feature_set or ALL
    results, preds = [], []
    for kind in TARGETS:
        for design in designs:
            for seed in seeds:
                p, model = fit_predict(df, feats, kind, seed, design)
                if tag == "full" and design == "pooled" and seed == seeds[0]:
                    MODELS.mkdir(parents=True, exist_ok=True)
                    model.save_model(MODELS / f"vm_{kind}_pooled.json")
                    cats = dict(zip(df["cell"].astype("category").cat.categories,
                                    range(len(df["cell"].astype("category").cat.categories))))
                    (MODELS / f"vm_{kind}_pooled.meta.json").write_text(json.dumps(
                        dict(features=feats + ["cell_code", "cavity_key"], cell_codes=cats)))
                te = df.loc[p.index]
                m = metrics(te, p, kind).assign(target=kind, design=design, seed=seed, tag=tag)
                results.append(m)
                if save_predictions and seed == seeds[0]:
                    preds.append(pd.DataFrame(dict(audit_id=te["audit_id"], shot_id=te["shot_id"], cavity_id=te["cavity_id"],
                                                   cell=te["cell"], shot_ts=te["shot_ts"], target=kind, design=design,
                                                   measured=te[TARGETS[kind][0]], predicted=p,
                                                   nominal=te[TARGETS[kind][1]],
                                                   tolerance=te["dimension_tolerance_mm"])))
    res = pd.concat(results, ignore_index=True)
    out = DATA_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"vm_metrics_{tag}.csv", index=False)
    if save_predictions and preds:
        pd.concat(preds).to_parquet(out / f"vm_test_predictions_{tag}.parquet", index=False)
    return res


def summarize(res):
    s = (res.groupby(["target", "design", "cell"])[["rmse", "rmse_over_gauge", "r2", "calibration_slope", "residual_lag1"]]
         .mean().reset_index())
    return s


if __name__ == "__main__":
    r = run()
    print(summarize(r).to_string())
