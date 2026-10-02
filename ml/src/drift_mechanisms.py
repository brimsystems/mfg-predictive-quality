"""
Drift detectors, each tested against the mechanism it is built for, at the deployed limits (firing events only).

  end-of-fill CUSUM   per vent-cleaning interval: did the first firing precede the burn rise, and by how many shots;
                      chance: the burn-rise time replaced by a random time in the interval (2,000 draws)
  fill EWMA           at resin lot changes mid-run: the share of lot changes with a real fill step (true fill
                      deviation moving at least 0.15 of the alarm band) flagged within 500 shots, against the share
                      of lot changes without one (under 0.05 of the band) flagged in the same window
  variance CUSUM      per check-ring wear episode (a job run on a worn ring: true leakage above 0.02): fired before the
                      first confirmed packing-variability defect (sink, void), and the lead; against the same window in
                      jobs on the same molds with an unworn ring (leakage under 0.01)

Intervals: Wilson 95% for shares; bootstrap over intervals or episodes for leads.

Usage: python -m ml.src.drift_mechanisms
"""
import json

import numpy as np
import pandas as pd

from .features import DATA_DIR, ROOT, connect

RESULTS = DATA_DIR / "results"
REF = ROOT / "data_source" / "reference"
LOT_WINDOW = 500
STEP, NO_STEP = 0.15, 0.05
WORN, UNWORN = 0.02, 0.01


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def share(flags):
    flags = np.asarray(flags, bool)
    return dict(n=int(len(flags)), k=int(flags.sum()), share=float(flags.mean()) if len(flags) else np.nan,
                ci=wilson(int(flags.sum()), len(flags)))


def boot_median(x, rng, b=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=np.nan, ci=[np.nan, np.nan])
    m = np.median(x[rng.integers(0, len(x), (b, len(x)))], axis=1)
    return dict(median=float(np.median(x)), ci=[float(np.quantile(m, .025)), float(np.quantile(m, .975))])


def load():
    con = connect()
    c = con.execute("""select c.shot_id, c.job_id, c.mold_id, c.press_id, c.shot_ts, c.lot_change_ts, c.resin_lot_id
                       from int_shot_context c where c.after_approval order by c.job_id, c.shot_ts""").df()
    d = con.execute("select shot_id, cusum_eof, ewma_fill_fire, cusum_eof_fire, cusum_pack_var_fire from drift_shot_state").df()
    maint = con.execute("select mold_id, event_ts from stg_toolroom__mold_maintenance where event_type = 'vent_cleaning'").df()
    cd = con.execute("""select job_id, hour_ts, defect_code from fct_confirmed_defects
                        where source in ('sort', 'audit', 'tally') and job_id is not null and defect_code in ('sink', 'void')""").df()
    con.close()
    st = pd.read_parquet(REF / "shot_state", columns=["shot_id", "fill_dev", "leak"])
    c = c.merge(d, on="shot_id", how="left").merge(st, on="shot_id", how="left")
    for k in ("ewma_fill_fire", "cusum_eof_fire", "cusum_pack_var_fire"):
        c[k] = c[k].fillna(False).astype(bool)
    c["pos"] = c.groupby("job_id").cumcount()
    return c, maint, cd


def vent(c, maint, rng):
    reg = pd.read_parquet(REF / "root_cause_register.parquet")
    burns = reg[reg["defect_code"] == "burn"].groupby("shot_id").size()
    c = c.assign(burn=c["shot_id"].map(burns).fillna(0))
    rows = []
    for m, g in c.groupby("mold_id"):
        if g["cusum_eof"].isna().all():
            continue                                      # no end-of-fill sensor on this mold
        g = g.sort_values("shot_ts")
        cl = maint[maint["mold_id"] == m]["event_ts"].sort_values()
        edges = [pd.Timestamp.min] + list(cl) + [pd.Timestamp.max]
        for a, b in zip(edges[:-1], edges[1:]):
            iv = g[(g["shot_ts"] > a) & (g["shot_ts"] <= b)].reset_index(drop=True)
            if len(iv) < 5000:
                continue
            f = np.where(iv["cusum_eof_fire"].to_numpy())[0]
            roll = iv["burn"].rolling(1000, min_periods=1000).sum().to_numpy()
            b0 = np.nanmean(roll[:3000]) if np.isfinite(roll[:3000]).any() else 0
            rise = np.where(roll > max(3.0, 3 * b0))[0]
            rows.append(dict(mold_id=m, length=len(iv), first_fire=int(f[0]) if len(f) else np.nan,
                             rise=int(rise[0]) if len(rise) else np.nan))
    r = pd.DataFrame(rows)
    wr = r[r["rise"].notna()].copy()
    wr["preceded"] = wr["first_fire"] < wr["rise"]
    wr["lead"] = np.where(wr["preceded"], wr["rise"] - wr["first_fire"], np.nan)
    # chance: the rise at a random time in the interval (after the first 1,000 shots, where the rise can first be seen)
    sims_p, sims_m = [], []
    L, F = wr["length"].to_numpy(), wr["first_fire"].to_numpy()
    for _ in range(2000):
        rr = rng.uniform(1000, L)
        p = np.nan_to_num(F, nan=np.inf) < rr
        sims_p.append(p.mean())
        sims_m.append(np.median((rr - F)[p]) if p.any() else np.nan)
    return dict(intervals=int(len(r)), with_rise=int(len(wr)), with_firing=int(r["first_fire"].notna().sum()),
                preceded=share(wr["preceded"]), lead=boot_median(wr["lead"], rng),
                lead_2000_6000=share((wr["lead"] >= 2000) & (wr["lead"] <= 6000)),
                chance_preceded=dict(mean=float(np.mean(sims_p)), range95=[float(np.quantile(sims_p, .025)), float(np.quantile(sims_p, .975))]),
                chance_lead_median=float(np.nanmedian(sims_m)),
                realism_requirement_met=bool(((wr["lead"] >= 2000) & (wr["lead"] <= 6000)).sum() / max(len(r), 1) >= 0.70))


def lot_steps(c):
    rows = []
    for j, g in c.groupby("job_id"):
        g = g.reset_index(drop=True)
        chg = (g["lot_change_ts"].ne(g["lot_change_ts"].shift()) & g["lot_change_ts"].notna()).to_numpy().copy()
        chg[0] = False
        for i in np.where(chg)[0]:
            if i < 300 or i + LOT_WINDOW > len(g):
                continue
            step = g["fill_dev"].iloc[i:i + 300].mean() - g["fill_dev"].iloc[i - 300:i].mean()
            rows.append(dict(job_id=j, mold_id=g["mold_id"][0], step=float(step),
                             fired=bool(g["ewma_fill_fire"].iloc[i:i + LOT_WINDOW].any()),
                             shots_to_fire=float(np.argmax(g["ewma_fill_fire"].iloc[i:i + LOT_WINDOW].to_numpy()))
                             if g["ewma_fill_fire"].iloc[i:i + LOT_WINDOW].any() else np.nan))
    r = pd.DataFrame(rows)
    s, ns = r[r["step"].abs() >= STEP], r[r["step"].abs() < NO_STEP]
    return dict(lot_changes=int(len(r)), with_step=share(s["fired"]), without_step=share(ns["fired"]),
                between=int(len(r) - len(s) - len(ns)), median_shots_to_flag=float(s["shots_to_fire"].median()),
                window_shots=LOT_WINDOW, step_threshold_band=STEP, no_step_threshold_band=NO_STEP)


def ring(c, cd, rng):
    jobs = c.groupby("job_id").agg(mold_id=("mold_id", "first"), press_id=("press_id", "first"), leak=("leak", "mean"),
                                   n=("pos", "size"), start=("shot_ts", "min"),
                                   leak_start=("leak", lambda x: x.iloc[:500].mean()), leak_end=("leak", lambda x: x.iloc[-500:].mean()))
    first_def = cd.groupby("job_id")["hour_ts"].min()
    def window(j):
        g = c[c["job_id"] == j]
        t = first_def.get(j)
        end = int(np.searchsorted(g["shot_ts"].to_numpy(), np.datetime64(t))) if t is not None and pd.notna(t) else len(g)
        return g.iloc[:end], end < len(g)
    worn = jobs[jobs["leak"] > WORN]
    rows = []
    for j, w in worn.iterrows():
        g, has_def = window(j)
        f = np.where(g["cusum_pack_var_fire"].to_numpy())[0]
        rows.append(dict(job_id=j, mold_id=w["mold_id"], press_id=w["press_id"], shots=len(g), defect=has_def,
                         fired=len(f) > 0, lead=float(len(g) - f[0]) if len(f) and has_def else np.nan))
        # matched: unworn jobs on the same mold, the same number of shots from approval
        for jj in jobs[(jobs["mold_id"] == w["mold_id"]) & (jobs["leak"] < UNWORN) & (jobs["n"] >= len(g))].index:
            gg = c[c["job_id"] == jj].iloc[:len(g)]
            rows.append(dict(job_id=jj, mold_id=w["mold_id"], press_id=jobs.loc[jj, "press_id"], shots=len(gg), defect=np.nan,
                             fired=bool(gg["cusum_pack_var_fire"].any()), lead=np.nan, matched_to=j))
    r = pd.DataFrame(rows)
    e = r[r.get("matched_to").isna()] if "matched_to" in r else r
    m = r[r["matched_to"].notna()] if "matched_to" in r else r.iloc[:0]
    return dict(wear_episodes=int(len(e)), presses=sorted(e["press_id"].unique()), with_defect=int(e["defect"].sum()),
                fired=share(e["fired"]), fired_before_defect=share(e.loc[e["defect"] == True, "fired"]),
                lead=boot_median(e["lead"], rng), matched_unworn=share(m["fired"]), matched_windows=int(len(m)),
                worn_leak=WORN, unworn_leak=UNWORN,
                # the variance baseline is each job's first 500 shots; wear changes little within a job
                leak_change_within_job=float((jobs.loc[worn.index, "leak_end"] - jobs.loc[worn.index, "leak_start"]).median()),
                first_defect_in_first_hour=int((e["shots"] < 500).sum()),
                fires_per_10k_worn=float(c[c["job_id"].isin(worn.index)]["cusum_pack_var_fire"].sum() / jobs.loc[worn.index, "n"].sum() * 1e4),
                fires_per_10k_unworn=float(c[c["job_id"].isin(jobs[jobs["leak"] < UNWORN].index)]["cusum_pack_var_fire"].sum()
                                           / jobs[jobs["leak"] < UNWORN]["n"].sum() * 1e4))


def main():
    rng = np.random.default_rng(7)
    c, maint, cd = load()
    out = dict(vent=vent(c, maint, rng), lot_steps=lot_steps(c), ring=ring(c, cd, rng))
    (RESULTS / "drift_mechanisms.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
