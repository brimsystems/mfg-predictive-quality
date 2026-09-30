"""
Realism checks on a generation run (checks 1 to 7, 10 and 11).

Reads the raw extracts and the reference register, prints each measure against
its target range, and writes reference/checks.json. Usage:

  python -m data_source.generate.checks
"""
import json
import time

import duckdb
import numpy as np
import pandas as pd

from .config import CELL, MOLDS, RAW_DIR, REFERENCE_DIR

RESULTS = []


def check(group, name, value, lo=None, hi=None, fmt="{:.3f}"):
    ok = None if lo is None else bool(lo <= value <= hi)
    RESULTS.append(dict(group=group, name=name, value=float(value), lo=lo, hi=hi, passed=ok))
    rng = "" if lo is None else f"[{fmt.format(lo)}, {fmt.format(hi)}]"
    flag = "" if ok is None else ("ok " if ok else "OUT")
    print(f"  {flag:<4}{name:<62} {fmt.format(value):>10}  {rng}")


def main():
    t0 = time.time()
    con = duckdb.connect()
    raw, ref = RAW_DIR.as_posix(), REFERENCE_DIR.as_posix()
    con.execute(f"create view summary as select * from read_parquet('{raw}/monitoring/cavity_shot_summary/*/*.parquet', hive_partitioning=true)")
    con.execute(f"create view machine as select * from read_parquet('{raw}/mes/machine_shot_data/*/*.parquet', hive_partitioning=true)")
    con.execute(f"create view curves as select * from read_parquet('{raw}/monitoring/cavity_curves/*/*/*.parquet', hive_partitioning=true)")
    con.execute(f"create view state as select * from read_parquet('{ref}/shot_state/*/*.parquet', hive_partitioning=true)")
    shots = pd.read_parquet(REFERENCE_DIR / "shot_truth.parquet")
    reg = pd.read_parquet(REFERENCE_DIR / "root_cause_register.parquet")
    runlog = pd.read_parquet(REFERENCE_DIR / "run_log.parquet")
    changes = pd.read_parquet(REFERENCE_DIR / "change_truth.parquet")
    tmpl = pd.read_csv(RAW_DIR / "monitoring" / "templates.csv")
    maint = pd.read_csv(RAW_DIR / "toolroom" / "mold_maintenance.csv", parse_dates=["event_ts"])
    prod = shots[shots["in_production"]]

    # ── 1. Curve shapes ─────────────────────────────────────────────────────
    print("1. Curve shapes")
    arr = con.execute("""
        with c as (select shot_id, sensor_id, min(t_ms) filter (where pressure_bar > 10) as arrive_ms,
                          max(pressure_bar) as peak, arg_max(t_ms, pressure_bar) as t_peak,
                          max(t_ms) filter (where pressure_bar > 15) as last_ms
                   from curves where shot_id % 7 = 0 group by 1, 2)
        select c.*, s.sensor_position, s.mold_id from c join summary s using (shot_id, sensor_id)""").df()
    arr[["arrive_ms", "peak", "t_peak", "last_ms"]] = arr[["arrive_ms", "peak", "t_peak", "last_ms"]].astype(float)
    ratios, pk = [], []
    for m, mm in MOLDS.items():
        a = arr[arr["mold_id"] == m]
        pg = a[a["sensor_position"] == "post_gate"].groupby("shot_id")[["arrive_ms", "peak"]].mean()
        eo = a[a["sensor_position"] == "end_of_fill"].groupby("shot_id")[["arrive_ms", "peak"]].mean()
        j = pg.join(eo, lsuffix="_pg", rsuffix="_eo", how="inner")
        if len(j):
            j = j.dropna()
            ratios.append(np.median(j["arrive_ms_eo"] / j["arrive_ms_pg"]) / mm["flow_ratio"])
            pk.append(np.mean(j["peak_eo"] < j["peak_pg"]))
    check("1", "end-of-fill / post-gate arrival over the flow-length ratio", np.median(ratios), 0.85, 1.15)
    check("1", "end-of-fill peak below post-gate peak (share of shots)", np.mean(pk), 0.95, 1.0)
    check("1", "pressure held after the peak (last reading > 15 bar after peak, share)",
          float(np.mean(arr["last_ms"] > arr["t_peak"] + 500)), 0.95, 1.0)

    # ── 2. Unit state agreement ─────────────────────────────────────────────
    print("2. Recorded alarm state against recomputation")
    check("2", "shots where the recorded state agrees", float(np.mean(shots["alarm_state"] == shots["recomputed_alarm_state"])), 0.99, 1.0)

    # ── 3. Section 4 targets ────────────────────────────────────────────────
    print("3. Target ranges")
    pieces = prod["active_cavities"].sum()
    rp = reg[reg["shot_id"].isin(prod["shot_id"])]
    check("3", "piece defect rate before sorting", len(rp) / pieces, 0.018, 0.030)
    sorted_ids = set(prod.loc[prod["sort_signal"] == "reject", "shot_id"])
    check("3", "shots sorted", len(sorted_ids) / len(prod), 0.005, 0.010)
    check("3", "defective pieces on sorted shots (sort removal)", rp["shot_id"].isin(sorted_ids).mean(), 0.30, 0.40)
    sd = pd.read_csv(RAW_DIR / "qms" / "sort_dispositions.csv")
    check("3", "false rejects: sorted shots found good at review", float(np.mean(sd["pieces_confirmed_defective"] == 0)), 0.25, 0.40)
    conf = rp[rp["caught_by"].notna()]
    sh = conf["root_cause_code"].value_counts(normalize=True)
    for g, lo, hi in [("G1", .15, .22), ("G2", .10, .15), ("G3", .15, .20), ("G4", .08, .12), ("G8", .06, .10),
                      ("G10", .03, .05), ("G0", .15, .25)]:
        check("3", f"confirmed defects from {g}", sh.get(g, 0.0), lo, hi)
    check("3", "confirmed defects from G5 to G7", sh.get("G5", 0) + sh.get("G6", 0) + sh.get("G7", 0), 0.08, 0.14)

    apt = pd.read_parquet(REFERENCE_DIR / "audit_piece_truth.parquet")
    # summary values relative to the template in force (the weight and dimension model's terms)
    tp = tmpl[["template_id", "sensor_id", "pack_integral_template_value", "gate_seal_time_s_template_value"]]
    pg = con.execute("""select shot_id, sensor_id, mold_id, press_id, pack_integral, gate_seal_time_s
                        from summary where sensor_position = 'post_gate'""").df()
    pg = pg.merge(shots[["shot_id", "template_id"]], on="shot_id").merge(tp, on=["template_id", "sensor_id"])
    pg["pack"] = pg["pack_integral"] / pg["pack_integral_template_value"] - 1
    pg["gs"] = pg["gate_seal_time_s"] / pg["gate_seal_time_s_template_value"] - 1
    pg = pg.groupby(["shot_id", "mold_id", "press_id"])[["pack", "gs"]].mean().reset_index()
    ap = apt.merge(pg, on="shot_id").dropna(subset=["weight_true", "dim_true", "pack", "gs"])
    for c in ("weight_true", "dim_true"):
        ap[c] = ap[c] - ap.groupby(["mold_id", "press_id", "cavity"])[c].transform("mean")
    cors, r2s = [], []
    for _, g in ap.groupby(["mold_id", "press_id"]):
        if len(g) < 200:
            continue
        cors.append(np.corrcoef(g["weight_true"], g["pack"])[0, 1])
        X = np.column_stack([np.ones(len(g)), g["pack"], g["gs"]])
        b, *_ = np.linalg.lstsq(X, g["dim_true"], rcond=None)
        r2s.append(1 - np.var(g["dim_true"] - X @ b) / np.var(g["dim_true"]))
    check("3", "weight vs pack integral correlation (median mold x press)", np.median(cors), 0.75, 0.90)
    check("3", "dimension R2 on pack integral and gate seal (median)", np.median(r2s), 0.65, 0.85)

    # G2: end-of-fill CUSUM signal against the burn rise, per cleaning interval
    eof = con.execute("""select s.shot_id, s.mold_id, s.shot_ts, avg(s.end_of_fill_pressure_bar) eof
                         from summary s where sensor_position = 'end_of_fill' group by 1, 2, 3 order by 3""").df()
    eof = eof.merge(shots[["shot_id", "template_id", "in_production"]], on="shot_id")
    tv = (tmpl.groupby("template_id")["end_of_fill_pressure_bar_template_value"].mean())
    eof["dev"] = eof["eof"] / eof["template_id"].map(tv) - 1
    burns = reg[reg["defect_code"] == "burn"].groupby("shot_id").size()
    eof["burn"] = eof["shot_id"].map(burns).fillna(0)
    leads, n_int, n_lead = [], 0, 0
    for m, g in eof.groupby("mold_id"):
        cl = maint[(maint["mold_id"] == m) & (maint["event_type"] == "vent_cleaning")]["event_ts"].sort_values()
        edges = [pd.Timestamp.min] + list(cl) + [pd.Timestamp.max]
        for a, b in zip(edges[:-1], edges[1:]):
            iv = g[(g["shot_ts"] > a) & (g["shot_ts"] <= b) & g["in_production"]].reset_index(drop=True)
            if len(iv) < 5000:
                continue
            n_int += 1
            # the platform's CUSUM: 25-shot block means, baseline and reset at each template (run)
            sig = None
            for tid, seg in iv.groupby("template_id", sort=False):
                d = seg["dev"].to_numpy()
                bm = pd.Series(d).rolling(25, min_periods=12).mean().to_numpy()
                ref = bm[25:500:25]
                ref = ref[np.isfinite(ref)]
                if len(ref) < 4:
                    continue
                z = (bm - ref.mean()) / (ref.std() or 1.0)
                c = 0.0
                for i in range(24, len(z), 25):
                    if not np.isfinite(z[i]):
                        continue
                    c = max(0.0, c - z[i] - 0.375)
                    if c > 5:
                        sig = int(seg.index[i])
                        break
                if sig is not None:
                    break
            roll = iv["burn"].rolling(1000, min_periods=1000).sum().to_numpy()
            b0 = np.nanmean(roll[:3000]) if np.isfinite(roll[:3000]).any() else 0
            rise = np.where(roll > max(3.0, 3 * b0))[0]
            if sig is not None and len(rise):
                leads.append(int(rise[0]) - sig)
                n_lead += int(rise[0]) - sig > 0
    check("3", "G2 CUSUM lead over the burn rise, shots (median)", np.median(leads) if leads else np.nan, 2000, 6000, "{:.0f}")
    check("3", "cleaning intervals with a positive lead", n_lead / max(n_int, 1), 0.70, 1.0)

    # tampering followed by a new confirmed code within four hours
    t = changes[changes["is_tamper"] == True]
    creg = reg[reg["caught_by"].notna()]
    hits = 0
    for c in t.itertuples():
        j = creg[creg["job_id"] == c.job_id]
        before = set(j[(j["shot_ts"] >= c.change_ts - pd.Timedelta(hours=4)) & (j["shot_ts"] < c.change_ts)]["defect_code"])
        after = j[(j["shot_ts"] >= c.change_ts) & (j["shot_ts"] < c.change_ts + pd.Timedelta(hours=4))]
        new = after[~after["defect_code"].isin(before)]["defect_code"].value_counts()
        hits += bool((new >= 3).any())
    check("3", "tampering followed by a new code within 4 hours", hits / max(len(t), 1), 0.30, 0.45)

    # ── 4. Mechanism signatures ─────────────────────────────────────────────
    print("4. Mechanism signatures")
    st = con.execute("""select st.shot_id, st.leak, st.dT, st.blocked, st.vent from state st""").df()
    pgs = con.execute("""select shot_id, mold_id, press_id, shot_ts, avg(pack_integral) pack, avg(fill_integral) fill,
                                avg(gate_seal_time_s) gs, avg(cooling_rate) cool
                         from summary where sensor_position = 'post_gate' group by all""").df()
    pgs = pgs.merge(shots[["shot_id", "template_id", "in_production", "true_job_id"]], on="shot_id").merge(st, on="shot_id")
    mach = con.execute("select shot_id, cushion_mm from machine").df()
    pgs = pgs.merge(mach, on="shot_id")
    pgs["pack_n"] = pgs["pack"] / pgs.groupby("template_id")["pack"].transform("median")
    im12 = pgs[(pgs["press_id"] == "IM-12") & pgs["in_production"]]
    es = pd.read_csv(RAW_DIR / "toolroom" / "equipment_service.csv", parse_dates=["event_ts"])
    ring = es[(es["press_id"] == "IM-12") & (es["equipment"] == "check_ring") & (es["event_ts"] >= "2025-01-01")]["event_ts"].min()
    pre = im12[(im12["shot_ts"] < ring) & (im12["shot_ts"] > ring - pd.Timedelta(days=60))]
    post = im12[(im12["shot_ts"] > ring) & (im12["shot_ts"] < ring + pd.Timedelta(days=60))]
    if len(pre) and len(post):
        def rsd(x):
            # robust spread of the shot-to-shot pack integral, run level removed
            r = (x["pack_n"] - x.groupby("template_id")["pack_n"].transform("median")).dropna()
            return 1.4826 * float(np.median(np.abs(r - np.median(r))))
        check("4", "IM-12 pack integral spread ratio (robust), 60 days before / after ring", rsd(pre) / rsd(post), 1.3, 5.0)
        check("4", "IM-12 cushion sd ratio before / after ring", pre["cushion_mm"].std() / post["cushion_mm"].std(), 1.3, 10.0)
        check("4", "IM-12 pack integral mean shift before / after ring (abs)", abs(pre["pack_n"].mean() - post["pack_n"].mean()), 0, 0.02)
    p = pgs[pgs["in_production"]]
    p = p.assign(gs_n=p["gs"] / p.groupby("template_id")["gs"].transform("median"))
    p = p.assign(dT_n=p["dT"] - p.groupby("template_id")["dT"].transform("median"))
    check("4", "gate seal time vs mold temperature, both relative to the template (corr)", np.corrcoef(p["gs_n"].fillna(1), p["dT_n"])[0, 1], 0.2, 1.0)
    blk = p[p["blocked"] > 0]["true_job_id"].unique()
    if len(blk):
        steps = []
        for j in blk:
            g = p[p["true_job_id"] == j]
            steps.append(abs(g[g["blocked"] > 0]["fill"].median() / g[g["blocked"] == 0]["fill"].median() - 1))
        check("4", "cavity block step in fill integral (median abs)", np.median(steps), 0.03, 0.5)
    ld = pd.read_parquet(REFERENCE_DIR / "load_truth.parquet").merge(pd.read_csv(RAW_DIR / "materials" / "material_loads.csv"), on="load_id")

    # ── 5. Sensor realism ───────────────────────────────────────────────────
    print("5. Sensor realism")
    sr = con.execute("""select avg((pack_integral is null)::int) drop_rate,
                               avg((template_match_score < 0.5)::int) spike_like from summary""").df().iloc[0]
    check("5", "summary rows dropped (one sensor missing)", sr["drop_rate"], 0.002, 0.004, "{:.4f}")
    ac = []
    for tid, g in pgs[pgs["in_production"]].groupby("template_id"):
        x = g.sort_values("shot_ts")["pack"].dropna().to_numpy()
        if len(x) > 2000:
            r = x - pd.Series(x).rolling(301, center=True, min_periods=50).median().to_numpy()
            ac.append(pd.Series(r).autocorr(1))
    check("5", "lag-1 autocorrelation of pack integral residual (median)", np.median(ac), 0.45, 0.85)

    # ── 6. Labels ───────────────────────────────────────────────────────────
    print("6. Labels")
    au = pd.read_csv(RAW_DIR / "qms" / "qc_audits.csv")
    check("6", "audits with the sampled cycle recorded", au["sampled_shot_cycle_no"].notna().mean(), 0.85, 0.90)
    tl = pd.read_csv(RAW_DIR / "qms" / "scrap_tallies.csv")
    check("6", "tally rows with a cavity", tl["cavity_id"].notna().mean(), 0.60, 0.70)
    rt = pd.read_csv(RAW_DIR / "qms" / "customer_returns.csv")
    check("6", "returns never attributed", rt["attributed_job_id"].isna().mean(), 0.15, 0.20)
    pcs = pd.read_csv(RAW_DIR / "qms" / "qc_audit_pieces.csv")
    check("6", "audits (count)", len(au), None, None, "{:.0f}")
    check("6", "audit pieces (count)", len(pcs), None, None, "{:.0f}")
    linked = reg[reg["caught_by"].isin(["sort", "audit"])]["defect_code"].value_counts()
    for code, n in linked.items():
        check("6", f"confirmed defects with shot linkage: {code}", n, None, None, "{:.0f}")

    # ── 7. Tampering ────────────────────────────────────────────────────────
    print("7. Tampering")
    mid = changes[changes["shot_index"].notna()]
    check("7", "tampering share of mid-run changes", mid["is_tamper"].mean(), 0.25, 0.40)

    # ── 10. Story consistency ───────────────────────────────────────────────
    print("10. Story consistency")
    lots = pd.read_csv(RAW_DIR / "materials" / "resin_lots.csv")
    ldl = ld.merge(lots[["lot_id", "supplier_id"]], left_on="resin_lot_id", right_on="lot_id").sort_values("actual_ts")
    ldl["prev_lot"] = ldl.groupby("press_id")["resin_lot_id"].shift()
    ldl["prev_sup"] = ldl.groupby("press_id")["supplier_id"].shift()
    ch = ldl[(ldl["resin_lot_id"] != ldl["prev_lot"]) & ldl["prev_lot"].notna()]
    fl = pgs[pgs["in_production"]].sort_values("shot_ts")
    steps = []
    for c in ch.itertuples():
        g = fl[(fl["press_id"] == c.press_id)]
        i = g["shot_ts"].searchsorted(c.actual_ts)
        before, after = g.iloc[max(0, i - 150):i], g.iloc[i:i + 150]
        if len(before) == 150 and len(after) == 150 and before["true_job_id"].iat[0] == after["true_job_id"].iat[-1]:
            steps.append((c.supplier_id == "RS-3", c.prev_sup == "RS-3",
                          after["fill"].mean() / before["fill"].mean() - 1))
    st_ = pd.DataFrame(steps, columns=["to_rs3", "from_rs3", "step"])
    if len(st_):
        into = st_[st_["to_rs3"] & ~st_["from_rs3"]]["step"].mean()
        other = st_[~st_["to_rs3"] & ~st_["from_rs3"]]["step"].mean()
        check("10", "fill integral step at a load into an RS-3 lot, minus other lot changes", into - other, 0.005, 0.2, "{:.4f}")
    splay = reg[reg["defect_code"] == "splay"].merge(shots[["shot_id", "press_id"]], on=["shot_id", "press_id"])
    splay = pd.merge_asof(splay.sort_values("shot_ts"), ldl[["actual_ts", "press_id", "dryer_id"]].rename(columns={"actual_ts": "shot_ts"}),
                          on="shot_ts", by="press_id")
    summer = splay[(splay["shot_ts"] >= CELL["faulty_from"]) & (splay["shot_ts"] <= CELL["faulty_to"])]
    check("10", "summer splay on the faulty dryer (share)", (summer["dryer_id"] == CELL["faulty_dryer"]).mean(), 0.5, 1.0)
    m18 = runlog[runlog["mold_id"] == "M-2118"]
    if len(m18):
        conv = []
        for r in m18.itertuples():
            g = con.execute(f"""select st.g3 from state st join (select shot_id from summary where job_id = '{r.job_id}'
                                 and sensor_id like 'M-2118-S1') s using (shot_id) order by shot_id""").df()["g3"].to_numpy()
            conv.append((r.technician_id, r.tenure_days, int(np.argmax(np.abs(g) < 0.005)) if len(g) else np.nan))
        cv = pd.DataFrame(conv, columns=["tech", "tenure", "shots"])
        print(cv.groupby("tech")[["tenure", "shots"]].median().to_string())

    # ── 11. Volume ──────────────────────────────────────────────────────────
    print("11. Volume")
    size = sum(f.stat().st_size for f in RAW_DIR.rglob("*.parquet")) / 1e9
    check("11", "Parquet size, GB", size, 0, 6.0)
    n_shots = len(shots)
    check("11", "shots (count)", n_shots, None, None, "{:.0f}")
    check("11", "curve rows (count)", con.execute("select count(*) from curves").fetchone()[0], None, None, "{:.0f}")

    ok = [r for r in RESULTS if r["passed"] is not None]
    print(f"\n{sum(r['passed'] for r in ok)} of {len(ok)} ranged checks pass ({time.time() - t0:.0f} s)")
    (REFERENCE_DIR / "checks.json").write_text(json.dumps(RESULTS, indent=1, default=str))


if __name__ == "__main__":
    main()
