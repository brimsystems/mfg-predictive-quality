"""
Generator: sort_dispositions, qc_audits, first_shot_approvals, scrap_tallies,
customer_returns
Source systems: QMS (reject-bin review, audits, approvals), packing scrap
tallies, customer returns log.

Every piece's quality is known from its shot; these are the shop's own
inspection processes over it, with their coverage and their noise:

  sort review      every shot the unit sorted is reviewed within the shift;
                   the inspector confirms the defective pieces and returns the
                   good ones (false rejects)
  hourly audit     a sample of 6 to 20 pieces pulled by the robot at a recorded
                   cycle (by hand, with no cycle, on the rest): weight and three
                   dimensions through the gauge, and a visual check whose
                   strictness varies by inspector
  first-shot       weight and dimensions per cavity at approval
  packing tally    defects packers find, by hour and code, with the cavity on
                   most; entered late, sometimes with the wrong code, the wrong
                   job when two jobs share a shift, or the adjacent hour
  returns          escaped defects that come back from the customer weeks later

Also returns quality engineering's root-cause register: every defective piece
with its root cause and where, if anywhere, it was caught.
"""
import numpy as np
import pandas as pd

from ..config import LABELS as L, MOLDS, RANDOM_SEED, SNAPSHOT_DATE
from .molding_cell import shift_of

INSPECTORS = {"A": ["QC-01", "QC-04"], "B": ["QC-02", "QC-05"], "C": ["QC-03", "QC-06"]}
# visual finding strictness per inspector on cosmetic codes; the newest are the most variable
STRICTNESS = {"QC-01": 1.0, "QC-02": 1.22, "QC-03": 0.82, "QC-04": 1.10, "QC-05": 0.75, "QC-06": 1.28}
COSMETIC = {"sink", "splay", "burn", "weld_line", "black_specks", "contamination", "gate_vestige", "flash", "warp"}
VISIBLE = {"short_shot": 0.98, "flash": 0.85, "sink": 0.65, "void": 0.10, "splay": 0.75, "burn": 0.85,
           "weld_line": 0.55, "warp": 0.45, "dimensional": 0.0, "black_specks": 0.70, "contamination": 0.60,
           "gate_vestige": 0.50, "other": 0.60}


def build_labels(cell: dict, parts: pd.DataFrame):
    rng = np.random.default_rng(RANDOM_SEED + 31)
    shots = cell["shots"]
    shots = shots[["shot_id", "press_id", "mold_id", "true_job_id", "shot_ts", "cycle_no", "sort_signal",
                   "in_production", "active_cavities"]].copy()
    shots["shot_ts"] = pd.to_datetime(shots["shot_ts"])
    sidx = shots.set_index("shot_id")
    defects = cell["defects"].merge(shots[["shot_id", "true_job_id", "shot_ts", "sort_signal", "press_id", "mold_id"]],
                                    on="shot_id", how="left")
    defects["defect_id"] = np.arange(1, len(defects) + 1)

    # ── Sort review ─────────────────────────────────────────────────────────
    sorted_shots = shots[(shots["sort_signal"] == "reject") & shots["in_production"]]
    d_sorted = defects[defects["sort_signal"] == "reject"]
    found = d_sorted[rng.random(len(d_sorted)) > L["review_miss"]]
    by_shot = found.groupby("shot_id").agg(n=("defect_code", "size"), codes=("defect_code", lambda s: ";".join(sorted(set(s)))))
    sd = sorted_shots[["shot_id", "shot_ts", "active_cavities"]].merge(by_shot, on="shot_id", how="left")
    sd["n"] = sd["n"].fillna(0).astype(int)
    rev = sd["shot_ts"] + pd.to_timedelta(rng.uniform(0.3, 6.5, len(sd)), unit="h")
    sort_disp = pd.DataFrame(dict(
        shot_id=sd["shot_id"], reviewed_ts=rev.dt.floor("min"),
        inspector_id=[rng.choice(INSPECTORS[shift_of(t)]) for t in sd["shot_ts"]],
        pieces_reviewed=sd["active_cavities"], pieces_confirmed_defective=sd["n"],
        defect_codes=sd["codes"], pieces_good=sd["active_cavities"] - sd["n"]))
    defects["caught_by"] = np.where(defects["defect_id"].isin(found["defect_id"]), "sort", None)

    # ── Audits ──────────────────────────────────────────────────────────────
    plan = cell["audit_plan"].merge(shots[["shot_id", "shot_ts", "cycle_no"]], on="shot_id")
    truth = cell["audits_truth"]
    shot_order = shots.sort_values("shot_id")["shot_id"].to_numpy()
    audit_rows, piece_rows, approvals = [], [], []
    dset = defects.set_index(["shot_id", "cavity"])
    audited_keys = set()
    tr = truth.set_index(["shot_id", "cavity"])
    counter = 1
    for p in plan.itertuples():
        m = MOLDS[p.mold_id]
        tol = m["tol_mm"]
        pos = np.searchsorted(shot_order, p.shot_id)
        need = p.sample_size
        cand = []
        k = 0
        while len(cand) < need and pos + k < len(shot_order) and k < 12:
            sid = int(shot_order[pos + k])
            for c in range(1, m["cavities"] + 1):
                if (sid, c) in tr.index:
                    cand.append((sid, c))
            k += 1
        cand = cand[:need]
        if not cand:
            continue
        insp = str(rng.choice(INSPECTORS[shift_of(p.shot_ts)]))
        strict = STRICTNESS[insp]
        robot = rng.random() < L["audit_robot_share"]
        aid = f"AU-{counter}" if not p.is_approval else f"FA-{counter}"
        counter += 1
        g_dim = tol * 2 * L["gauge_rr_frac"] / 6
        g_wt = m["weight_g"] * 0.0018
        codes_found = []
        rows_of = {sid: k for k, sid in enumerate(dict.fromkeys(sid for sid, _ in cand))}
        for sid, c in cand:
            t = tr.loc[(sid, c)]
            wt = float(t["weight_true"] + rng.normal(0, g_wt))
            dims = [float(t["dim_true"] + rng.normal(0, g_dim)), float(m["dim_mm"] * 0.62 + rng.normal(0, g_dim * 1.2)),
                    float(m["dim_mm"] * 0.31 + rng.normal(0, g_dim * 1.1))]
            code = None
            if (sid, c) in dset.index:
                dc = dset.loc[(sid, c)]
                dcode = dc["defect_code"] if isinstance(dc, pd.Series) else dc["defect_code"].iloc[0]
                vis = VISIBLE[dcode] * (strict if dcode in COSMETIC else 1.0)
                if dcode == "dimensional" or rng.random() < min(0.98, vis):
                    code = dcode
                    audited_keys.add((sid, c))
            elif rng.random() < 0.002 * strict:
                code = "other"                    # a strict inspector's call on a good piece
            if code:
                codes_found.append(code)
            piece_rows.append(dict(audit_id=aid, shot_id=sid, tray_row=rows_of[sid], cavity_id=c, part_weight_g=round(wt, 3),
                                   dimension_1=round(dims[0], 3), dimension_2=round(dims[1], 3),
                                   dimension_3=round(dims[2], 3), visual_result="reject" if code else "accept",
                                   defect_codes=code))
        row = dict(audit_id=aid, job_id=p.job_id, press_id=p.press_id, mold_id=p.mold_id,
                   audit_ts=(p.shot_ts + pd.Timedelta(minutes=float(rng.uniform(4, 25)))).floor("min"),
                   inspector_id=insp, sample_size=len(cand),
                   sampled_shot_cycle_no=int(p.cycle_no) if robot else None,
                   cavity_ids_sampled=";".join(str(c) for _, c in cand),
                   first_shot=bool(p.is_approval))
        if p.is_approval:
            approvals.append(row)
        else:
            audit_rows.append(row)
    audits = pd.DataFrame(audit_rows)
    pieces = pd.DataFrame(piece_rows)
    defects.loc[defects.set_index(["shot_id", "cavity"]).index.isin(list(audited_keys)) & defects["caught_by"].isna(),
                "caught_by"] = "audit"

    # audit chart rules: mean part weight per audit against limits from the mold x press's first 25 audits
    aw = pieces[pieces["audit_id"].str.startswith("AU")].groupby("audit_id")["part_weight_g"].mean()
    audits["mean_weight"] = audits["audit_id"].map(aw)
    viol = []
    for (mo, pr), g in audits.sort_values("audit_ts").groupby(["mold_id", "press_id"]):
        base = g["mean_weight"].head(25)
        mu, sdv = base.mean(), base.std()
        z = ((g["mean_weight"] - mu) / sdv).to_numpy()
        for i, ix in enumerate(g.index):
            f = []
            if abs(z[i]) > 3:
                f.append("WE1")
            if i >= 2 and ((z[i - 2:i + 1] > 2).sum() >= 2 or (z[i - 2:i + 1] < -2).sum() >= 2):
                f.append("WE2")
            if i >= 4 and ((z[i - 4:i + 1] > 1).sum() >= 4 or (z[i - 4:i + 1] < -1).sum() >= 4):
                f.append("WE4")
            if i >= 7 and ((z[i - 7:i + 1] > 0).all() or (z[i - 7:i + 1] < 0).all()):
                f.append("WE5")
            viol.append((ix, ";".join(f) if f else None))
    vmap = dict(viol)
    audits["audit_rule_violations"] = audits.index.map(vmap)
    audits["disposition"] = np.where(audits["audit_rule_violations"].notna(),
                                     rng.choice(["hold", "adjust", "continue"], len(audits), p=[0.2, 0.5, 0.3]), "continue")
    audits = audits.drop(columns=["mean_weight", "first_shot"])

    # first-shot approvals
    ap_run = cell["approvals"].set_index("job_id")
    fsa_rows = []
    for a in approvals:
        pc = pieces[pieces["audit_id"] == a["audit_id"]]
        r = ap_run.loc[a["job_id"]]
        fsa_rows.append(dict(job_id=a["job_id"], press_id=a["press_id"], mold_id=a["mold_id"],
                             approval_ts=r["approval_ts"], technician_id=r["technician_id"], inspector_id=a["inspector_id"],
                             shots_to_approval=int(r["shots_to_approval"]),
                             part_weight_g=";".join(f"{v:.3f}" for v in pc["part_weight_g"]),
                             dimension_values=";".join(f"{v:.3f}" for v in pc["dimension_1"]),
                             visual_result="accept", approved=True,
                             template_id_established=r["template_id_established"]))
    first_shot = pd.DataFrame(fsa_rows)
    pieces = pieces[pieces["audit_id"].str.startswith("AU")]
    # the extract links a piece to its shot only through the robot's recorded cycle and tray row
    piece_truth = pieces[["audit_id", "tray_row", "cavity_id", "shot_id"]]
    pieces = pieces.drop(columns=["shot_id"])

    # ── Packing tallies ─────────────────────────────────────────────────────
    left = defects[defects["caught_by"].isna()].copy()
    vis = left["defect_code"].map(VISIBLE).fillna(0.5) * L["pack_find_rate"] / 0.72
    tallied = rng.random(len(left)) < vis * 0.95
    defects.loc[left.index[tallied], "caught_by"] = "tally"
    t = left[tallied].copy()
    t["hour"] = t["shot_ts"].dt.floor("h")
    adj = rng.random(len(t)) < L["tally_adjacent_hour"]
    t.loc[adj, "hour"] = t.loc[adj, "hour"] + pd.to_timedelta(rng.choice([-1, 1], adj.sum()), unit="h")
    codes = list(VISIBLE)
    wrong = rng.random(len(t)) < L["tally_wrong_code"]
    t.loc[wrong, "defect_code"] = rng.choice(codes, wrong.sum())
    t["cavity_id"] = np.where(rng.random(len(t)) < L["tally_cavity_share"], t["cavity"], None)
    # wrong job when two jobs share a shift on the press (the other job of that shift)
    t["shift_date"] = (t["shot_ts"] - pd.Timedelta(hours=6)).dt.normalize()
    t["shift"] = [shift_of(x) for x in t["shot_ts"]]
    jobs_in_shift = shots.assign(shift_date=(shots["shot_ts"] - pd.Timedelta(hours=6)).dt.normalize(),
                                 shift=[shift_of(x) for x in shots["shot_ts"]])
    js = jobs_in_shift.groupby(["press_id", "shift_date", "shift"])["true_job_id"].unique()
    t["job_id"] = t["true_job_id"]
    wj = rng.random(len(t)) < L["tally_wrong_job"] * 3
    for i in np.where(wj)[0]:
        key = (t["press_id"].iat[i], t["shift_date"].iat[i], t["shift"].iat[i])
        others = [j for j in js.get(key, []) if j != t["true_job_id"].iat[i]]
        if others:
            t.iloc[i, t.columns.get_loc("job_id")] = others[0]
    tallies = (t.groupby(["job_id", "press_id", "shift_date", "shift", "hour", "cavity_id", "defect_code"], dropna=False)
               .size().rename("qty").reset_index())
    tallies["hour_bucket"] = tallies["hour"].dt.hour
    tallies["entered_by"] = [f"PK-{int(x):02d}" for x in rng.integers(1, 9, len(tallies))]
    tallies["entered_ts"] = (tallies["hour"] + pd.Timedelta(hours=1)
                             + pd.to_timedelta(rng.uniform(0, 6, len(tallies)), unit="h")).dt.floor("min")
    tallies.insert(0, "tally_id", [f"TL-{i + 1}" for i in range(len(tallies))])
    tallies = tallies.drop(columns=["hour"])

    # ── Customer returns ────────────────────────────────────────────────────
    esc = defects[defects["caught_by"].isna()].copy()
    ret = esc[rng.random(len(esc)) < L["return_rate"]].copy()
    ret["received"] = ret["shot_ts"] + pd.to_timedelta(rng.uniform(20, 90, len(ret)), unit="D")
    ret = ret[ret["received"] <= pd.Timestamp(SNAPSHOT_DATE)]
    defects.loc[ret.index, "caught_by"] = "return"
    ret["date_code"] = ret["shot_ts"].dt.strftime("%y%W")
    part_of = {m: MOLDS[m]["part"] for m in MOLDS}
    cust_of = {m: MOLDS[m]["customer"] for m in MOLDS}
    ret["part_id"] = ret["mold_id"].map(part_of)
    ret["customer_id"] = ret["mold_id"].map(cust_of)
    ret["received_date"] = (ret["received"] + pd.to_timedelta(rng.integers(0, 7, len(ret)), unit="D")).dt.normalize()
    returns = (ret.groupby(["customer_id", "part_id", "date_code", "true_job_id", "cavity", "defect_code"])
               .agg(qty=("defect_id", "size"), received_date=("received_date", "max")).reset_index())
    unattr = rng.random(len(returns)) < L["return_unattributed"]
    returns["attributed_job_id"] = np.where(unattr, None, returns["true_job_id"])
    returns["attributed_cavity_id"] = np.where(unattr, None, returns["cavity"])
    returns.insert(0, "return_id", [f"RT-{i + 1}" for i in range(len(returns))])
    returns = returns.drop(columns=["true_job_id", "cavity"])

    register = defects.rename(columns={"true_job_id": "job_id"})[
        ["defect_id", "shot_id", "cavity", "job_id", "press_id", "mold_id", "shot_ts", "defect_code",
         "root_cause_code", "caught_by"]]
    return dict(sort_dispositions=sort_disp, qc_audits=audits, qc_audit_pieces=pieces,
                first_shot_approvals=first_shot, audit_piece_links=piece_truth, scrap_tallies=tallies, customer_returns=returns,
                register=register)
