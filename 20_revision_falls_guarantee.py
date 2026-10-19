"""
STEP 20 (revision) - How far does the fall-alarm guarantee reach?

Usage: python 20_revision_falls_guarantee.py split | har70

split  Exchangeability by design. In script 10 the calibration scores of the training people are out-of-fold scores of
       inner-CV models, whereas the held-out person is scored by the model trained on all training people. Here the
       training people of every leave-one-person-out fold are split once at random into a proper-training set (6
       people) and a calibration set (10 people); the deployed model is trained on the proper-training set only and
       the SAME model scores the calibration people and the held-out person, so the conformal argument applies
       exactly (n = 10, quantile index 10 = the largest calibration threshold).

har70  Older adults as an ADL-only source. The fall kit of the paper (waist + wrist IMU) cannot be applied to HAR70+
       (lower-back and thigh accelerometers, no gyroscope), so a waist-accelerometer-magnitude detector (orientation
       independent; CARES = GBT on the 22 magnitude descriptors + LR on MiniRocket-lite of the magnitude) is trained on
       the young volunteers and streamed over the free-living recordings of the 18 adults aged 70-95 years
       (resampled to 25 Hz, converted from g to m/s^2; a window every 0.4 s, 30-s refractory period). Every alert on
       HAR70+ is a false alarm. Two calibration sources are compared:
         young   - conformal threshold from the 17 young volunteers (as in the paper), applied to older adults;
         older   - conformal threshold from older adults' free-living data only (leave-one-older-adult-out), whose
                   fall sensitivity is then measured on the volunteers' simulated falls (held-out-person scores).
Outputs: TAB_PATH/T6b_falls_split_conformal.csv, S18_falls_split_per_subject.csv,
         TAB_PATH/T6c_falls_older_adults.csv, S19_falls_older_adults_per_subject.csv
"""
import importlib
import os
import pickle
import sys
import time
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import resample_poly
from sklearn.model_selection import GroupKFold

from config import PRED_PATH, TAB_PATH, DATA_PATH, RAW, N_JOBS, SEED, ALARM
from lib.alarms import alarms, conformal_threshold, THRESH_GRID
from lib.features import extract, signal_stats
from lib.minirocket import MiniRocketLite
from lib.zoo import CARES, make_rocket

A = importlib.import_module("10_calibrated_alarms")
DELTA = ALARM["delta"]
REFR = ALARM["refractory_s"]["falls"]
BUDGETS = [0.0, 2.0]
G = 9.80665


def theta_from_rates(rates, B):
    ok = rates <= B + 1e-12
    adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
    return THRESH_GRID[np.where(adm)[0][0]] if adm.any() else 1.0


def eval_person(Mt, p, thr):
    tr = A.falls_trial_eval(Mt, p, thr, REFR)
    f = tr[tr.fall]
    fa = tr.false_alarms.sum()
    h = tr.adl_hours.sum()
    return dict(fa=int(fa), adl_hours=h, fa_per_h=fa / h, falls=len(f), detected=int(f.detected.sum()))


# ============================================================================================== split conformal
def split_fold(F, R, M, s):
    cache = os.path.join(PRED_PATH, "cache", f"split_falls_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    subs = np.array(sorted(set(M.subject) - {s}))
    rng = np.random.default_rng(SEED + int(s))
    cal = np.sort(rng.choice(subs, size=10, replace=False))
    prop = np.setdiff1d(subs, cal)
    subj = M.subject.to_numpy()
    trn = np.isin(subj, prop) & ((M.pos == 1) | (M.adl == 1)).to_numpy()
    y = M.pos.to_numpy()
    mr = make_rocket("falls", R[trn][::4], seed=SEED)
    use = np.isin(subj, np.r_[cal, [s]])
    Mf = np.zeros((len(M), 0), np.float32)
    Mf_use = mr.transform(R[use])
    Mtr = mr.transform(R[trn])
    m = CARES(SEED).fit(F[trn], Mtr, y[trn], 2)
    p_use = m.predict_proba(F[use], Mf_use)[:, 1]
    out = dict(subject=s, cal=cal, prop=prop, idx=np.where(use)[0], p=p_use.astype(np.float32))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    print("split fold done", s, flush=True)
    return out


def run_split():
    t0 = time.time()
    X, M = A.stream_windows()
    sens = {"waist_acc": [0, 1, 2], "waist_gyr": [3, 4, 5], "wrist_acc": [6, 7, 8], "wrist_gyr": [9, 10, 11]}
    F, _ = extract(X, 25.0, sens)
    R = np.stack([np.linalg.norm(X[:, :, c:c + 3], axis=2) for c in (0, 3, 6, 9)], -1).astype(np.float32)
    del X
    print("features", F.shape, "%.1f min" % ((time.time() - t0) / 60), flush=True)
    subs = np.unique(M.subject)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(split_fold)(F, R, M, s) for s in subs)
    rows = []
    for r in res:
        s = r["subject"]
        p = np.zeros(len(M)); p[r["idx"]] = r["p"]
        thetas = {B: [] for B in BUDGETS}
        for j in r["cal"]:
            jj = (M.subject == j).to_numpy()
            rates = A.adl_fa_rates(M[jj], p[jj], THRESH_GRID, REFR)
            for B in BUDGETS:
                thetas[B].append(theta_from_rates(rates, B))
        te = (M.subject == s).to_numpy()
        for B in BUDGETS:
            n = len(thetas[B]); rr = int(np.ceil((n + 1) * (1 - DELTA)))
            thr = min(conformal_threshold(thetas[B], DELTA), 1.0)
            ev = eval_person(M[te], p[te], thr)
            rows.append(dict(policy="split conformal (exact exchangeability)", budget_per_h=B, subject=s, n_cal=n, quantile_index=rr,
                             threshold=thr, within_budget=ev["fa_per_h"] <= B + 1e-9, **ev))
    Rr = pd.DataFrame(rows)
    Rr.to_csv(os.path.join(TAB_PATH, "S18_falls_split_per_subject.csv"), index=False)
    summ = Rr.groupby(["policy", "budget_per_h"]).apply(lambda g: pd.Series(dict(
        people=len(g), covered=int(g.within_budget.sum()), coverage=g.within_budget.mean(),
        pooled_fa_per_h=g.fa.sum() / g.adl_hours.sum(), total_fa=int(g.fa.sum()), adl_hours=g.adl_hours.sum(),
        people_with_fa=int((g.fa > 0).sum()), fall_sensitivity=g.detected.sum() / g.falls.sum(),
        median_threshold=g.threshold.median(), n_cal=int(g.n_cal.iloc[0]), quantile_index=int(g.quantile_index.iloc[0])))).reset_index()
    summ.to_csv(os.path.join(TAB_PATH, "T6b_falls_split_conformal.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


# ============================================================================================== HAR70+ transfer
def mag_views(Xacc, fs=25.0):
    """Xacc: (N, L, 3) waist/lower-back acceleration in m/s^2 -> 22 descriptors of the magnitude and the magnitude."""
    mag = np.linalg.norm(Xacc, axis=2).astype(np.float32)
    D = np.concatenate([signal_stats(mag[a:a + 4000][:, None, :], fs).reshape(-1, 22) for a in range(0, len(mag), 4000)])
    D[~np.isfinite(D)] = 0.0
    return D, mag[:, :, None]


def har_streams():
    """Continuous lower-back acceleration of every HAR70+ person: 25 Hz, m/s^2, 4-s windows every 0.4 s."""
    z = zipfile.ZipFile(os.path.join(DATA_PATH, RAW["har70"]))
    Xs, meta = [], []
    for n in sorted(x for x in z.namelist() if x.endswith(".csv")):
        sid = int(os.path.basename(n)[:-4])
        d = pd.read_csv(z.open(n))
        tt = pd.to_datetime(d["timestamp"]).astype("int64").to_numpy() / 1e9
        sig = d[["back_x", "back_y", "back_z"]].to_numpy(np.float64) * G
        cut = np.where(np.diff(tt) > 1.0)[0] + 1
        for segi, (a, b) in enumerate(zip(np.r_[0, cut], np.r_[cut, len(tt)])):
            s25 = resample_poly(sig[a:b], 1, 2, axis=0).astype(np.float32)
            if len(s25) < 100:
                continue
            for k in range(0, len(s25) - 100 + 1, 10):
                Xs.append(s25[k:k + 100])
                meta.append((sid, segi, (k + 100) / 25.0))
    X = np.stack(Xs)
    Mh = pd.DataFrame(meta, columns=["subject", "segment", "t_end"])
    return X, Mh


def har_fa_rates(Mh_j, p_j, grid):
    hours = 0.0
    counts = np.zeros(len(grid))
    for _, g in Mh_j.assign(p=p_j).groupby("segment"):
        te, pp = g.t_end.to_numpy(), g.p.to_numpy()
        hours += (te.max() - te.min() + 4.0) / 3600.0
        for k, th in enumerate(grid):
            if pp.max() < th:
                break
            counts[k] += len(alarms(te, pp, th, REFR))
    return counts / hours, counts, hours


def mag_fold(D, Rm, M, s):
    """Held-out-person scores and inner OOF scores of the magnitude detector (same protocol as script 10)."""
    cache = os.path.join(PRED_PATH, "cache", f"magfall_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    te = (M.subject == s).to_numpy()
    keep = ((M.pos == 1) | (M.adl == 1)).to_numpy()
    y = M.pos.to_numpy()
    trn = (~te) & keep
    mr = MiniRocketLite(dilations=(1, 2, 4, 8), n_bias=1, seed=SEED).fit(Rm[trn][::4])
    MF = mr.transform(Rm)
    m = CARES(SEED).fit(D[trn], MF[trn], y[trn], 2)
    out = dict(subject=s, idx_te=np.where(te)[0], p_te=m.predict_proba(D[te], MF[te])[:, 1].astype(np.float32))
    pickle.dump(out, open(cache, "wb"))
    print("mag fold done", s, flush=True)
    return out


def run_har70():
    t0 = time.time()
    X, M = A.stream_windows()
    D, Rm = mag_views(X[:, :, 0:3])
    del X
    subs = np.unique(M.subject)
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(mag_fold)(D, Rm, M, s) for s in subs)
    p_young = np.zeros(len(M))
    for r in res:
        p_young[r["idx_te"]] = r["p_te"]
    print("young LOPO done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # final detector trained on all 17 volunteers
    keep = ((M.pos == 1) | (M.adl == 1)).to_numpy()
    y = M.pos.to_numpy()
    mr = MiniRocketLite(dilations=(1, 2, 4, 8), n_bias=1, seed=SEED).fit(Rm[keep][::4])
    final = CARES(SEED).fit(D[keep], mr.transform(Rm[keep]), y[keep], 2)
    Xh, Mh = har_streams()
    Dh, Rh = mag_views(Xh)
    del Xh
    p_old = np.concatenate([final.predict_proba(Dh[a:a + 20000], mr.transform(Rh[a:a + 20000]))[:, 1] for a in range(0, len(Dh), 20000)])
    print("HAR70+ windows", len(Mh), "%.1f min" % ((time.time() - t0) / 60), flush=True)
    np.savez_compressed(os.path.join(PRED_PATH, "stream_har70_falls_detector.npz"), p=p_old, **{c: Mh[c].to_numpy() for c in Mh.columns})
    # per-person false-alarm rate curves
    young_rates = {}
    for s in subs:
        jj = (M.subject == s).to_numpy()
        young_rates[s] = A.adl_fa_rates(M[jj], p_young[jj], THRESH_GRID, REFR)
    old_rates, old_counts, old_hours = {}, {}, {}
    for s in np.unique(Mh.subject):
        jj = (Mh.subject == s).to_numpy()
        old_rates[s], old_counts[s], old_hours[s] = har_fa_rates(Mh[jj], p_old[jj], THRESH_GRID)
    rows, per = [], []
    gi = lambda th: int(np.searchsorted(THRESH_GRID, th - 1e-12))
    for B in BUDGETS:
        th_young = {s: theta_from_rates(young_rates[s], B) for s in subs}
        th_old = {s: theta_from_rates(old_rates[s], B) for s in old_rates}
        # (1) young-calibrated threshold (17 volunteers) applied to every older adult
        thr_y = min(conformal_threshold(list(th_young.values()), DELTA), 1.0)
        for s in old_rates:
            k = gi(thr_y)
            per.append(dict(calibration="young volunteers (17)", budget_per_h=B, population="older adults (HAR70+)", subject=s,
                            threshold=thr_y, fa=int(old_counts[s][k]) if k < len(THRESH_GRID) else 0, hours=old_hours[s],
                            fa_per_h=old_rates[s][k] if k < len(THRESH_GRID) else 0.0))
        # (1b) the same threshold plus a personal ADL margin from the first 5 minutes of the older adult's own recording
        #      (threshold raised above every score of those minutes); evaluated on the rest of the recording
        for s in old_rates:
            jj = np.where((Mh.subject == s).to_numpy())[0]
            t_abs = Mh.segment.to_numpy()[jj] * 1e6 + Mh.t_end.to_numpy()[jj]
            order = jj[np.argsort(t_abs, kind="stable")]
            on, rest = order[:750], order[750:]                    # 750 windows x 0.4 s = 5 minutes
            thr_m = max(thr_y, min(float(p_old[on].max()) + 1e-3, 1.0))
            fa_rest = 0; h_rest = 0.0
            Mr = Mh.iloc[rest].assign(p=p_old[rest])
            for _, g in Mr.groupby("segment"):
                te_, pp = g.t_end.to_numpy(), g.p.to_numpy()
                h_rest += (te_.max() - te_.min() + 4.0) / 3600.0
                fa_rest += len(alarms(te_, pp, thr_m, REFR))
            k = gi(thr_y)
            fa_base = 0
            for _, g in Mr.groupby("segment"):
                fa_base += len(alarms(g.t_end.to_numpy(), g.p.to_numpy(), thr_y, REFR))
            per.append(dict(calibration="young volunteers (17) + personal ADL margin", budget_per_h=B, population="older adults (HAR70+)",
                            subject=s, threshold=thr_m, fa=fa_rest, hours=h_rest, fa_per_h=fa_rest / h_rest,
                            margin_raised=bool(thr_m > thr_y), fa_without_margin_same_period=fa_base))
        # (2) older-adult-calibrated threshold, leave-one-older-adult-out
        for s in old_rates:
            others = [th_old[j] for j in old_rates if j != s]
            thr = min(conformal_threshold(others, DELTA), 1.0)
            k = gi(thr)
            per.append(dict(calibration="older adults, leave-one-out (17)", budget_per_h=B, population="older adults (HAR70+)", subject=s,
                            threshold=thr, fa=int(old_counts[s][k]) if k < len(THRESH_GRID) else 0, hours=old_hours[s],
                            fa_per_h=old_rates[s][k] if k < len(THRESH_GRID) else 0.0))
        # (3) young volunteers with their own leave-one-out conformal threshold (reference for this reduced detector)
        for s in subs:
            others = [th_young[j] for j in subs if j != s]
            thr = min(conformal_threshold(others, DELTA), 1.0)
            jj = (M.subject == s).to_numpy()
            ev = eval_person(M[jj], p_young[jj], thr)
            per.append(dict(calibration="young volunteers, leave-one-out (16)", budget_per_h=B, population="young volunteers (falls dataset)",
                            subject=s, threshold=thr, fa=ev["fa"], hours=ev["adl_hours"], fa_per_h=ev["fa_per_h"],
                            falls=ev["falls"], detected=ev["detected"]))
        # fall sensitivity (volunteers' held-out scores) of the two deployable thresholds
        thr_o = min(conformal_threshold(list(th_old.values()), DELTA), 1.0)
        for lab, thr in [("young volunteers (17)", thr_y), ("older adults (18)", thr_o)]:
            det = fal = 0
            for s in subs:
                jj = (M.subject == s).to_numpy()
                ev = eval_person(M[jj], p_young[jj], thr)
                det += ev["detected"]; fal += ev["falls"]
            rows.append(dict(budget_per_h=B, calibration_source=lab, threshold=thr, n_cal=17 if lab.startswith("young") else 18,
                             fall_sensitivity_on_volunteers=det / fal, falls=fal))
    P = pd.DataFrame(per)
    P.to_csv(os.path.join(TAB_PATH, "S19_falls_older_adults_per_subject.csv"), index=False)
    P["B"] = P.budget_per_h                                    # grouping columns are not passed to apply (pandas >= 2)
    summ = P.groupby(["calibration", "population", "budget_per_h"]).apply(lambda g: pd.Series(dict(
        people=len(g), covered=int((g.fa_per_h <= g.B + 1e-9).sum()), coverage=(g.fa_per_h <= g.B + 1e-9).mean(),
        people_with_fa=int((g.fa > 0).sum()), total_fa=int(g.fa.sum()), hours=g.hours.sum(),
        pooled_fa_per_h=g.fa.sum() / g.hours.sum(), median_fa_per_h=g.fa_per_h.median(), max_fa_per_h=g.fa_per_h.max(),
        median_threshold=g.threshold.median(),
        people_margin_raised=int(g.margin_raised.fillna(False).astype(bool).sum()) if "margin_raised" in g else 0,
        fall_sensitivity=(g.detected.sum() / g.falls.sum()) if "falls" in g and g.falls.notna().any() else np.nan))).reset_index()
    summ.to_csv(os.path.join(TAB_PATH, "T6c_falls_older_adults.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(TAB_PATH, "T6d_falls_threshold_sources.csv"), index=False)
    from sklearn.metrics import roc_auc_score
    keep_m = ((M.pos == 1) | (M.adl == 1)).to_numpy()
    auc_m = roc_auc_score(M.pos.to_numpy()[keep_m], p_young[keep_m])
    print("magnitude detector window AUROC (young, LOPO): %.4f" % auc_m)
    import json
    json.dump(dict(window_auroc_young_lopo=float(auc_m), har70_windows=int(len(Mh)), har70_people=int(Mh.subject.nunique()),
                   har70_hours=float(sum(old_hours.values()))),
              open(os.path.join(TAB_PATH, "S19b_magnitude_detector.json"), "w"))
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    {"split": run_split, "har70": run_har70}[sys.argv[1]]()
