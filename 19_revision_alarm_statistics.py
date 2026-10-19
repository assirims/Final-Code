"""
STEP 19 (revision) - Uncertainty and per-person detail of the false-alarm budget results (scripts 10 and 20).

  (a) Calibration size n and conformal quantile index r = ceil((n+1)(1-delta)) of every leave-one-person-out fold,
      with the attainable guarantee r/(n+1).
  (b) Exact (Clopper-Pearson) 95% confidence intervals of every coverage estimate and exact McNemar tests between
      threshold policies evaluated on the same people.
  (c) Per-person false alarms: number of people with at least one false alarm, monitored hours per person, and the
      pooled false-alarm rate with an exact Poisson 95% CI and a person-cluster bootstrap CI.
  (d) Personal ADL margin: when does it change anything? The margin (threshold raised above every score of the first
      5 minutes of the new person's ADL) is added to the fixed, Youden and a less conservative conformal threshold
      (delta = 0.5, i.e. the median calibration threshold), using the stored streaming scores of script 10.
Rows for the older-adult analysis of script 20 (S19) are added to T6e/T6f when that table exists.
Outputs: TAB_PATH/S21_conformal_quantile_index.csv, T6e_coverage_ci.csv, S22_false_alarms_per_person.csv,
         T6f_false_alarm_distribution.csv, T6g_personal_margin_scenarios.csv
"""
import importlib
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from config import PRED_PATH, TAB_PATH, ALARM
from lib.alarms import conformal_threshold, THRESH_GRID
from lib.metrics import youden_threshold

A = importlib.import_module("10_calibrated_alarms")
DELTA = ALARM["delta"]
REFR = ALARM["refractory_s"]["falls"]


def cp(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def poisson_ci(k, a=0.05):
    lo = stats.chi2.ppf(a / 2, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2
    return lo, hi


def mcnemar(a, b):
    """Exact McNemar (binomial) test on paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return n01, n10, p


def load_tables():
    S7 = pd.read_csv(os.path.join(TAB_PATH, "S7_alarm_per_subject_fog.csv"))
    S7["fa"] = np.round(S7.fa_per_h * S7.hours).astype(int)
    S8 = pd.read_csv(os.path.join(TAB_PATH, "S8_alarm_per_subject_falls.csv"))
    S8["fa"] = np.round(S8.fa_per_h * S8.adl_hours).astype(int)
    S8 = S8.rename(columns={"adl_hours": "hours"})
    tabs = {"FoG": S7, "Falls": S8}
    p = os.path.join(TAB_PATH, "S18_falls_split_per_subject.csv")
    if os.path.exists(p):
        S18 = pd.read_csv(p).rename(columns={"adl_hours": "hours"})
        tabs["Falls (split conformal)"] = S18
    p = os.path.join(TAB_PATH, "S19_falls_older_adults_per_subject.csv")
    if os.path.exists(p):
        S19 = pd.read_csv(p).rename(columns={"calibration": "policy"})
        S19["within_budget"] = S19.fa_per_h <= S19.budget_per_h + 1e-9
        for pop, g in S19.groupby("population"):
            tabs["Falls, magnitude detector: " + pop] = g.reset_index(drop=True)
    return tabs


def part_a():
    rows = []
    for svc, n in [("FoG", 9), ("Falls", 16), ("Falls (split conformal)", 10), ("Falls, older-adult calibration (HAR70+)", 17)]:
        r = int(np.ceil((n + 1) * (1 - DELTA)))
        rows.append(dict(service=svc, folds={"FoG": 10, "Falls": 17, "Falls (split conformal)": 17}.get(svc, 18), n_calibration=n,
                         quantile_index=r, uses_maximum=r == n, attainable_level=r / (n + 1),
                         smallest_delta_with_finite_threshold=1 / (n + 1)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB_PATH, "S21_conformal_quantile_index.csv"), index=False)
    print(T.to_string(index=False))


def part_bc(tabs):
    rows, dist, per = [], [], []
    for svc, S in tabs.items():
        for (pol, B), g in S.groupby(["policy", "budget_per_h"]):
            k, n = int(g.within_budget.sum()), len(g)
            lo, hi = cp(k, n)
            fa, h = int(g.fa.sum()), g.hours.sum()
            plo, phi = poisson_ci(fa)
            # person-cluster bootstrap of the pooled rate
            rng = np.random.default_rng(3)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                bs.append(g.fa.to_numpy()[i].sum() / g.hours.to_numpy()[i].sum())
            rows.append(dict(service=svc, policy=pol, budget_per_h=B, people=n, covered=k, coverage=k / n, coverage_lo=lo, coverage_hi=hi))
            dist.append(dict(service=svc, policy=pol, budget_per_h=B, people=n, people_with_fa=int((g.fa > 0).sum()), total_fa=fa,
                             fa_per_person=";".join(str(x) for x in sorted(g.fa.astype(int), reverse=True)),
                             hours_total=h, hours_min=g.hours.min(), hours_median=g.hours.median(), hours_max=g.hours.max(),
                             pooled_fa_per_h=fa / h, pooled_lo_poisson=plo / h, pooled_hi_poisson=phi / h,
                             pooled_lo_boot=np.percentile(bs, 2.5), pooled_hi_boot=np.percentile(bs, 97.5),
                             upper95_rate_if_zero_median_person=-np.log(0.05) / g.hours.median()))
            per += [dict(service=svc, policy=pol, budget_per_h=B, subject=r.subject, fa=int(r.fa), hours=r.hours, fa_per_h=r.fa_per_h,
                         within_budget=bool(r.within_budget)) for r in g.itertuples()]
    T = pd.DataFrame(rows)
    # exact McNemar between the conformal rule and every other policy at the same budget (same people)
    pv = []
    for svc, S in tabs.items():
        conf = S[S.policy.str.startswith("conformal (proposed)") | S.policy.str.startswith("split")]
        for (pol, B), g in S.groupby(["policy", "budget_per_h"]):
            c = conf[conf.budget_per_h == B]
            if pol.startswith("conformal (proposed)") or pol.startswith("split") or not len(c):
                pv.append(np.nan); continue
            a = c.set_index("subject").within_budget
            b = g.set_index("subject").within_budget.reindex(a.index)
            pv.append(mcnemar(a.values, b.values)[2])
    T["mcnemar_p_vs_conformal"] = pv
    T.to_csv(os.path.join(TAB_PATH, "T6e_coverage_ci.csv"), index=False)
    D = pd.DataFrame(dist); D.to_csv(os.path.join(TAB_PATH, "T6f_false_alarm_distribution.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(TAB_PATH, "S22_false_alarms_per_person.csv"), index=False)
    print(T.round(3).to_string(index=False))
    print(D.drop(columns=["fa_per_person"]).round(3).to_string(index=False))


def fast_rates(Mj, pj):
    """Exact false alarms per ADL hour for every grid threshold. A trial shorter than the refractory period can raise at
    most one alarm (at its first window above the threshold), so its count is 1[max score >= threshold]."""
    adl = (Mj.adl == 1).to_numpy()
    g = pd.DataFrame(dict(trial=Mj.trial.to_numpy()[adl], t=Mj.t_end.to_numpy()[adl], p=pj[adl]))
    agg = g.groupby("trial").agg(pmax=("p", "max"), t0=("t", "min"), t1=("t", "max"))
    dur = agg.t1 - agg.t0
    hours = (dur + 4.0).sum() / 3600.0
    short = dur.to_numpy() < REFR
    counts = (agg.pmax.to_numpy()[short][:, None] >= THRESH_GRID[None, :]).sum(0).astype(float)
    for tri in agg.index[~short]:
        gg = g[g.trial == tri]
        counts += np.array([len(A.alarms(gg.t.to_numpy(), gg.p.to_numpy(), th, REFR)) for th in THRESH_GRID])
    return counts / hours


def part_d():
    d = np.load(os.path.join(PRED_PATH, "stream_falls.npz"))
    cols = ["subject", "activity", "rep", "trial", "fall", "t_end", "t_peak", "pos", "adl"]
    M = pd.DataFrame({c: d[c] for c in cols})
    P = d["p"]
    rows = []
    for s in np.unique(M.subject):
        r = pickle.load(open(os.path.join(PRED_PATH, "cache", f"stream_falls_{s}.pkl"), "rb"))
        Mi = M.iloc[r["idx_tr"]]
        thetas = []
        for j in np.unique(Mi.subject):
            jj = (Mi.subject == j).to_numpy()
            rates = fast_rates(Mi[jj], r["oof"][jj])
            ok = rates <= 1e-12
            adm = np.flip(np.logical_and.accumulate(np.flip(ok)))
            thetas.append(THRESH_GRID[np.where(adm)[0][0]] if adm.any() else 1.0)
        yy = Mi.pos.to_numpy(); keep = ((Mi.pos == 1) | (Mi.adl == 1)).to_numpy()
        base = {"fixed 0.5": 0.5, "Youden (window-optimal)": youden_threshold(yy[keep], r["oof"][keep]),
                "conformal, delta = 0.10 (paper)": min(conformal_threshold(thetas, 0.10), 1.0),
                "conformal, delta = 0.50 (median)": min(conformal_threshold(thetas, 0.50), 1.0)}
        Mt = M.iloc[r["idx_te"]]; pt = r["p_te"]
        adl_tr = Mt[Mt.adl == 1].sort_values(["activity", "rep"]).trial.unique()
        cum, onboard = 0.0, []
        for tri in adl_tr:
            if cum >= 300:
                break
            onboard.append(tri); cum += (Mt.trial == tri).sum() * 0.4 + 3.6
        on = Mt.trial.isin(onboard).to_numpy()
        pmax = pt[on].max() if on.any() else 0.0
        for nm, th in base.items():
            for margin in (False, True):
                thr = max(th, min(pmax + 1e-3, 1.0)) if margin else th
                ev = A.falls_trial_eval(Mt[~on], pt[~on], thr, REFR)
                f = ev[ev.fall]
                fa = int(ev.false_alarms.sum()); h = ev.adl_hours.sum()
                rows.append(dict(base_threshold=nm, personal_margin=margin, subject=s, threshold=thr, margin_raised=bool(margin and thr > th),
                                 fa=fa, hours=h, zero_fa=fa == 0, falls=len(f), detected=int(f.detected.sum())))
    R = pd.DataFrame(rows)
    S = R.groupby(["base_threshold", "personal_margin"]).apply(lambda g: pd.Series(dict(
        people=len(g), people_threshold_raised=int(g.margin_raised.sum()), people_zero_fa=int(g.zero_fa.sum()),
        coverage_B0=g.zero_fa.mean(), pooled_fa_per_h=g.fa.sum() / g.hours.sum(), total_fa=int(g.fa.sum()),
        fall_sensitivity=g.detected.sum() / g.falls.sum()))).reset_index()
    S.to_csv(os.path.join(TAB_PATH, "T6g_personal_margin_scenarios.csv"), index=False)
    R.to_csv(os.path.join(TAB_PATH, "S23_personal_margin_per_person.csv"), index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    part_a()
    part_bc(load_tables())
    part_d()
