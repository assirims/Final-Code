"""
STEP 18 (revision) - What does the CARES dual-view ensemble add, and are its probabilities calibrated?

Usage: python 18_revision_fusion_calibration.py offline          (uses the stored predictions of script 08)
       python 18_revision_fusion_calibration.py retrain fog|har70|falls

offline
  (a) Weight ablation: p = w * GBT(descriptors) + (1 - w) * LR(MiniRocket-lite), w = 0, 0.1, ..., 1 (w = 0.5 is CARES).
  (b) Nested weight (binary services): in every leave-one-person-out fold, w is chosen on the inner out-of-fold scores
      of the training people (maximum AUROC) and applied to the held-out person.
  (c) Stacking (binary): logistic meta-learner on the inner out-of-fold logits of the two branches.
  (d) Calibration of every model: expected calibration error (ECE, 15 equal-width bins; positive class for binary
      services, top-label for activity), Brier score and reliability curves; plus CARES after Platt scaling fitted on the
      inner out-of-fold scores of the training people (binary services).
retrain
  (e) Simple single-model alternatives that use the same two views: one GBT on the concatenation of descriptors and
      MiniRocket-lite features (early fusion), and the ensemble of GBT(descriptors) and GBT(MiniRocket-lite).
Outputs: TAB_PATH/T3b_fusion_weights.csv, T3c_fusion_alternatives.csv, T3d_calibration.csv, S20_reliability.csv
"""
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from config import PRED_PATH, TAB_PATH, N_JOBS, SEED, HGB
from lib.metrics import wilcoxon_p, holm, cluster_bootstrap
from lib.zoo import load_service, make_rocket

SERVICES = ["falls", "fog", "har70"]
W = np.round(np.linspace(0, 1, 11), 2)
MODEL_LABEL = {"Threshold": "Conventional threshold", "LR": "Logistic regression", "RF": "Random forest", "HGB": "GBT (descriptors)",
               "MLP": "MLP", "MiniRocket-LR": "MiniRocket-lite + LR", "CARES (dual-view)": "CARES dual-view ensemble"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, P, bins=15):
    """Binary: calibration of P(y=1). Multiclass: top-label ECE."""
    if P.ndim == 1:
        conf, corr = P, y.astype(float)
    else:
        conf, corr = P.max(1), (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
            rows.append((b, int(m.sum()), float(conf[m].mean()), float(corr[m].mean())))
    return e, rows


def score_of(service, y, P):
    if service == "har70":
        return f1_score(y, P.argmax(1), average="macro")
    return roc_auc_score(y, P[:, 1] if P.ndim == 2 else P)


def offline():
    wrows, arows, crows, rel = [], [], [], []
    for s in SERVICES:
        b = np.load(os.path.join(PRED_PATH, f"bench_{s}.npz"))
        y, subj = b["y"], b["subject"]
        Pg, Pm = b["P::HGB"], b["P::MiniRocket-LR"]
        binary = Pg.shape[1] == 2
        subs = np.unique(subj)
        # (a) weight grid
        for w in W:
            P = w * Pg + (1 - w) * Pm
            per = []
            for p_ in subs:
                i = subj == p_
                if binary and len(np.unique(y[i])) < 2:
                    continue
                per.append(score_of(s, y[i], P[i]))
            r = dict(service=s, w_gbt=w, pooled=score_of(s, y, P), person_median=float(np.median(per)))
            if binary:
                r["AUPRC"] = average_precision_score(y, P[:, 1])
            wrows.append(r)
        if binary:
            # (b) nested weight and (c) stacking, chosen on inner OOF scores of the training people only
            Pn = np.zeros(len(y)); Ps = np.zeros(len(y)); Pc = np.zeros(len(y)); chosen = []
            for p_ in subs:
                te = subj == p_
                ytr = y[~te]
                ig, im = b[f"inner::HGB|{p_}"], b[f"inner::MiniRocket-LR|{p_}"]
                auc_w = [roc_auc_score(ytr, w * ig + (1 - w) * im) for w in W]
                w_star = W[int(np.argmax(auc_w))]
                chosen.append(w_star)
                Pn[te] = w_star * Pg[te, 1] + (1 - w_star) * Pm[te, 1]
                meta = LogisticRegression(C=1.0).fit(np.c_[logit(ig), logit(im)], ytr)
                Ps[te] = meta.predict_proba(np.c_[logit(Pg[te, 1]), logit(Pm[te, 1])])[:, 1]
                # Platt scaling of CARES on the inner OOF CARES scores
                ic = b[f"inner::CARES (dual-view)|{p_}"]
                pl = LogisticRegression(C=1e6).fit(logit(ic)[:, None], ytr)
                Pc[te] = pl.predict_proba(logit(b["P::CARES (dual-view)"][te, 1])[:, None])[:, 1]
            for nm, P in [("CARES, fixed w = 0.5", b["P::CARES (dual-view)"][:, 1]), ("nested weight (chosen on inner folds)", Pn),
                          ("stacking (logistic meta-learner on inner folds)", Ps)]:
                lo, hi = cluster_bootstrap(subj, lambda ii: roc_auc_score(y[ii], P[ii]), n_boot=500, seed=1)
                arows.append(dict(service=s, method=nm, AUROC=roc_auc_score(y, P), AUROC_lo=lo, AUROC_hi=hi,
                                  AUPRC=average_precision_score(y, P),
                                  chosen_w=f"{np.min(chosen):.1f}-{np.max(chosen):.1f} (median {np.median(chosen):.1f})" if nm.startswith("nested") else ""))
            models = [m for m in b["models"]] + ["CARES + Platt (inner folds)"]
        else:
            models = list(b["models"])
            Pc = None
        # (d) calibration
        for m in models:
            if m == "CARES + Platt (inner folds)":
                P = Pc
            else:
                P = b[f"P::{m}"]
                P = P[:, 1] if binary else P
            e, rr = ece(y, P)
            brier = float(np.mean((P - y) ** 2)) if binary else float(np.mean(((P - np.eye(P.shape[1])[y]) ** 2).sum(1)))
            lo, hi = cluster_bootstrap(subj, lambda ii: ece(y[ii], P[ii])[0], n_boot=300, seed=2)
            crows.append(dict(service=s, model=MODEL_LABEL.get(m, m), ECE=e, ECE_lo=lo, ECE_hi=hi, Brier=brier))
            rel += [dict(service=s, model=MODEL_LABEL.get(m, m), bin=bb, n=n, confidence=c, observed=o) for bb, n, c, o in rr]
    T = pd.DataFrame(wrows); T.to_csv(os.path.join(TAB_PATH, "T3b_fusion_weights.csv"), index=False)
    A_ = pd.DataFrame(arows); A_.to_csv(os.path.join(TAB_PATH, "T3c_fusion_nested_stacking.csv"), index=False)
    C = pd.DataFrame(crows); C.to_csv(os.path.join(TAB_PATH, "T3d_calibration.csv"), index=False)
    pd.DataFrame(rel).to_csv(os.path.join(TAB_PATH, "S20_reliability.csv"), index=False)
    print(T.round(4).to_string(index=False)); print(A_.round(4).to_string(index=False)); print(C.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------- retrain
def gbt():
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **HGB)


def retrain_fold(service, d, s):
    cache = os.path.join(PRED_PATH, "cache", f"fusion_{service}_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    te = d["subject"] == s
    tr = ~te
    K = d["K"]
    mr = make_rocket(service, d["R"][tr], seed=SEED)
    M = mr.transform(d["R"])
    out = dict(subject=s, idx=np.where(te)[0])

    def fitp(X):
        m = gbt().fit(X[tr], d["y"][tr])
        P = np.zeros((te.sum(), K)); P[:, m.classes_] = m.predict_proba(X[te])
        return P.astype(np.float32)
    out["GBT on concatenated views (early fusion)"] = fitp(np.concatenate([d["F"], M], 1))
    out["GBT (MiniRocket-lite)"] = fitp(M)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    print("fusion fold done", service, s, flush=True)
    return out


def retrain(service):
    t0 = time.time()
    d = load_service(service)
    subs = np.unique(d["subject"])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(retrain_fold)(service, d, s) for s in subs)
    b = np.load(os.path.join(PRED_PATH, f"bench_{service}.npz"))
    y, subj = d["y"], d["subject"]
    K = d["K"]
    P = {k: np.zeros((len(y), K), np.float32) for k in ["GBT on concatenated views (early fusion)", "GBT (MiniRocket-lite)"]}
    for r in res:
        for k in P:
            P[k][r["idx"]] = r[k]
    P["GBT(descriptors) + GBT(MiniRocket-lite), averaged"] = 0.5 * (b["P::HGB"] + P["GBT (MiniRocket-lite)"])
    P["CARES dual-view ensemble"] = b["P::CARES (dual-view)"]
    np.savez_compressed(os.path.join(PRED_PATH, f"fusion_{service}.npz"), **{k: v for k, v in P.items()})
    rows, per = [], {}
    for k, Pk in P.items():
        if K == 2:
            v = roc_auc_score(y, Pk[:, 1]); lo, hi = cluster_bootstrap(subj, lambda ii: roc_auc_score(y[ii], Pk[ii, 1]), n_boot=500, seed=1)
            per[k] = pd.Series({p_: roc_auc_score(y[subj == p_], Pk[subj == p_, 1]) for p_ in np.unique(subj) if len(np.unique(y[subj == p_])) == 2})
            rows.append(dict(service=service, method=k, AUROC=v, AUROC_lo=lo, AUROC_hi=hi, AUPRC=average_precision_score(y, Pk[:, 1])))
        else:
            v = f1_score(y, Pk.argmax(1), average="macro")
            lo, hi = cluster_bootstrap(subj, lambda ii: f1_score(y[ii], Pk[ii].argmax(1), average="macro"), n_boot=500, seed=1)
            per[k] = pd.Series({p_: f1_score(y[subj == p_], Pk[subj == p_].argmax(1), labels=np.unique(y[subj == p_]), average="macro", zero_division=0) for p_ in np.unique(subj)})
            rows.append(dict(service=service, method=k, MacroF1=v, MacroF1_lo=lo, MacroF1_hi=hi, Acc=float((Pk.argmax(1) == y).mean())))
    T = pd.DataFrame(rows)
    ref = per["CARES dual-view ensemble"]
    T["p_vs_CARES"] = [wilcoxon_p(ref.values, per[m].reindex(ref.index).values) if m != "CARES dual-view ensemble" else np.nan for m in T.method]
    T["p_holm"] = holm(T.p_vs_CARES.values)
    f = os.path.join(TAB_PATH, "T3c_fusion_alternatives.csv")
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    if len(old):
        old = old[old.service != service]
    pd.concat([old, T], ignore_index=True).to_csv(f, index=False)
    print(T.round(4).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    if sys.argv[1] == "offline":
        offline()
    else:
        retrain(sys.argv[2])
