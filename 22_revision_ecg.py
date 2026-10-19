"""
STEP 22 (revision) - Additional analyses of the arrhythmia service requested in peer review.

  (a) Records 201 and 202 come from the same person but belong to DS1 and DS2. Sensitivity analysis:
      (i) evaluate the published models on DS2 without record 202;
      (ii) retrain every global and patient-adaptive model on DS1 without record 201 and evaluate on DS2 without 202.
  (b) Hyper-parameters: the GBT setting is shared by all services (config.HGB). To show that it was not chosen on DS2,
      we select the GBT setting of the CARES ensemble by record-grouped 3-fold cross-validation inside DS1 only
      (12 candidate settings) and evaluate the DS1-selected setting once on DS2.
  (c) Per-class sensitivity, PPV and F1 for all five AAMI classes (N, S, V, F, Q), macro-F1 over N,S,V and over
      all five classes, and confusion matrices of every global and patient-adaptive model.
  (d) Size of the Q class in the DS2 evaluation set (context for the Q -> N confusion in Figure 15).

Outputs: TAB_PATH/S13_ecg_per_class.csv, S14_ecg_confusion.csv, S15_ecg_201_202.csv, S16_ecg_ds1_hyperparameters.csv
"""
import importlib
import itertools
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_PATH, PRED_PATH, TAB_PATH, SEED, HGB
from lib.minirocket import MiniRocketLite

E = importlib.import_module("08b_benchmark_ecg")
CLASSES = ["N", "S", "V", "F", "Q"]
K = 5
ONBOARD_S = 300.0


def per_class(y, yh):
    out = dict(n=len(y), Acc=float((y == yh).mean()))
    f1 = []
    for c, nm in enumerate(CLASSES):
        tp = int(((yh == c) & (y == c)).sum()); fn = int(((yh != c) & (y == c)).sum()); fp = int(((yh == c) & (y != c)).sum())
        out[f"n_{nm}"] = int((y == c).sum())
        out[f"Se_{nm}"] = tp / (tp + fn) if tp + fn else np.nan
        out[f"PPV_{nm}"] = tp / (tp + fp) if tp + fp else np.nan
        out[f"F1_{nm}"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        f1.append(out[f"F1_{nm}"])
    out["MacroF1_NSV"] = float(np.nanmean(f1[:3]))
    out["MacroF1_NSVF"] = float(np.nanmean(f1[:4]))
    out["MacroF1_NSVFQ"] = float(np.nanmean(f1))
    return out


def load():
    d = np.load(os.path.join(PROCESSED_PATH, "ecg_beats.npz"))
    F = np.concatenate([d["rr"], d["morph"], d["extra"]], 1).astype(np.float32)
    seg = d["seg"].astype(np.float32)
    return d, F, seg, d["label"].astype(int), d["record"].astype(int), d["t"]


def part_c():
    b = np.load(os.path.join(PRED_PATH, "bench_ecg.npz"))
    y, rec, t = b["y"], b["record"], b["t"]
    d = np.load(os.path.join(PROCESSED_PATH, "ecg_beats.npz"))
    late = np.isin(rec, d["ds2"]) & (t >= ONBOARD_S)
    rows, cms = [], []
    for k in b.files:
        if not (k.startswith("PA::") or k.startswith("PB::")):
            continue
        sc = "global" if k.startswith("PA::") else "patient-adaptive"
        m = k.split("::", 1)[1]
        if sc == "global" and m.startswith("Fed"):
            sc = "federated"
        if sc == "patient-adaptive" and m.startswith("Fed"):
            sc = "federated + fine-tuning"
        yh = b[k][late].argmax(1)
        rows.append(dict(scenario=sc, model=m, **per_class(y[late], yh)))
        C = np.zeros((K, K), int)
        np.add.at(C, (y[late], yh), 1)
        for i in range(K):
            for j in range(K):
                cms.append(dict(scenario=sc, model=m, true=CLASSES[i], pred=CLASSES[j], count=int(C[i, j])))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB_PATH, "S13_ecg_per_class.csv"), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(TAB_PATH, "S14_ecg_confusion.csv"), index=False)
    print(T[["scenario", "model", "Acc", "F1_N", "F1_S", "F1_V", "F1_F", "F1_Q", "n_Q", "MacroF1_NSV", "MacroF1_NSVFQ"]].round(3).to_string(index=False))


def scenarios(F, M, y, rec, tt, train_mask, test_records, label):
    """Global (A) and patient-adaptive (B) models; returns per-class rows on the beats after 5 min."""
    rows = []
    late = np.isin(rec, test_records) & (tt >= ONBOARD_S)
    G = E.ECGModels(SEED).fit_all(F[train_mask], M[train_mask], y[train_mask])
    PA = G.proba(F[late], M[late])
    for nm, P in PA.items():
        rows.append(dict(analysis=label, scenario="global", model=nm, **per_class(y[late], P.argmax(1))))
    PB = {nm: np.zeros((len(y), K)) for nm in ["LR", "HGB", "MiniRocket-LR", "CARES (dual-view)", "MLP"]}
    scz = G.sc
    for r in test_records:
        loc = (rec == r) & (tt < ONBOARD_S)
        ev = (rec == r) & (tt >= ONBOARD_S)
        Fa = np.concatenate([F[train_mask]] + [F[loc]] * 5); Ma = np.concatenate([M[train_mask]] + [M[loc]] * 5)
        ya = np.concatenate([y[train_mask]] + [y[loc]] * 5)
        Pm = E.ECGModels(SEED).fit_all(Fa, Ma, ya, full=False).proba(F[ev], M[ev])
        for nm in Pm:
            PB[nm][ev] = Pm[nm]
        mft = E.finetune(G.net, np.clip(scz.transform(F[loc]), -8, 8), y[loc], K, epochs=20, lr=1e-3, last_only=True, seed=SEED)
        PB["MLP"][ev] = mft.predict_proba(np.clip(scz.transform(F[ev]), -8, 8))
    for nm, P in PB.items():
        rows.append(dict(analysis=label, scenario="patient-adaptive", model=nm, **per_class(y[late], P[late].argmax(1))))
    return rows


def part_a(d, F, M, y, rec, tt):
    ds1, ds2 = list(d["ds1"]), list(d["ds2"])
    rows = []
    # (i) published predictions, DS2 without record 202
    b = np.load(os.path.join(PRED_PATH, "bench_ecg.npz"))
    for excl, lab in [([], "published models, DS2 (22 records)"), ([202], "published models, DS2 without 202")]:
        late = np.isin(rec, ds2) & ~np.isin(rec, excl) & (tt >= ONBOARD_S)
        for k in b.files:
            if k.startswith("PA::") or k.startswith("PB::"):
                m = k.split("::", 1)[1]
                if m.startswith("Fed"):
                    continue
                sc = "global" if k.startswith("PA::") else "patient-adaptive"
                rows.append(dict(analysis=lab, scenario=sc, model=m, **per_class(y[late], b[k][late].argmax(1))))
    # (ii) retrained without record 201, evaluated without 202
    tr = np.isin(rec, [r for r in ds1 if r != 201])
    rows += scenarios(F, M, y, rec, tt, tr, [r for r in ds2 if r != 202], "retrained on DS1 without 201, DS2 without 202")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB_PATH, "S15_ecg_201_202.csv"), index=False)
    print(T[T.model.isin(["CARES (dual-view)", "HGB", "RF", "LR"])][["analysis", "scenario", "model", "Se_S", "PPV_S", "Se_V", "PPV_V", "MacroF1_NSV"]].round(3).to_string(index=False))


def part_b(d, F, M, y, rec, tt):
    ds1 = np.array(d["ds1"])
    rng = np.random.default_rng(SEED)
    recs = rng.permutation(ds1)
    folds = [recs[i::3] for i in range(3)]
    grid = [dict(max_iter=it, learning_rate=lr, max_leaf_nodes=nl, l2_regularization=1.0)
            for it, lr, nl in itertools.product([150, 300], [0.03, 0.06, 0.1], [15, 31])]
    rows = []
    for gi, g in enumerate(grid):
        P_oof = np.zeros((len(y), K)); mask = np.zeros(len(y), bool)
        for k in range(3):
            va = np.isin(rec, folds[k]); trm = np.isin(rec, ds1) & ~va
            idx = np.where(trm)[0][E.undersample(y[trm], SEED)]
            gbt = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **g).fit(F[idx], y[idx])
            scm = StandardScaler().fit(M[trm])
            lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(scm.transform(M[trm]), y[trm])
            P = 0.5 * (E.expand(gbt.predict_proba(F[va]), gbt.classes_) + E.expand(lr.predict_proba(scm.transform(M[va])), lr.classes_))
            P_oof[va] = P; mask |= va
        m = per_class(y[mask], P_oof[mask].argmax(1))
        rows.append(dict(setting=gi, **g, inner_MacroF1_NSV=m["MacroF1_NSV"], inner_MacroF1_NSVFQ=m["MacroF1_NSVFQ"],
                         is_default=all(g[k] == HGB[k] for k in g)))
        print("grid", gi, g, "inner macro-F1 %.3f" % m["MacroF1_NSV"], flush=True)
    G = pd.DataFrame(rows)
    best = G.sort_values("inner_MacroF1_NSV", ascending=False).iloc[0]
    late = np.isin(rec, d["ds2"]) & (tt >= ONBOARD_S)
    trm = np.isin(rec, ds1)
    out = []
    for lab, g in [("DS1-selected", {k: best[k] for k in ["max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization"]}),
                   ("default (config.HGB)", HGB)]:
        g = {"max_iter": int(g["max_iter"]), "learning_rate": float(g["learning_rate"]), "max_leaf_nodes": int(g["max_leaf_nodes"]),
             "l2_regularization": float(g["l2_regularization"])}
        idx = np.where(trm)[0][E.undersample(y[trm], SEED)]
        gbt = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **g).fit(F[idx], y[idx])
        scm = StandardScaler().fit(M[trm])
        lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(scm.transform(M[trm]), y[trm])
        P = 0.5 * (E.expand(gbt.predict_proba(F[late]), gbt.classes_) + E.expand(lr.predict_proba(scm.transform(M[late])), lr.classes_))
        out.append(dict(setting=lab, **g, DS2_global=True, **per_class(y[late], P.argmax(1))))
    G.to_csv(os.path.join(TAB_PATH, "S16_ecg_ds1_hyperparameters.csv"), index=False)
    pd.DataFrame(out).to_csv(os.path.join(TAB_PATH, "S16b_ecg_ds1_selected_on_ds2.csv"), index=False)
    print(G.round(3).to_string(index=False))
    print(pd.DataFrame(out)[["setting", "max_iter", "learning_rate", "max_leaf_nodes", "MacroF1_NSV", "Se_S", "PPV_S"]].round(3).to_string(index=False))


def main():
    t0 = time.time()
    part_c()
    d, F, seg, y, rec, tt = load()
    ds1 = np.isin(rec, d["ds1"])
    mr = MiniRocketLite(dilations=(1, 2, 4, 8), n_bias=1, seed=SEED).fit(seg[ds1][:, :, None])
    M = mr.transform(seg[:, :, None])
    part_b(d, F, M, y, rec, tt)
    print("b done %.1f min" % ((time.time() - t0) / 60), flush=True)
    # MiniRocket biases refitted without record 201 for the strict sensitivity analysis
    ds1b = np.isin(rec, [r for r in d["ds1"] if r != 201])
    mr2 = MiniRocketLite(dilations=(1, 2, 4, 8), n_bias=1, seed=SEED).fit(seg[ds1b][:, :, None])
    M2 = mr2.transform(seg[:, :, None])
    part_a(d, F, M2, y, rec, tt)
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
