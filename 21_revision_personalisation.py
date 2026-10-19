"""
STEP 21 (revision) - Can on-device personalisation avoid forgetting? (activity of adults 70+ and FoG)

The original analysis (script 09) fine-tuned all layers of the federated MLP on the first m minutes of a new
person's routine and found that this lowered macro-F1 (activity) and pooled AUROC (FoG), because the onboarding
minutes rarely contain stairs, shuffling or freezes. Here we test, on the same leave-one-person-out folds and seeds:

  FT all layers                 - as in script 09 (reproduces Table 5)
  FT last layer                 - only the output layer is adapted
  FT all + statistics replay    - class-balanced replay of pseudo-windows drawn from class-conditional Gaussians whose
                                  means/variances are federated statistics (sums over clients; no raw data shared)
  FT last + statistics replay   - both
  FT all + real-data replay     - class-balanced replay of real training windows (privacy-costly reference)
  Guided onboarding (activity)  - the m onboarding minutes are spread over every activity class the person performs
                                  (a scripted session) instead of the first minutes of the routine
  Onboarding with a freeze (FoG)- the m onboarding minutes start 30 s before the person's first annotated freeze

Outputs: TAB_PATH/T5b_personalisation_revision_<service>.csv, S17_personalisation_per_subject_<service>.csv
Usage:   python 21_revision_personalisation.py fog|har70
"""
import json
import os
import pickle
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, f1_score

from config import PRED_PATH, TAB_PATH, N_JOBS, SEEDS, FL, PROCESSED_PATH
from lib.fed import fedavg
from lib.metrics import wilcoxon_p, holm
from lib.nn import MLP, Adam, train_local
from lib.zoo import load_service

MINUTES = (1, 5)
REPLAY_PER_CLASS = 200


def window_minutes(service):
    return 1.0 / 60.0 if service == "fog" else 2.5 / 60.0


def score(K, y, P):
    if K == 2:
        return roc_auc_score(y, P[:, 1]) if len(np.unique(y)) == 2 else np.nan
    return f1_score(y, P.argmax(1), labels=np.unique(y), average="macro", zero_division=0)


def finetune(model, X, y, K, epochs=15, lr=1e-3, batch=64, last_only=False, seed=0):
    m = MLP(model.params["W0"].shape[0], [model.params[f"W{i}"].shape[1] for i in range(model.n_layers - 1)], K, seed=seed, dropout=0.0)
    m.set(model.get())
    keys = [f"W{m.n_layers - 1}", f"b{m.n_layers - 1}"] if last_only else None
    if len(y) == 0:
        return m
    opt = Adam(m.params, lr=lr)
    for ep in range(epochs):
        train_local(m, X, y, K, epochs=1, lr=lr, batch=batch, seed=seed + ep, keys=keys, opt=opt)
    return m


def run_fold(service, d, s):
    cache = os.path.join(PRED_PATH, "cache", f"pers_{service}_{s}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    K = d["K"]
    te = np.where(d["subject"] == s)[0]
    tr = np.where(d["subject"] != s)[0]
    order = te[np.argsort(d["t"][te], kind="stable")]
    wpm = window_minutes(service)
    n_max = int(round(max(MINUTES) / wpm))
    routine_all = order[:n_max]
    ev = np.setdiff1d(te, routine_all)                         # same evaluation windows as script 09 / Table 5
    mu = d["F"][tr].mean(0); sd = d["F"][tr].std(0) + 1e-6
    Z = np.clip((d["F"] - mu) / sd, -8, 8)
    clients = [(Z[d["subject"] == c], d["y"][d["subject"] == c]) for c in np.unique(d["subject"][tr])]
    # federated class-conditional statistics (per-client sums of x and x^2 per class -> mean / variance)
    cls_mu = np.zeros((K, Z.shape[1])); cls_sd = np.ones((K, Z.shape[1])); present = []
    for c in range(K):
        Zc = Z[tr][d["y"][tr] == c]
        if len(Zc):
            cls_mu[c] = Zc.mean(0); cls_sd[c] = Zc.std(0) + 1e-6; present.append(c)
    # guided / event-containing onboarding sets (alternative sessions of the same length)
    guided = {}
    for m in MINUTES:
        n = int(round(m / wpm))
        if service == "har70":
            cl = [c for c in range(K) if (d["y"][te] == c).any()]
            per = int(np.ceil(n / len(cl)))
            sel = np.concatenate([order[d["y"][order] == c][:per] for c in cl])[:n]
            guided[m] = sel
        else:
            eps = d["y"][order] == 1
            if eps.any():
                first = np.argmax(eps)
                a = max(0, first - 30)
                guided[m] = order[a:a + n]
            else:
                guided[m] = np.zeros(0, int)
    excl = np.unique(np.concatenate([routine_all] + [g for g in guided.values()]))
    ev_g = np.setdiff1d(te, excl)
    out = dict(subject=s, ev=ev, ev_g=ev_g, P={}, Pg={})
    rng_all = np.random.default_rng(s)
    for sd_ in SEEDS:
        g, _ = fedavg(clients, Z.shape[1], K, rounds=FL["rounds"], local_epochs=FL["local_epochs"], lr=FL["lr"],
                      batch=FL["batch"], hidden=FL["hidden"], mu=0.0, dropout=FL["dropout"], seed=sd_, frac=1.0)
        out["P"].setdefault("FedAvg MLP", []).append(g.predict_proba(Z[ev]))
        out["Pg"].setdefault("FedAvg MLP", []).append(g.predict_proba(Z[ev_g]))
        rng = np.random.default_rng(1000 * sd_ + int(s))
        stat_X = np.concatenate([rng.normal(cls_mu[c], cls_sd[c], size=(REPLAY_PER_CLASS, Z.shape[1])) for c in present])
        stat_y = np.concatenate([np.full(REPLAY_PER_CLASS, c) for c in present])
        real_idx = np.concatenate([rng.choice(tr[d["y"][tr] == c], size=REPLAY_PER_CLASS, replace=(d["y"][tr] == c).sum() < REPLAY_PER_CLASS) for c in present])
        for m in MINUTES:
            on = order[:int(round(m / wpm))]
            Xo, yo = Z[on], d["y"][on]
            variants = {
                f"FT all layers ({m} min)": (Xo, yo, False),
                f"FT last layer ({m} min)": (Xo, yo, True),
                f"FT all + statistics replay ({m} min)": (np.concatenate([Xo, np.clip(stat_X, -8, 8)]), np.concatenate([yo, stat_y]), False),
                f"FT last + statistics replay ({m} min)": (np.concatenate([Xo, np.clip(stat_X, -8, 8)]), np.concatenate([yo, stat_y]), True),
                f"FT all + real-data replay ({m} min)": (np.concatenate([Xo, Z[real_idx]]), np.concatenate([yo, d["y"][real_idx]]), False),
            }
            for nm, (X_, y_, last) in variants.items():
                ft = finetune(g, X_, y_.astype(int), K, epochs=15, lr=1e-3, last_only=last, seed=sd_)
                out["P"].setdefault(nm, []).append(ft.predict_proba(Z[ev]))
            # guided / event-containing onboarding, evaluated on windows outside every onboarding set
            gl = guided[m]
            gname = "guided onboarding" if service == "har70" else "onboarding with a freeze"
            if len(gl):
                for nm, last in [(f"FT all layers, {gname} ({m} min)", False), (f"FT last layer, {gname} ({m} min)", True)]:
                    ft = finetune(g, Z[gl], d["y"][gl], K, epochs=15, lr=1e-3, last_only=last, seed=sd_)
                    out["Pg"].setdefault(nm, []).append(ft.predict_proba(Z[ev_g]))
                ft = finetune(g, Z[order[:int(round(m / wpm))]], d["y"][order[:int(round(m / wpm))]], K, epochs=15, lr=1e-3, last_only=False, seed=sd_)
                out["Pg"].setdefault(f"FT all layers, routine onboarding ({m} min)", []).append(ft.predict_proba(Z[ev_g]))
    out["P"] = {k: np.mean(v, 0).astype(np.float32) for k, v in out["P"].items()}
    out["Pg"] = {k: np.mean(v, 0).astype(np.float32) for k, v in out["Pg"].items() if len(v) == len(SEEDS)}
    out["guided_class_counts"] = {m: np.bincount(d["y"][g_], minlength=K).tolist() for m, g_ in guided.items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    print("fold done", service, s, flush=True)
    return out


def summarise(service, d, res, key):
    K = d["K"]
    methods = list(dict.fromkeys(m for r in res for m in r[key].keys()))
    # people for whom every method exists (FoG sessions with a freeze need a person with freezes), so that pooled and
    # per-person scores of all methods, including the reference, are computed on the same people
    common = [r["subject"] for r in res if all(m in r[key] for m in methods)]
    rows, per = [], []
    for m in methods:
        ys, Ps = [], []
        for r in res:
            if m not in r[key] or r["subject"] not in common:
                continue
            ev = r["ev"] if key == "P" else r["ev_g"]
            ys.append(d["y"][ev]); Ps.append(r[key][m])
            per.append(dict(method=m, set=key, subject=r["subject"], score=score(K, d["y"][ev], r[key][m]), n_eval=len(ev)))
        y = np.concatenate(ys); P = np.concatenate(Ps)
        row = dict(method=m, eval_set="after routine onboarding (as Table 5)" if key == "P" else "outside every onboarding set",
                   people=len(ys))
        if K == 2:
            row["AUROC"] = roc_auc_score(y, P[:, 1])
        else:
            row["MacroF1"] = f1_score(y, P.argmax(1), average="macro"); row["Acc"] = float((P.argmax(1) == y).mean())
            f1c = f1_score(y, P.argmax(1), labels=np.arange(K), average=None, zero_division=0)
            for c, nm in enumerate(d["classes"]):
                row[f"F1_{nm}"] = f1c[c]
        rows.append(row)
    T = pd.DataFrame(rows); Pp = pd.DataFrame(per)
    T = T.merge(Pp.groupby("method").score.median().rename("subject_median"), left_on="method", right_index=True)
    a = Pp[Pp.method == "FedAvg MLP"].set_index("subject").score
    T["p_vs_FedAvg"] = [wilcoxon_p(a.values, Pp[Pp.method == m].set_index("subject").score.reindex(a.index).values) if m != "FedAvg MLP" else np.nan for m in T.method]
    T["p_holm"] = holm(T["p_vs_FedAvg"].values)
    return T, Pp


def main(service):
    t0 = time.time()
    d = load_service(service)
    subs = np.unique(d["subject"])
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_fold)(service, d, s) for s in subs)
    T1, P1 = summarise(service, d, res, "P")
    T2, P2 = summarise(service, d, res, "Pg")
    T = pd.concat([T1, T2], ignore_index=True)
    T.to_csv(os.path.join(TAB_PATH, f"T5b_personalisation_revision_{service}.csv"), index=False)
    pd.concat([P1, P2]).to_csv(os.path.join(TAB_PATH, f"S17_personalisation_per_subject_{service}.csv"), index=False)
    cc = {int(r["subject"]): r["guided_class_counts"] for r in res}
    json.dump(cc, open(os.path.join(TAB_PATH, f"S17b_onboarding_class_counts_{service}.json"), "w"))
    print(T.round(4).to_string(index=False))
    print("elapsed %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main(sys.argv[1])
