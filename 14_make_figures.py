"""
STEP 14 - Result figures (PNG, 300 dpi). Figures 1-4 are the diagrams of script 15.
  Fig5  benchmark of the edge detectors (falls, FoG, activity, ECG)
  Fig6  person-level results (every held-out person)
  Fig7  fusion-weight ablation and reliability of the predicted probabilities (revision)
  Fig8  arrhythmia service: global vs patient-adaptive vs federated
  Fig9  federated learning and on-device personalisation
  Fig10 calibrated alarm budget: coverage (with exact 95% CIs) and event sensitivity
  Fig11 example streams: FoG scores with freeze episodes and alerts; fall trial
  Fig12 cognitive-health surveillance (ROC, risk by group, informative tasks)
  Fig13 robustness: sensor placement, sensor dropout, sampling rate, noise
  Fig14 alert explanations and faithfulness
  Fig15 edge cost, data minimisation and alert delivery
  Fig16 confusion matrices (activity; ECG global and patient-adaptive)
  Fig17 FedAvg learning curves
  FigS1 confusion matrices of every global ECG model (supplement)
Colour rules: the proposed CARES model is always slot-1 blue, comparators are neutral grey (emphasis, not a rainbow);
when several entities must be distinguished, the validated categorical slots are used in a fixed order.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

from config import FIG_PATH, TAB_PATH, PRED_PATH, PROCESSED_PATH, ALARM

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # validated order
PROP = CAT[0]                       # CARES (entity colour, every figure)
GREY, GREY_D, INK, MUTED = "#b4b8bd", "#6b7178", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 9.5, "savefig.dpi": 300, "figure.dpi": 100,
                     "axes.edgecolor": "#8a8f95", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "axes.grid": False, "legend.frameon": False, "lines.linewidth": 2})
SVC = {"falls": "Fall detection (17 people)", "fog": "Freezing of gait (10 people with PD)",
       "har70": "Activity monitoring (18 adults aged 70+)"}
NAME = {"Threshold": "Conventional threshold", "HGB": "GBT (descriptors)", "MiniRocket-LR": "MiniRocket-lite + LR",
        "CARES (dual-view)": "CARES (proposed)", "LR": "Logistic regression", "RF": "Random forest", "MLP": "MLP"}


def nm(m):
    return NAME.get(m, m)


def T(name):
    p = os.path.join(TAB_PATH, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG_PATH, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def hbars(ax, labels, vals, lo=None, hi=None, prop_mask=None, xlabel="", title="", xlim=None, digits=3):
    y = np.arange(len(labels))
    cols = [PROP if p else GREY for p in prop_mask]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", linewidth=1)
    if lo is not None:
        ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    right = hi if hi is not None else vals
    span = (xlim[1] - xlim[0]) if xlim else 1
    for yi, v, r in zip(y, vals, right):
        ax.text(r + 0.012 * span, yi, f"{v:.{digits}f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)


def fig5():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw=dict(wspace=0.75))
    for ax, s in zip(axes[:3], ["falls", "fog", "har70"]):
        t = T(f"T3_benchmark_{s}.csv")
        if t is None:
            ax.axis("off"); continue
        k = "AUROC" if s != "har70" else "MacroF1"
        lo_lim = {"falls": 0.95, "fog": 0.5, "har70": 0.5}[s]
        hbars(ax, [nm(m) for m in t.model], t[k].to_numpy(), t[f"{k}_lo"].to_numpy(), t[f"{k}_hi"].to_numpy(),
              ["CARES" in m for m in t.model], xlabel=("AUROC" if k == "AUROC" else "macro-F1") + " (95% person-bootstrap CI)",
              title=SVC[s], xlim=(lo_lim, 1.0 + (0.012 if s == "falls" else 0.12)), digits=4 if s == "falls" else 3)
    e = T("T4_ecg_arrhythmia.csv")
    if e is not None:
        a = e[(e.scenario.str.startswith("A")) & (e["eval"] == "DS2 (all beats)")]
        hbars(axes[3], [nm(m) for m in a.model], a.MacroF1_NSV.to_numpy(), prop_mask=["CARES" in m for m in a.model],
              xlabel="macro-F1 over N, S, V beats", title="Arrhythmia, inter-patient (DS2, 22 people)", xlim=(0, 1.12))
    save(fig, "Fig5_benchmark.png")


def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, s in zip(axes, ["falls", "fog", "har70"]):
        p = T(f"S2_per_subject_{s}.csv")
        if p is None:
            ax.axis("off"); continue
        k = "AUROC" if s != "har70" else "MacroF1"
        models = list(dict.fromkeys(p.model))
        data = [p[p.model == m][k].dropna().to_numpy() for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False, medianprops=dict(color=INK, linewidth=1.2))
        for b, m in zip(bp["boxes"], models):
            b.set_facecolor("#cfe0f6" if "CARES" in m else "#eceef0"); b.set_edgecolor(PROP if "CARES" in m else GREY_D)
        rng = np.random.default_rng(0)
        for i, (d, m) in enumerate(zip(data, models)):
            ax.scatter(i + 1 + rng.uniform(-0.13, 0.13, len(d)), d, s=10, color=PROP if "CARES" in m else GREY_D, zorder=3, linewidths=0)
        ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels([nm(m) for m in models], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(f"{'AUROC' if k == 'AUROC' else 'macro-F1'} of each held-out person"); ax.set_title(SVC[s])
    fig.tight_layout()
    save(fig, "Fig6_person_level.png")


def fig7():
    e = T("T4_ecg_arrhythmia.csv")
    if e is None:
        return
    e = e[e["eval"] == "DS2 after 5 min"]
    sel = [("A", "HGB", "global\nGBT"), ("A", "MLP", "global\nMLP"), ("A", "CARES (dual-view)", "global\nCARES"),
           ("C", "FedAvg MLP", "federated\nFedAvg"), ("B", "HGB", "adaptive\nGBT"), ("B", "MLP", "adaptive\nMLP"),
           ("C", "FedAvg MLP + on-device fine-tuning", "federated\n+ fine-tuning"), ("B", "CARES (dual-view)", "adaptive\nCARES")]
    rows = [(lab, e[(e.scenario.str.startswith(sc)) & (e.model == m)].iloc[0]) for sc, m, lab in sel
            if len(e[(e.scenario.str.startswith(sc)) & (e.model == m)])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.7))
    for ax, cls in zip(axes, ["S", "V"]):
        x = np.arange(len(rows))
        ax.bar(x - 0.19, [r[f"Se_{cls}"] for _, r in rows], 0.36, color=CAT[1], label="sensitivity", edgecolor="white")
        ax.bar(x + 0.19, [r[f"PPV_{cls}"] for _, r in rows], 0.36, color=CAT[6], label="positive predictive value", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([n for n, _ in rows], fontsize=7.3)
        ax.set_ylim(0, 1.05); ax.set_title({"S": "Supraventricular ectopic beats (S)", "V": "Ventricular ectopic beats (V)"}[cls])
        ax.axvline(3.5, color=GREY, lw=0.8, ls=":")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "Fig8_arrhythmia.png")


def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, s in zip(axes, ["fog", "har70", "falls"]):
        t = T(f"T5_federated_personalised_{s}.csv")
        if t is None:
            ax.axis("off"); continue
        k = "AUROC" if s != "har70" else "MacroF1"
        sc = t.set_index("method")[k]
        mins = [1, 2, 5]
        for name, col, mk, base in [("FedAvg + FT", CAT[1], "o", "FedAvg MLP"), ("Local only", CAT[2], "s", None),
                                    ("CARES + PC", PROP, "D", "CARES (dual-view)")]:
            vals = [sc.get(f"{name} ({m} min)", np.nan) for m in mins]
            if np.all(np.isnan(vals)):
                continue
            xs, ys = ([0] + mins, [sc.get(base)] + vals) if base else (mins, vals)
            ax.plot(xs, ys, marker=mk, ms=6, color=col,
                    label={"FedAvg + FT": "FedAvg + on-device fine-tuning", "Local only": "local-only MLP",
                           "CARES + PC": "CARES + personal recalibration"}[name])
        refs = [("Central MLP", "-", "centralised MLP"), ("FedProx MLP", "--", "FedProx"), ("FedAvg MLP (50% gateways online)", ":", "FedAvg, 50% gateways online")]
        for ref, ls, lab in refs:
            if ref in sc:
                ax.axhline(sc[ref], color=GREY_D, ls=ls, lw=1.1, label=lab)
        if s == "falls" and "CARES (dual-view)" in sc:
            ax.axhline(sc["CARES (dual-view)"], color=PROP, lw=1.6, label="CARES (central)")
            ax.axhline(sc["FedAvg MLP"], color=CAT[1], lw=1.6, label="FedAvg MLP")
            ax.set_xticks([])
        ax.set_xlabel("labelled onboarding minutes of the new person" if s != "falls" else "(no onboarding for the fall service)")
        ax.set_ylabel(("pooled AUROC" if k == "AUROC" else "pooled macro-F1") + " (new people)")
        ax.set_title({"fog": "Freezing of gait", "har70": "Activity monitoring (70+)", "falls": "Fall detection"}[s])
        ax.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "Fig9_federated_personalisation.png")


POLCOL = {"conformal (proposed)": PROP, "naive pooled calibration": CAT[1], "Youden (window-optimal)": CAT[2],
          "fixed 0.5": CAT[3]}


def fig9():
    """Coverage of the false-alarm budget with exact (Clopper-Pearson) 95% CIs, and event sensitivity."""
    from scipy import stats as st
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    for j, s in enumerate(["fog", "falls"]):
        t = T(f"T6_alarm_budget_{s}.csv")
        per = T(f"S7_alarm_per_subject_fog.csv" if s == "fog" else "S8_alarm_per_subject_falls.csv")
        if t is None:
            continue
        ax, ax2 = axes[2 * j], axes[2 * j + 1]
        pols = [p for p in POLCOL if p in set(t.policy)]
        budgets = sorted(t.budget_per_h.unique())
        w = 0.8 / len(pols)
        for i, p in enumerate(pols):
            g = t[t.policy == p].set_index("budget_per_h").reindex(budgets)
            xs = np.arange(len(budgets)) + i * w - 0.4 + w / 2
            ax.bar(xs, g.coverage, w * 0.92, color=POLCOL[p], label=p, edgecolor="white")
            lo, hi = [], []
            for B in budgets:
                q = per[(per.policy == p) & (per.budget_per_h == B)]
                k, n = int(q.within_budget.sum()), len(q)
                lo.append(st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0)
                hi.append(st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
            cov = g.coverage.to_numpy()
            ax.errorbar(xs, cov, yerr=[cov - np.array(lo), np.array(hi) - cov], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
            sens = g.episode_sensitivity if "episode_sensitivity" in g else g.fall_sensitivity
            ax2.plot(np.arange(len(budgets)), sens, marker="o", ms=5, color=POLCOL[p], label=p)
        ax.axhline(1 - ALARM["delta"], color=INK, ls="--", lw=1)
        ax.text(len(budgets) - 0.45, 1.02, "target 90%", ha="right", fontsize=7, color=INK)
        for a_ in (ax, ax2):
            a_.set_xticks(range(len(budgets))); a_.set_xticklabels([f"{b:g}" for b in budgets])
            a_.set_xlabel("false-alarm budget B (per hour" + (" of ADL)" if s == "falls" else ")"))
        n = per.subject.nunique()
        ax.set_ylabel(f"share of the {n} new people within budget"); ax.set_ylim(0, 1.08)
        ax.set_title(("FoG cueing" if s == "fog" else "Fall alarms (young volunteers)") + ": coverage, 95% CI")
        ax2.set_ylabel("freezing episodes detected" if s == "fog" else "falls detected")
        ax2.set_ylim((0, 1.03) if s == "fog" else (0.9, 1.005))
        ax2.set_title(("FoG cueing" if s == "fog" else "Fall alarms") + ": sensitivity")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for c in POLCOL.values()], list(POLCOL.keys()), loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "Fig10_alarm_budget.png")


def fig_fusion():
    """Weight ablation of the dual-view ensemble and reliability of the predicted probabilities."""
    W = T("T3b_fusion_weights.csv"); R = T("S20_reliability.csv")
    if W is None or R is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), gridspec_kw=dict(wspace=0.38))
    for ax, s, lab in [(axes[0], "fog", "pooled AUROC"), (axes[1], "har70", "pooled macro-F1")]:
        g = W[W.service == s]
        ax.plot(g.w_gbt, g.pooled, color=PROP, marker="o", ms=4, label="pooled")
        ax.plot(g.w_gbt, g.person_median, color=GREY_D, marker="s", ms=3.5, ls="--", label="median over people")
        ax.axvline(0.5, color=INK, lw=0.8, ls=":"); yl = ax.get_ylim(); ax.text(0.52, yl[0] + 0.30 * (yl[1] - yl[0]), "CARES\n(w = 0.5)", fontsize=7)
        ax.set_xlabel("weight of the GBT (descriptor) branch, w"); ax.set_ylabel(lab)
        ax.set_title(("(a) Freezing of gait" if s == "fog" else "(b) Activity of adults 70+") + ": fusion weight")
        ax.legend(fontsize=7, loc="lower left")
    show = {"CARES dual-view ensemble": PROP, "GBT (descriptors)": CAT[1], "MiniRocket-lite + LR": CAT[2], "CARES + Platt (inner folds)": CAT[6]}
    for ax, s, ttl in [(axes[2], "fog", "(c) Reliability, freezing of gait"), (axes[3], "har70", "(d) Reliability (top label), activity")]:
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
        for m, c in show.items():
            g = R[(R.service == s) & (R.model == m) & (R.n >= 20)]
            if len(g):
                ax.plot(g.confidence, g.observed, marker="o", ms=3.5, color=c, label=m)
        ax.set_xlabel("predicted probability" if s == "fog" else "confidence of the predicted class"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_title(ttl); ax.legend(fontsize=6.8, loc="upper left")
    save(fig, "Fig7_fusion_calibration.png")


def fig10():
    p = os.path.join(PRED_PATH, "bench_fog.npz")
    if not os.path.exists(p):
        return
    from lib.alarms import smooth, alarms
    d = np.load(p)
    w = np.load(os.path.join(PROCESSED_PATH, "fog_windows.npz"))
    eps = json.load(open(os.path.join(PROCESSED_PATH, "fog_episodes.json")))
    a = T("S7_alarm_per_subject_fog.csv")
    s, seg_k = 3, 1                       # a person with frequent freezing (segment = one recording run)
    m = d["subject"] == s
    t = w["t"][m] + 4.0
    pr = d["P::CARES (dual-view)"][m, 1]
    segs = np.split(np.arange(len(t)), np.where(np.diff(t) > 2)[0] + 1)
    g_ = segs[seg_k]
    t, pr = t[g_], pr[g_]
    t0 = t[0]
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.9), gridspec_kw=dict(height_ratios=[2, 1.4], hspace=0.5))
    ax = axes[0]
    ps = smooth(pr, ALARM["smooth_k"]["fog"])
    for e0, e1 in eps.get(str(s), []):
        if t[0] <= e0 <= t[-1]:
            ax.axvspan((e0 - t0) / 60, (e1 - t0) / 60, color="#fde3d6", lw=0)
    ax.plot((t - t0) / 60, ps, color=GREY_D, lw=0.9, label="CARES freezing probability (smoothed)")
    if a is not None:
        for B, ls, yv in [(4.0, "--", 1.10), (16.0, ":", 1.04)]:
            r = a[(a.policy == "conformal (proposed)") & (a.budget_per_h == B) & (a.subject == s)].threshold
            if len(r):
                th = float(r.iloc[0])
                ax.axhline(th, color=PROP, ls=ls, lw=1.2, label=f"conformal threshold, B = {B:g}/h ({th:.2f})")
                al = alarms(t, ps, th, ALARM["refractory_s"]["fog"])
                ax.plot((al - t0) / 60, np.full(len(al), yv), "v", color=PROP, ms=5, alpha=1.0 if B == 4 else 0.55,
                        label=f"cues at B = {B:g}/h")
    ax.set_ylim(0, 1.16); ax.set_ylabel("P(freeze)"); ax.set_xlabel("minutes")
    ax.set_title(f"Freezing-of-gait stream of held-out person {s} (shaded: annotated freezes)")
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax = axes[1]
    st = os.path.join(PRED_PATH, "stream_falls.npz")
    if os.path.exists(st):
        f = np.load(st)
        S = pd.DataFrame({k: f[k] for k in f.files})
        subj = 101
        for act, col, lab in [(919, CAT[1], "fall: syncope"), (808, CAT[2], "ADL: stumble"), (811, CAT[6], "ADL: sitting down hard on a chair")]:
            tri = S[(S.subject == subj) & (S.activity == act)].trial.unique()
            if len(tri):
                g = S[S.trial == tri[0]]
                ax.plot(g.t_end - g.t_peak.iloc[0], g.p, color=col, label=lab, lw=1.6)
        ax.axvline(0, color=GREY_D, lw=0.8, ls=":")
        ax.set_xlabel("seconds relative to the acceleration peak (alert time = end of window)"); ax.set_ylabel("P(fall)")
        ax.set_title(f"Streaming fall probability of held-out person {subj} for one fall and two fall-like activities")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    save(fig, "Fig11_example_streams.png")


def fig11():
    p = os.path.join(PRED_PATH, "cog_predictions.npz")
    t = T("T7_cognitive_surveillance.csv")
    if t is None or not os.path.exists(p):
        return
    d = np.load(p, allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25], wspace=0.45))
    fpr, tpr, _ = roc_curve(d["y"], d["p"])
    r = t[(t.task.str.startswith("A")) & (t.model.str.startswith("CARES"))].iloc[0]
    axes[0].plot(fpr, tpr, color=PROP, label=f"impaired vs. healthy 45+\n(ensemble, AUROC {r.AUROC:.2f})")
    axes[0].plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    axes[0].set_xlabel("1 − specificity"); axes[0].set_ylabel("sensitivity"); axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].set_title("Cross-validated ROC")
    g = pd.DataFrame(dict(group=d["group"], risk=d["p"]))
    order = ["middle-aged", "young-old", "old-old", "MCI", "dementia"]
    data = [g[g.group == o].risk.to_numpy() for o in order]
    bp = axes[1].boxplot(data, patch_artist=True, showfliers=False, widths=0.5, medianprops=dict(color=INK))
    for b, o in zip(bp["boxes"], order):
        b.set_facecolor("#fde3d6" if o in ("MCI", "dementia") else "#eceef0"); b.set_edgecolor(CAT[1] if o in ("MCI", "dementia") else GREY_D)
    for i, dd in enumerate(data):
        axes[1].scatter(i + 1 + np.random.default_rng(i).uniform(-0.13, 0.13, len(dd)), dd, s=8, color=GREY_D, zorder=3, linewidths=0)
    axes[1].set_xticks(range(1, 6)); axes[1].set_xticklabels([f"{o}\n(n = {len(x)})" for o, x in zip(order, data)], fontsize=7.5)
    axes[1].set_ylabel("cross-validated impairment risk"); axes[1].set_title("Risk by clinical group")
    imp = T("S11_cognitive_importance_by_task.csv")
    if imp is not None:
        imp.columns = ["task", "importance"]
        imp = imp.head(8)[::-1]
        names = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "casas_tasks.json")))
        axes[2].barh(range(len(imp)), imp.importance, color=GREY_D, height=0.6)
        axes[2].set_yticks(range(len(imp))); axes[2].set_yticklabels([names.get(x, x) for x in imp.task], fontsize=7.5)
        axes[2].set_xlabel("permutation importance (AUROC drop)"); axes[2].set_title("Most informative tasks (dementia/MCI vs. healthy)")
    save(fig, "Fig12_cognitive.png")


def fig12():
    titles = {"(a) sensor placement": "Sensor placement", "(b) sensor dropout": "Sensor lost at run time",
              "(c) sampling rate": "Sampling rate (battery)", "(d) sensor noise (test only)": "Noise at test time"}
    svc_col = {"falls": CAT[1], "fog": CAT[2], "har70": CAT[6]}
    svc_lab = {"falls": "falls (AUROC)", "fog": "FoG (AUROC)", "har70": "activity 70+ (macro-F1)"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw=dict(wspace=1.05))
    for ax, an in zip(axes, titles):
        labels, vals, cols = [], [], []
        for s in ["falls", "fog", "har70"]:
            t = T(f"T8_robustness_{s}.csv")
            if t is None:
                continue
            g = t[t.analysis == an]
            for _, r in g.iterrows():
                labels.append(f"{ {'falls': 'Falls', 'fog': 'FoG', 'har70': 'Activity'}[s]}: {r.setting}")
                vals.append(r.score); cols.append(svc_col[s])
        y = np.arange(len(labels))
        ax.scatter(vals, y, c=cols, s=28, zorder=3)
        for yi, v in zip(y, vals):
            ax.plot([0, v], [yi, yi], color="#e3e5e8", lw=1, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.4); ax.invert_yaxis()
        ax.set_xlim(min(vals) - 0.05 if vals else 0, 1.01)
        ax.set_title(titles[an]); ax.set_xlabel("score")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=c) for c in svc_col.values()], svc_lab.values(), loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    save(fig, "Fig13_robustness.png")


def fig13():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(wspace=0.45))
    for ax, s in zip(axes[:2], ["falls", "fog"]):
        t = T(f"T9_alert_explanations_{s}.csv")
        if t is None:
            ax.axis("off"); continue
        pv = t.pivot_table(index="site", columns="family", values="alerts", aggfunc="sum").fillna(0)
        pv = pv / pv.values.sum() * 100
        ax.imshow(pv.values, cmap="Blues", vmin=0, vmax=max(1, pv.values.max()))
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=25, ha="right", fontsize=7.5)
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=8)
        for i in range(pv.shape[0]):
            for k in range(pv.shape[1]):
                v = pv.values[i, k]
                ax.text(k, i, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white" if v > 0.6 * pv.values.max() else INK)
        ax.set_title(("Fall alerts" if s == "falls" else "FoG alerts") + ": dominant evidence")
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[2]
    rows = [(s, T(f"T9b_explanation_faithfulness_{s}.csv").iloc[0]) for s in ["falls", "fog"] if T(f"T9b_explanation_faithfulness_{s}.csv") is not None]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r.mean_drop_top for _, r in rows], 0.36, color=PROP, label="remove top-attributed group", edgecolor="white")
    ax.bar(x + 0.19, [r.mean_drop_random for _, r in rows], 0.36, color=GREY, label="remove a random group", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["falls" if s == "falls" else "FoG" for s, _ in rows])
    ax.set_ylabel("mean drop of alert probability"); ax.set_title("Deletion faithfulness"); ax.legend(fontsize=7.5)
    save(fig, "Fig14_explanations.png")


def fig14():
    bw, sim = T("T11_bandwidth.csv"), os.path.join(PRED_PATH, "alert_delivery_sim.npz")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9), gridspec_kw=dict(wspace=0.3))
    if bw is not None:
        x = np.arange(len(bw))
        for i, (c, lab, col) in enumerate([("raw_MB_per_day", "raw streaming to the cloud", GREY_D),
                                            ("features_MB_per_day", "descriptor streaming", GREY),
                                            ("cares_MB_per_day", "IoT-CARES edge (alerts, summaries, 1 model update)", PROP)]):
            axes[0].bar(x + (i - 1) * 0.27, bw[c], 0.25, color=col, label=lab, edgecolor="white")
        axes[0].set_yscale("log"); axes[0].set_xticks(x)
        axes[0].set_xticklabels([{"fog": "FoG", "har70": "activity 70+", "falls": "falls", "ecg": "ECG patch"}.get(s, s) for s in bw.service])
        axes[0].set_ylabel("MB leaving the home per person-day (log)"); axes[0].legend(fontsize=7.3, loc="lower left")
        axes[0].set_title("Data minimisation")
    if os.path.exists(sim):
        d = np.load(sim)
        keys = [k for k in d.files if k.startswith("loss")]
        blues = ["#9ec5f0", "#6aa6e6", "#2a78d6", "#1d5aa6", "#123b70"]
        for k, col in zip(keys, blues):
            v = np.sort(d[k])
            axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.6, label=f"packet {k}")
        for k in d.files:
            if k.startswith("cloud"):
                v = np.sort(d[k])
                axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=CAT[1], lw=1.6, ls="--", label=k)
        axes[1].set_xscale("log"); axes[1].set_xlabel("seconds from fall impact to caregiver notification (log)")
        axes[1].set_ylabel("cumulative share of alerts"); axes[1].legend(fontsize=7.3, loc="lower right")
        axes[1].axvline(60, color=GREY, ls=":", lw=1)
        axes[1].set_title("Simulated alert delivery (20,000 alerts per scenario)")
    save(fig, "Fig15_edge_iot.png")


def fig15():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(wspace=0.38))
    p = os.path.join(PRED_PATH, "bench_har70.npz")
    if os.path.exists(p):
        d = np.load(p)
        w = np.load(os.path.join(PROCESSED_PATH, "har70_windows.npz"))
        cls = list(w["classes"])
        cm = confusion_matrix(d["y"], d["P::CARES (dual-view)"].argmax(1), labels=range(len(cls)), normalize="true")
        _cm(axes[0], cm, cls, "(a) Activity, adults 70+ (CARES)")
    p = os.path.join(PRED_PATH, "bench_ecg.npz")
    if os.path.exists(p):
        d = np.load(p)
        ds2 = np.load(os.path.join(PROCESSED_PATH, "ecg_beats.npz"))["ds2"]
        for ax, key, ttl in [(axes[1], "PA::CARES (dual-view)", "(b) ECG DS2, global CARES"), (axes[2], "PB::CARES (dual-view)", "(c) ECG DS2, patient-adaptive CARES")]:
            P = d[key]
            m = (d["t"] >= 300) & np.isin(d["record"], ds2)
            cm = confusion_matrix(d["y"][m], P[m].argmax(1), labels=range(5))
            _cm(ax, cm / np.maximum(cm.sum(1, keepdims=True), 1), ["N", "S", "V", "F", "Q"], ttl,
                counts=np.bincount(d["y"][m], minlength=5))
    save(fig, "Fig16_confusion.png")
    # supplementary: confusion matrices of every global ECG model
    if os.path.exists(p):
        d = np.load(p)
        ms = [("PA::LR", "LR"), ("PA::RF", "RF"), ("PA::HGB", "GBT"), ("PA::MLP", "MLP"), ("PA::MiniRocket-LR", "MiniRocket-lite + LR"), ("PA::CARES (dual-view)", "CARES")]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw=dict(wspace=0.35, hspace=0.45))
        for ax, (k, lab) in zip(axes.ravel(), ms):
            P = d[k]
            m = (d["t"] >= 300) & np.isin(d["record"], np.load(os.path.join(PROCESSED_PATH, "ecg_beats.npz"))["ds2"])
            cm = confusion_matrix(d["y"][m], P[m].argmax(1), labels=range(5))
            _cm(ax, cm / np.maximum(cm.sum(1, keepdims=True), 1), ["N", "S", "V", "F", "Q"], f"Global {lab}, DS2",
                counts=np.bincount(d["y"][m], minlength=5))
        save(fig, "FigS1_ecg_global_confusion.png")


def _cm(ax, cm, cls, title, counts=None):
    ax.imshow(np.nan_to_num(cm), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            v = cm[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ylab = cls if counts is None else [f"{c} (n = {int(n):,})" for c, n in zip(cls, counts)]
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=30, ha="right"); ax.set_yticks(range(len(cls))); ax.set_yticklabels(ylab)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title + ", row-normalised", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig16():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for ax, s in zip(axes, ["fog", "har70", "falls"]):
        c = T(f"S6_fedavg_convergence_{s}.csv")
        if c is None or len(c) == 0:
            ax.axis("off"); continue
        for sid, g in c.groupby("subject"):
            ax.plot(g["round"], g.score, color="#d5d8dc", lw=0.8)
        m = c.groupby("round").score.median()
        ax.plot(m.index, m.values, color=CAT[1], lw=2, label="median over held-out people")
        ax.set_xlabel("federated round"); ax.set_ylabel("AUROC" if s != "har70" else "macro-F1")
        ax.set_title({"fog": "Freezing of gait", "har70": "Activity monitoring (70+)", "falls": "Fall detection"}[s])
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "Fig17_fedavg_curves.png")


if __name__ == "__main__":
    for f in [fig5, fig6, fig_fusion, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16]:
        try:
            f()
        except Exception as ex:  # keep going: a missing result only skips its figure
            print("skipped", f.__name__, "-", repr(ex))
