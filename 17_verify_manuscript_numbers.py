"""
STEP 17 (optional) - Check that the Word manuscript quotes the generated numbers.

Extracts the text of the clean revised manuscript (00_PAPER/ or manuscript/IoT-CARES_Revised_Clean.docx; falls back to
IoT-CARES_Manuscript.docx) with pandoc and checks that the headline metrics of every service, rounded as in the paper
(Section 3.8: values in [0, 1] with three decimals, trial-level fall AUROC with four, percentages with one, false-alarm
rates with two), occur in the text.
"""
import os
import subprocess
import sys

import pandas as pd

from config import TAB_PATH, ROOT

NAMES = ["IoT-CARES_Revised_Clean.docx", "IoT-CARES_Manuscript.docx"]


def main():
    cands = [os.path.join(ROOT, d, n) for n in NAMES for d in ("00_PAPER", "manuscript")]
    if len(sys.argv) > 1:
        cands.insert(0, sys.argv[1])
    docx = next((p for p in cands if os.path.exists(p)), None)
    if docx is None:
        print("manuscript not found"); return 1
    txt = subprocess.run(["pandoc", "-t", "plain", "--wrap=none", docx], capture_output=True, text=True).stdout
    txt = txt.replace("−", "-")
    checks, missing = 0, []

    def want(s, label):
        nonlocal checks
        checks += 1
        if s not in txt:
            missing.append((label, s))

    def tab(name):
        return pd.read_csv(os.path.join(TAB_PATH, name + ".csv"))

    for s, k, nd in [("falls", "AUROC", 4), ("fog", "AUROC", 3), ("har70", "MacroF1", 3)]:
        for _, r in tab(f"T3_benchmark_{s}").iterrows():
            want(f"{r[k]:.{nd}f}", f"{s} {r.model} {k}")
    for _, r in tab("T4_ecg_arrhythmia").iterrows():
        if r["eval"] == "DS2 after 5 min" and r.model != "MiniRocket-LR":          # the ECG table omits this branch
            want(f"{r.MacroF1_NSV:.3f}", f"ecg {r.scenario} {r.model}")
    shown = {("FoG", "fixed 0.5", 4), ("FoG", "Youden (window-optimal)", 4), ("FoG", "naive pooled calibration", 4),
             ("FoG", "naive pooled calibration", 16), ("FoG", "conformal (proposed)", 4), ("FoG", "conformal (proposed)", 16),
             ("Falls", "fixed 0.5", 0), ("Falls", "Youden (window-optimal)", 0), ("Falls", "conformal (proposed)", 0),
             ("Falls", "conformal (proposed)", 2), ("Falls (split conformal)", "split conformal (exact exchangeability)", 0),
             ("Falls (split conformal)", "split conformal (exact exchangeability)", 2)}      # rows of the alarm table
    for _, r in tab("T6e_coverage_ci").iterrows():
        if (r.service, r.policy, r.budget_per_h) not in shown and not r.service.startswith("Falls, magnitude"):
            continue
        want(f"{100 * r.coverage:.1f}%", f"alarm {r.service} {r.policy} B={r.budget_per_h} coverage")
        want(f"{int(r.covered)}/{int(r.people)}", f"alarm {r.service} {r.policy} B={r.budget_per_h} count")
    for _, r in tab("T6f_false_alarm_distribution").iterrows():
        if (r.service, r.policy, r.budget_per_h) not in shown and not r.service.startswith("Falls, magnitude"):
            continue
        want(f"{r.pooled_fa_per_h:.2f}", f"alarm {r.service} {r.policy} B={r.budget_per_h} pooled FA/h")
    for _, r in tab("T7_cognitive_surveillance").iterrows():
        want(f"{r.AUROC:.3f}", f"cog {r.task} {r.model}")
    for _, r in tab("T3b_fusion_weights").iterrows():
        if r.w_gbt in (0.0, 0.5, 1.0):
            want(f"{r.pooled:.{4 if r.service == 'falls' else 3}f}", f"fusion {r.service} w={r.w_gbt}")
    for _, r in tab("T3d_calibration").iterrows():
        if r.model in ("CARES dual-view ensemble", "CARES + Platt (inner folds)", "GBT (descriptors)", "MiniRocket-lite + LR"):
            want(f"{r.ECE:.3f}", f"ECE {r.service} {r.model}")
    for s, k in [("fog", "AUROC"), ("har70", "MacroF1")]:
        p = os.path.join(TAB_PATH, f"T5b_personalisation_revision_{s}.csv")
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                want(f"{r[k]:.3f}", f"remedy {s} {r.method} [{r.eval_set}]")
    p = os.path.join(TAB_PATH, "T6c_falls_older_adults.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            want(f"{int(r.covered)}/{int(r.people)}", f"older adults {r.calibration} {r.population} B={r.budget_per_h}")
    print(f"{docx}\n{checks - len(missing)}/{checks} numbers found in the manuscript text")
    for m in missing:
        print("  NOT FOUND:", m)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
