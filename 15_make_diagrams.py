"""
STEP 15 - Architecture and flow diagrams (Graphviz `dot`, 300 dpi PNG + .dot sources).
  Fig1_architecture.png      four-layer IoT-CARES architecture (sensing, edge, federation/cloud, care)
  Fig2_study_design.png      datasets, monitoring services, protocols and analyses
  Fig3_processing_flow.png   per-window processing pipeline inside the home gateway
  Fig4_federated_calibration.png   federated training, onboarding personalisation and conformal alarm calibration
"""
import os
import subprocess

import pandas as pd

from config import FIG_PATH, TAB_PATH

FONT = "Helvetica"


def render(name, src):
    p = os.path.join(FIG_PATH, name)
    open(p.replace(".png", ".dot"), "w").write(src)
    subprocess.run(["dot", "-Tpng", "-Gdpi=300", "-o", p, p.replace(".png", ".dot")], check=True)
    print("wrote", p)


def architecture():
    src = f'''digraph G {{
 rankdir=TB; compound=true; nodesep=0.22; ranksep=0.42; newrank=true;
 node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=10, color="#37474F", fillcolor="#FFFFFF"];
 edge [color="#455A64", arrowsize=0.6, fontname="{FONT}", fontsize=8];
 subgraph cluster_s {{ label="1  Sensing layer (home)"; fontname="{FONT}"; fontsize=12; style="rounded,filled"; fillcolor="#E3F2FD"; color="#90CAF9";
   w1 [label="Belt / pendant IMU\\n(waist acc + gyro)"]; w2 [label="Smartwatch IMU\\n(wrist acc + gyro)"];
   w3 [label="Leg / trunk accelerometers\\n(gait & FoG cueing kit)"]; w4 [label="Single-lead ECG patch"];
   a1 [label="Ambient sensors\\nPIR motion, door, item use\\n(no camera)"]; }}
 subgraph cluster_e {{ label="2  Edge layer (home gateway)"; fontname="{FONT}"; fontsize=12; style="rounded,filled"; fillcolor="#E8F5E9"; color="#A5D6A7";
   e1 [label="Windowing + edge features\\n(time, spectral, gait regularity)"];
   e2 [label="MiniRocket-lite\\nrandom-kernel transform"];
   e3 [label="CARES dual-view ensemble\\n(boosted trees + linear model, averaged)", fillcolor="#C8E6C9"];
   e4 [label="Calibrated alarm manager\\nsmoothing, refractory, conformal threshold", fillcolor="#FFF59D"];
   e5 [label="Alert explanation\\n(body site × feature family)"];
   e6 [label="Federated client\\n(local training, personal fine-tuning)"];
   e7 [label="Local siren / voice prompt\\n(works offline)"]; }}
 subgraph cluster_c {{ label="3  Federation & cloud layer"; fontname="{FONT}"; fontsize=12; style="rounded,filled"; fillcolor="#F3E5F5"; color="#CE93D8";
   c1 [label="Federation server\\nFedAvg / FedProx aggregation"];
   c2 [label="Calibration service\\nper-person thresholds → conformal quantile"];
   c3 [label="MQTT broker (QoS 1)\\n+ event store"];
   c4 [label="Longitudinal analytics\\nmobility, ectopy burden,\\ncognitive-health risk"]; }}
 subgraph cluster_k {{ label="4  Care layer"; fontname="{FONT}"; fontsize=12; style="rounded,filled"; fillcolor="#FFF3E0"; color="#FFCC80";
   k1 [label="Family caregiver app\\n(push + SMS fallback)"]; k2 [label="Nurse / clinician dashboard"];
   k3 [label="Emergency escalation\\n(no acknowledgement in 60 s)"]; }}
 {{rank=same; w1; w2; w3; w4; a1;}}
 {{rank=same; e1; e2; e6;}}
 {{rank=same; e3;}}
 {{rank=same; e4; e7; e5;}}
 {{rank=same; c1; c2; c3; c4;}}
 {{rank=same; k1; k2; k3;}}
 w1 -> e1 [label=" BLE"]; w2 -> e1; w3 -> e1; w4 -> e1; a1 -> e1 [label=" Zigbee"];
 e1 -> e3; e1 -> e2 -> e3; e3 -> e4 -> e5; e4 -> e7;
 e6 -> c1 [dir=both, style=dashed, label=" model updates only"];
 e4 -> c2 [dir=both, style=dashed, label=" thresholds"];
 e5 -> c3 [label=" alert + explanation\\n (~256 B)"]; e3 -> c4 [style=dotted, label=" hourly summaries"];
 c3 -> k1; c3 -> k2; k1 -> k3 [style=dashed]; c4 -> k2;
}}'''
    render("Fig1_architecture.png", src)


def processing_flow():
    src = f'''digraph G {{
 rankdir=TB; nodesep=0.3; ranksep=0.32;
 node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=10, color="#37474F", fillcolor="#FFFFFF"];
 edge [color="#455A64", arrowsize=0.6, fontname="{FONT}", fontsize=8];
 s [label="Raw sensor stream x(t)\\n(25-64 Hz IMU, 360 Hz ECG, ambient events)", fillcolor="#E3F2FD"];
 w [label="Sliding window (FoG 4 s / 1 s hop, falls 4 s / 0.4 s, activity 5 s / 2.5 s, ECG one beat)"];
 f [label="View 1: edge descriptors (22 per axis + magnitude, inter-axis correlation)\\nmean, SD, percentiles, RMS, jerk, spectral entropy, band powers, freeze index, ACF peak", fillcolor="#E8F5E9"];
 r [label="View 2: MiniRocket-lite (84 kernels × 4 dilations × channels)\\nproportion of positive values of each random convolution", fillcolor="#E8F5E9"];
 c [label="CARES dual-view ensemble (fixed, unweighted average)\\np = ½ [ GBT(view 1) + LR(view 2) ]", fillcolor="#C8E6C9"];
 a [label="Alarm manager: causal smoothing (k windows) → threshold θ → refractory period", fillcolor="#FFF59D"];
 th [label="θ = conformal quantile of per-person thresholds\\nP(false alarms of a new person > B per hour) ≤ δ\\n(if people are exchangeable and scored by the same model)", shape=note, fillcolor="#FFFDE7"];
 x [label="Explanation: group occlusion → top body site & feature family", fillcolor="#FCE4EC"];
 o [label="Alert message {{service, time, risk, explanation}} → caregiver; local siren if offline", fillcolor="#FFEBEE"];
 s -> w; w -> f; w -> r; f -> c; r -> c; c -> a; th -> a [style=dashed]; a -> x -> o;
}}'''
    render("Fig3_processing_flow.png", src)


def study_design():
    def n(path, col, default="?"):
        try:
            return pd.read_csv(os.path.join(TAB_PATH, path))
        except Exception:  # noqa
            return None
    src = f'''digraph G {{
 rankdir=LR; nodesep=0.22; ranksep=0.45;
 node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=9.5, color="#37474F", fillcolor="#FFFFFF"];
 edge [color="#455A64", arrowsize=0.6];
 subgraph cluster_d {{ label="Open datasets (no new data collected)"; fontname="{FONT}"; style="rounded,dashed"; color="#90A4AE";
  d1 [label="UCI Simulated Falls & ADL\\n17 people, 20 falls + 16 ADLs,\\n6 IMUs at 25 Hz", fillcolor="#E3F2FD"];
  d2 [label="UCI HAR70+\\n18 adults aged 70-95,\\n2 accelerometers at 50 Hz", fillcolor="#E3F2FD"];
  d3 [label="UCI Daphnet FoG\\n10 people with Parkinson's,\\n3 accelerometers at 64 Hz", fillcolor="#E3F2FD"];
  d4 [label="PhysioNet MIT-BIH\\n44 non-paced records,\\nlead MLII at 360 Hz", fillcolor="#E3F2FD"];
  d5 [label="CASAS smart apartment\\n350 people incl. dementia / MCI,\\nambient sensors", fillcolor="#E3F2FD"]; }}
 subgraph cluster_v {{ label="Monitoring services"; fontname="{FONT}"; style="rounded,dashed"; color="#90A4AE";
  v1 [label="S1 fall detection\\n+ calibrated alarms", fillcolor="#E8F5E9"]; v2 [label="S2 mobility / activity\\nprofile of older adults", fillcolor="#E8F5E9"];
  v3 [label="S3 freezing-of-gait\\ndetection & cueing", fillcolor="#E8F5E9"]; v4 [label="S4 arrhythmia &\\nectopy burden", fillcolor="#E8F5E9"];
  v5 [label="S5 cognitive-health\\nsurveillance", fillcolor="#E8F5E9"]; }}
 subgraph cluster_p {{ label="Evaluation protocols"; fontname="{FONT}"; style="rounded,dashed"; color="#90A4AE";
  p1 [label="Leave-one-person-out\\n(inner subject-grouped CV for\\nthresholds & calibration)", fillcolor="#FFF8E1"];
  p2 [label="Inter-patient DS1 → DS2\\n(AAMI EC57)", fillcolor="#FFF8E1"];
  p3 [label="Repeated stratified\\n5-fold CV (10 x)", fillcolor="#FFF8E1"]; }}
 subgraph cluster_a {{ label="Analyses"; fontname="{FONT}"; style="rounded,dashed"; color="#90A4AE";
  a1 [label="Model benchmark (7 detectors)"]; a2 [label="Federated learning & on-device\\npersonalisation (onboarding 1-5 min)"];
  a3 [label="Alarm budget guarantee\\n(conformal vs. naive thresholds,\\nsplit conformal, older-adult ADL check)"]; a4 [label="Explainability, sensor placement,\\ndropout, sampling rate, noise"];
  a5 [label="Edge latency, data minimisation,\\nalert-delivery simulation"]; a6 [label="Fusion ablation and\\nprobability calibration"]; }}
 d1 -> v1; d2 -> v2; d3 -> v3; d4 -> v4; d5 -> v5;
 v1 -> p1; v2 -> p1; v3 -> p1; v4 -> p2; v5 -> p3;
 p1 -> a1; p1 -> a2; p1 -> a3; p1 -> a4; p2 -> a1; p2 -> a2; p3 -> a1; p1 -> a5; p1 -> a6;
}}'''
    render("Fig2_study_design.png", src)


def federated_calibration():
    src = f'''digraph G {{
 rankdir=TB; nodesep=0.3; ranksep=0.35; newrank=true; compound=true;
 node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=10, color="#37474F", fillcolor="#FFFFFF"];
 edge [color="#455A64", arrowsize=0.6, fontname="{FONT}", fontsize=8.5];
 subgraph cluster_f {{ label="A. Federated training across homes\n(raw data never leave a home)"; fontname="{FONT}"; style="rounded,filled"; fillcolor="#F3E5F5"; color="#CE93D8";
  srv [label="Server aggregation\nw(r+1) = w(r) + Σ_k (n_k / n) Δ_k\nFedProx: + μ/2 ‖w − w(r)‖²"];
  g1 [label="Gateway 1\none local epoch"]; g2 [label="Gateway 2"]; gk [label="Gateway K\n(may be offline)", style="rounded,dashed"];
  srv -> g1 [label=" w(r)"]; srv -> g2; srv -> gk [style=dashed]; g1 -> srv [label=" Δ_1"]; g2 -> srv [label=" Δ_2"]; }}
 subgraph cluster_o {{ label="B. Onboarding of a new person\n(first 1–5 minutes, supervised)"; fontname="{FONT}"; style="rounded,filled"; fillcolor="#E8F5E9"; color="#A5D6A7";
  on [label="Labelled onboarding minutes\nFoG: walking course\nactivity: usual routine\nECG: first 5 min\nfalls: ADL only (alarm threshold)"];
  ft [label="On-device personalisation\nfine-tune MLP (all / last layer,\n± class-balanced replay), or\nPlatt-recalibrate CARES"]; on -> ft; }}
 subgraph cluster_c {{ label="C. Alarm calibration with a\nper-person guarantee"; fontname="{FONT}"; style="rounded,filled"; fillcolor="#FFFDE7"; color="#FFE082";
  oof [label="Out-of-fold scores of\nn calibration people"]; tj [label="θ_j = smallest threshold\nwith FA_j(θ) ≤ B"];
  q [label="θ̂ = ⌈(n+1)(1−δ)⌉-th\nsmallest θ_j\n(= largest θ_j for n ≤ 18, δ = 0.1)"];
  gu [label="P( FA_new(θ̂) ≤ B ) ≥ 1 − δ\nif (A1) people are exchangeable and\n(A2) all are scored by the same model", shape=note, fillcolor="#FFF59D"];
  oof -> tj -> q -> gu; }}
 {{rank=same; srv; on; oof;}}
 {{rank=same; g1; g2; gk; ft; tj;}}
 srv -> on [label=" global model", constraint=false];
}}'''
    render("Fig4_federated_calibration.png", src)


if __name__ == "__main__":
    architecture(); processing_flow(); study_design(); federated_calibration()
