// Back matter (MDPI): supplementary materials, statements, data availability, abbreviations, Appendix A.
const L = require("./lib");
const { P, H1, Tab, hl } = L;

module.exports = function (N) {
  const out = [];
  const sec = (t) => new L.Paragraph({ spacing: { before: 120, after: 40 }, children: L.runs(`**${t}**`) });
  out.push(sec("Supplementary Materials:"));
  out.push(P("The following supporting information is provided with this article: Tables S1–S12 (feature-extraction timing, person-level results of every model, federated seeds, ECG burden per record, federated person-level results, learning curves, alarm results per person for FoG and falls, fall-alarm results per activity, cognitive risk by group, informative tasks and features, and simulation assumptions){+; Tables S13–S23 added in the revision (S13 per-class ECG metrics for all five AAMI classes; S14 ECG confusion counts; S15 sensitivity analysis without records 201 and 202; S16 DS1-only hyper-parameter selection; S17 personalisation remedies per person and onboarding class counts; S18 split-conformal fall alarms per person; S19 fall alarms on older adults' free-living data per person; S20 reliability-curve bins; S21 calibration size and conformal quantile index per fold; S22 false alarms and monitored hours per person; S23 personal ADL margin scenarios); and Figure S1 (confusion matrices of every global ECG model)+}. The complete code (numbered scripts 01–{+22+}), the raw and processed data, and all generated results (CSV tables, figures, predictions and logs) are provided in the supplementary archive IoTCARES_complete.zip."));
  out.push(sec("Author Contributions:"));
  out.push(P("Conceptualization, A.B.; methodology, A.B.; software, A.B.; validation, A.B.; formal analysis, A.B.; investigation, A.B.; data curation, A.B.; writing—original draft preparation, A.B.; writing—review and editing, A.B. and [co-authors]; visualization, A.B.; supervision, [name]; project administration, [name]; funding acquisition, [name]. All authors have read and agreed to the published version of the manuscript."));
  out.push(sec("Funding:"));
  out.push(P("This research received no external funding. [Please update if applicable.]"));
  out.push(sec("Institutional Review Board Statement:"));
  out.push(P("Ethical review and approval were waived for this study because it is a secondary analysis of fully de-identified, publicly available datasets whose original collection was approved by the responsible ethics committees, as described in the original publications [@ozdemir2014;@ustad2023;@bachlin2010;@moody2001;@dawadi2013]."));
  out.push(sec("Informed Consent Statement:"));
  out.push(P("Not applicable. Only de-identified public data were analysed; informed consent was obtained by the original investigators."));
  out.push(sec("Data Availability Statement:"));
  out.push(P("This study used five publicly available, de-identified datasets; no new data were collected. (1) Simulated Falls and Daily Living Activities Data Set, UCI Machine Learning Repository, https://doi.org/10.24432/C52028 (CC BY 4.0) [@ozdemir2014]. (2) HAR70+, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/780/har70 (CC BY 4.0) [@ustad2023]. (3) Daphnet Freezing of Gait, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait (CC BY 4.0) [@bachlin2010]. (4) MIT-BIH Arrhythmia Database v1.0.0, PhysioNet, https://physionet.org/content/mitdb/1.0.0/ (Open Data Commons Attribution License v1.0) [@moody2001;@goldberger2000]. (5) CASAS smart home dataset: scripted complex activities, activity scores and cognitive diagnosis, Zenodo, https://doi.org/10.5281/zenodo.15713579 (CC BY 4.0) [@cook2013;@dawadi2013]. The exact files used (with SHA-256 checksums), the extracted accelerometer/gyroscope subset of the falls dataset, all processing and analysis code, processed features, predictions and result tables are available in the supplementary archive. Every number in this article can be regenerated with the numbered scripts; dataset and result paths are set in config.py. [Optional: add the permanent repository link, e.g., Zenodo DOI or GitHub URL.]"));
  out.push(sec("Acknowledgments:"));
  out.push(P("The authors thank the creators of the UCI Simulated Falls and Daily Living Activities, HAR70+, Daphnet, MIT-BIH and CASAS datasets for making their data openly available. During the preparation of this study, the authors used an AI assistant (Claude, Anthropic) to help write analysis code and draft and edit text. The authors reviewed, verified and edited all code, results and text, and take full responsibility for the content of this publication."));
  out.push(sec("Conflicts of Interest:"));
  out.push(P("The authors declare no conflicts of interest."));
  out.push(sec("Abbreviations:"));
  const abbr = [["AAMI", "Association for the Advancement of Medical Instrumentation"], ["ADL", "activity of daily living"], ["AUPRC / AUROC", "area under the precision–recall / receiver-operating-characteristic curve"],
    ["BLE", "Bluetooth Low Energy"], ["CARES", "Calibrated, Adaptive, Resource-aware, Explainable Surveillance (framework); CARES dual-view ensemble (classifier)"],
    ["CASAS", "Center for Advanced Studies in Adaptive Systems"], ["CI", "confidence interval"], ["CPU", "central processing unit"], ["CV", "cross-validation"],
    ["DS1 / DS2", "development / test half of the inter-patient MIT-BIH split"], ["ECE", "expected calibration error"], ["ECG", "electrocardiogram"],
    ["FA", "false alarm"], ["FedAvg / FedProx", "federated averaging / federated proximal optimisation"], ["FI", "freeze index"],
    ["FoG", "freezing of gait"], ["GBT", "gradient-boosted trees"], ["IMU", "inertial measurement unit"], ["IoT", "Internet of Things"],
    ["LIME", "Local Interpretable Model-agnostic Explanations"], ["LOPO", "leave-one-person-out"], ["LR", "logistic regression"], ["MCI", "mild cognitive impairment"],
    ["MIT-BIH", "Massachusetts Institute of Technology–Beth Israel Hospital"], ["MLII", "modified limb lead II"], ["MLP", "multilayer perceptron"],
    ["MQTT", "Message Queuing Telemetry Transport"], ["PD", "Parkinson's disease"], ["PPV", "positive predictive value"], ["RF", "random forest"],
    ["S / V / F / N / Q", "AAMI heartbeat classes: supraventricular ectopic, ventricular ectopic, fusion, normal, unclassifiable"],
    ["SD", "standard deviation"], ["Se", "sensitivity"], ["SHAP", "SHapley Additive exPlanations"], ["SMS", "short message service"],
    ["TRIPOD+AI", "Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, AI extension"], ["UCI", "University of California, Irvine (Machine Learning Repository)"]];
  const old = ["ADL", "AUPRC / AUROC", "BLE", "CI", "FA", "FedAvg / FedProx", "FI", "FoG", "GBT", "IMU", "IoT", "LOPO", "LR", "MCI", "MLP", "MQTT", "RF", "S / V / F / N / Q"];
  const abbrRows = abbr.map(([k, v]) => (old.includes(k) ? [k, v] : [hl(k), hl(v)]));
  out.push(...Tab("Abbreviations used in this manuscript.", ["Abbreviation", "Meaning"], abbrRows, [24, 76], { key: "abbr" }));
  // ------------------------------------------------------------------ Appendix A
  out.push(H1("Appendix A. Hyper-Parameters and Reproducibility"));
  const hp = [
    ["Windows", "FoG 4 s / 1 s hop (64 Hz); activity 5 s / 2.5 s (50 Hz); falls 4 s, peak-centred (benchmark) or every 0.4 s (streaming), 25 Hz"],
    ["Edge descriptors", "22 per signal × (3 axes + magnitude) + 3 axis correlations + signal-magnitude area = 92 per sensor"],
    ["MiniRocket-lite", "84 kernels × dilations {1, 2, 4, 8} × channels, one bias per kernel (random quantile in [0.1, 0.9] of training outputs)"],
    ["GBT", "300 iterations, learning rate 0.06, 31 leaves, L2 = 1.0, no early stopping {+(shared by all services; not tuned on held-out people or DS2)+}"],
    ["RF / LR", "300 trees, min. 2 samples per leaf, balanced class weights / C = 0.5 (descriptors), C = 0.01 (random-convolution features)"],
    ["MLP", "128–64 ReLU units, dropout 0.1, Adam (lr 10⁻³), batch 256, weight decay 10⁻⁵, class-balanced loss, 12 epochs (benchmark)"],
    ["Federated", "40 rounds, 1 local epoch, lr 2 × 10⁻³, FedProx μ = 0.01, 50% participation variant, seeds 11, 23, 47; central reference 13 epochs"],
    ["Personalisation", "onboarding 1, 2, 5 min; fine-tuning 15 epochs (all layers); Platt recalibration; ECG: first 5 min, ×5 up-weighting{+; remedies: output layer only, class-balanced replay of 200 windows per class (Gaussian from federated class statistics, or real windows), guided onboarding+}"],
    ["Alarm manager", "δ = 0.10; smoothing k = 3 (FoG), 1 (falls); refractory 10 s (FoG), 30 s (falls); threshold grid of 191 values in [0, 1]{+; calibration people n = 9 (FoG), 16 (falls), 10 (falls, split conformal), 17 (older adults); quantile index = n in every case+}"],
    ["Statistics", "person-cluster bootstrap (500), Wilcoxon signed-rank with Holm correction; inner 3-fold person-grouped CV for thresholds{+; Clopper–Pearson and exact McNemar for coverage; exact Poisson CI for pooled FA/h; ECE with 15 equal-width bins+}"],
    ["Software", "Python 3, NumPy, SciPy, pandas, scikit-learn, Matplotlib; Graphviz for diagrams; no GPU or deep-learning framework"]];
  out.push(...Tab("Hyper-parameters of IoT-CARES (all fixed before evaluation).", ["Component", "Setting"], hp, [20, 80], { key: "hyper" }));
  out.push(...appendixB(N));
  return out;
};


// Appendix B (revision): which configuration and evaluation set each headline number refers to.
function appendixB(N) {
  const U = require("./results_util")(N);
  const { T, f } = U;
  const g = (t, m, k) => T(t).find((r) => r.model === m || r.method === m)[k];
  const K = require("./key_numbers")(N);
  const rob = T("T8_robustness_har70").find((r) => r.setting.startsWith("back+thigh")).score;
  const CI = T("T6e_coverage_ci");
  const cc = (svc, p, b) => { const r = CI.find((x) => x.service === svc && x.policy === p && x.budget_per_h === b); return `${r.covered}/${r.people}`; };
  const E = T("T4_ecg_arrhythmia").filter((r) => r.eval === "DS2 after 5 min");
  const er = (sc, m) => E.find((r) => r.scenario.startsWith(sc) && r.model === m);
  const trials = T("T1a_falls_cohort").reduce((a, r) => a + r.trials, 0);
  const rows = [
    ["Falls AUROC " + K.fallsAUROC + " (abstract, " + L.tab("bench") + ")", "CARES dual-view ensemble (GBT 300 iterations + MiniRocket-lite LR)", `trial level, impact-centred windows, all ${trials} trials, LOPO`],
    ["FoG AUROC " + f(g("T3_benchmark_fog", "CARES (dual-view)", "AUROC")) + " (abstract, " + L.tab("bench") + ")", "CARES dual-view ensemble", "all 4-s windows of every held-out person, LOPO"],
    ["FoG AUROC " + f(g("T5_federated_personalised_fog", "CARES (dual-view)", "AUROC")) + " (" + L.tab("fed") + ")", "same CARES predictions", "windows after each person's first 5 minutes"],
    ["Activity macro-F1 " + f(g("T3_benchmark_har70", "CARES (dual-view)", "MacroF1")) + " (abstract, " + L.tab("bench") + ")", "CARES dual-view ensemble", "all 5-s windows of every held-out person, LOPO"],
    ["Activity macro-F1 " + f(g("T5_federated_personalised_har70", "CARES (dual-view)", "MacroF1")) + " (" + L.tab("fed") + ")", "same CARES predictions", "windows after each person's first 5 minutes"],
    ["Activity macro-F1 " + f(rob) + " (" + L.tab("robust") + ")", "descriptor branch only, GBT with 80 iterations", "all windows; robustness analysis only"],
    ["MLP: activity " + f(g("T3_benchmark_har70", "MLP", "MacroF1")) + " (" + L.tab("bench") + ") vs. " + f(g("T5_federated_personalised_har70", "Central MLP", "MacroF1")) + " (" + L.tab("fed") + ")", L.tab("bench") + ": 12 epochs, seed 2026; " + L.tab("fed") + ": 13 epochs, mean of seeds 11, 23, 47", L.tab("bench") + ": all windows; " + L.tab("fed") + ": after the first 5 minutes"],
    [`ECG macro-F1 ${f(er("A", "CARES (dual-view)").MacroF1_NSV)} / ${K.ecgBestGlobal} / ${K.ecgAdaptive} (` + L.tab("ecg") + ")", `global CARES / global ${K.ecgBestGlobalModel} (best global) / patient-adaptive CARES`, "DS2 beats after the first 5 minutes; N, S, V classes"],
    [`Fall-alarm coverage ${cc("Falls", "conformal (proposed)", 0)} (` + L.tab("alarm") + ")", "streaming CARES (waist + wrist IMU), conformal threshold, B = 0/h", "ADL trials after the 5-min onboarding trials, young volunteers"],
    [`FoG coverage ${cc("FoG", "conformal (proposed)", 4)} (` + L.tab("alarm") + ")", "CARES window scores, smoothing k = 3, conformal threshold, B = 4/h", "complete protocol of every held-out person"],
  ];
  return [L.H1(hl("Appendix B. Configuration Behind Each Headline Number")),
    ...Tab(hl("Configuration and evaluation set of numbers that appear in more than one place."), ["Number (where)", "Model / configuration", "Evaluation set"], rows, [30, 38, 32], { hl: true, key: "key" })];
}
