// Section 4 (continued): federated learning, alarm budget, cognition, robustness/explanations, edge/IoT.
const L = require("./lib");
const { P, H2, Fig, Tab, disp, hl, fig, tab } = L;

module.exports = function (N, F) {
  const U = require("./results_util")(N);
  const { T, f, pct, pv } = U;
  const c = N.cohort;
  const out = [];

  // ------------------------------------------------------------------ 4.4 federated + personalisation
  out.push(H2("4.4. Federated Training and On-Device Personalisation"));
  const fl = { fog: T("T5_federated_personalised_fog"), har70: T("T5_federated_personalised_har70"), falls: T("T5_federated_personalised_falls") };
  const g = (s, m) => fl[s].find((r) => r.method === m);
  const key = (s) => (s === "har70" ? "MacroF1" : "AUROC");
  const methods = ["Central MLP", "FedAvg MLP", "FedProx MLP", "FedAvg MLP (50% gateways online)", "Local only (1 min)", "Local only (5 min)",
    "FedAvg + FT (1 min)", "FedAvg + FT (2 min)", "FedAvg + FT (5 min)", "CARES (dual-view)", "CARES + PC (1 min)", "CARES + PC (5 min)"];
  const lab = (m) => m.replace("FedAvg + FT", "FedAvg + on-device fine-tuning").replace("CARES + PC", "CARES + personal recalibration")
    .replace("(dual-view)", "(global, no onboarding)").replace("Local only", "Local-only MLP");
  const rows = methods.map((m) => [lab(m),
    g("fog", m) ? `${f(g("fog", m).AUROC)} / ${f(g("fog", m).subject_median)}` : "–",
    g("har70", m) ? `${f(g("har70", m).MacroF1)} / ${f(g("har70", m).subject_median)}` : "–",
    g("falls", m) ? `${f(g("falls", m).AUROC, 4)}` : "–"]);
  out.push(...Tab("Federated training and personalisation for a new person who joins after training (leave-one-person-out; all methods evaluated on the same windows after the 5-minute onboarding period; pooled / median over people; neural models averaged over three seeds). {+Because the first 5 minutes of every person are excluded, CARES values differ slightly from those in " + tab("bench") + " (computed on all windows); the centralised MLP was trained for 13 epochs and averaged over three seeds (" + tab("bench") + ": 12 epochs, one seed).+}",
    ["Method", "FoG AUROC (pooled / median)", "Activity macro-F1 (pooled / median)", "Falls AUROC (pooled)"], rows, [40, 20, 22, 18], { key: "fed" }));
  const fa = g("fog", "FedAvg MLP"), fc = g("fog", "Central MLP"), fp = g("fog", "FedProx MLP"), fh = g("fog", "FedAvg MLP (50% gateways online)");
  const ha = g("har70", "FedAvg MLP"), hc = g("har70", "Central MLP");
  const ft5 = g("fog", "FedAvg + FT (5 min)"), ft1 = g("fog", "FedAvg + FT (1 min)"), lo5 = g("fog", "Local only (5 min)");
  const hft5 = g("har70", "FedAvg + FT (5 min)"), hlo5 = g("har70", "Local only (5 min)");
  const pc5 = g("fog", "CARES + PC (5 min)"), c0 = g("fog", "CARES (dual-view)");
  out.push(P(`Federated averaging over the gateways of the training people came close to centralised training without pooling any data: FoG AUROC ${f(fa.AUROC)} vs. ${f(fc.AUROC)}, activity macro-F1 ${f(ha.MacroF1)} vs. ${f(hc.MacroF1)}, and fall AUROC ${f(g("falls", "FedAvg MLP").AUROC, 4)} vs. ${f(g("falls", "Central MLP").AUROC, 4)} (${tab("fed")}). FedProx behaved like FedAvg (FoG ${f(fp.AUROC)}). When only half of the gateways were online in each round, FoG AUROC was ${f(fh.AUROC)} and activity macro-F1 ${f(g("har70", "FedAvg MLP (50% gateways online)").MacroF1)}. ${convergence(T)}`));
  const hft1 = g("har70", "FedAvg + FT (1 min)"), hlo1 = g("har70", "Local only (1 min)"), lo1 = g("fog", "Local only (1 min)");
  out.push(P(`On-device fine-tuning on the first minutes of a new person did not help the wearable gait and activity services. For activity monitoring, fine-tuning all layers on 1 and 5 minutes hardly changed accuracy (${f(ha.Acc)} without onboarding; ${f(hft1.Acc)} and ${f(hft5.Acc)} after 1 and 5 minutes) but lowered macro-F1 from ${f(ha.MacroF1)} to ${f(hft1.MacroF1)} and ${f(hft5.MacroF1)} (Holm-adjusted p ${pv(hft5.p_holm) === "<0.001" ? "< 0.001" : "= " + pv(hft5.p_holm)}). The onboarding minutes of a typical routine rarely contain stairs or shuffling, and the fine-tuned model partly forgot these rare classes. For FoG, fine-tuning on 1 and 5 minutes gave pooled AUROCs of ${f(ft1.AUROC)} and ${f(ft5.AUROC)}, compared with ${f(fa.AUROC)} without onboarding, because short onboarding sessions contain few freezes. Personal Platt recalibration of CARES leaves each person's ranking, and hence the person-level AUROC, unchanged (median ${f(c0.subject_median)}), and lowered the pooled AUROC only slightly (${f(c0.AUROC)} → ${f(pc5.AUROC)}). By contrast, models trained locally on the same minutes were far weaker than every collaborative model: activity macro-F1 ${f(hlo1.MacroF1)} (1 min) and ${f(hlo5.MacroF1)} (5 min), and FoG AUROC ${f(lo1.AUROC)} and ${f(lo5.AUROC)}. In short, federated collaboration was essential{+, whereas personalisation of the classifier helped only when the onboarding data contained the target events, as for ECG (Section 4.3), where the 5 labelled minutes contain the person's own ectopic beats and the patient-adaptive models were retrained with these beats up-weighted (a different procedure from fine-tuning the MLP)+}.`));
  // revision: remedies for forgetting
  const RV = { fog: T("T5b_personalisation_revision_fog"), har70: T("T5b_personalisation_revision_har70") };
  const rv = (s_, m) => (m.endsWith("@g") ? RV[s_].find((r) => r.method === m.slice(0, -2) && r.eval_set.startsWith("outside"))
    : RV[s_].find((r) => r.method === m && r.eval_set.startsWith("after")));
  const kk = (s_) => (s_ === "har70" ? "MacroF1" : "AUROC");
  const cell = (s_, m) => { const r = rv(s_, m); return r ? `${f(r[kk(s_)])} / ${f(r.subject_median)}` : "–"; };
  const fRows = [
    ["FedAvg MLP, no onboarding", cell("har70", "FedAvg MLP"), "", cell("fog", "FedAvg MLP"), ""],
    ["Fine-tuning, all layers", cell("har70", "FT all layers (1 min)"), cell("har70", "FT all layers (5 min)"), cell("fog", "FT all layers (1 min)"), cell("fog", "FT all layers (5 min)")],
    ["Fine-tuning, output layer only", cell("har70", "FT last layer (1 min)"), cell("har70", "FT last layer (5 min)"), cell("fog", "FT last layer (1 min)"), cell("fog", "FT last layer (5 min)")],
    ["All layers + class-balanced replay (federated statistics)", cell("har70", "FT all + statistics replay (1 min)"), cell("har70", "FT all + statistics replay (5 min)"), cell("fog", "FT all + statistics replay (1 min)"), cell("fog", "FT all + statistics replay (5 min)")],
    ["Output layer + class-balanced replay (federated statistics)", cell("har70", "FT last + statistics replay (1 min)"), cell("har70", "FT last + statistics replay (5 min)"), cell("fog", "FT last + statistics replay (1 min)"), cell("fog", "FT last + statistics replay (5 min)")],
    ["All layers + replay of real training windows (privacy-costly reference)", cell("har70", "FT all + real-data replay (1 min)"), cell("har70", "FT all + real-data replay (5 min)"), cell("fog", "FT all + real-data replay (1 min)"), cell("fog", "FT all + real-data replay (5 min)")],
    ["**Evaluated outside every onboarding session:**", "", "", "", ""],
    ["FedAvg MLP, no onboarding", cell("har70", "FedAvg MLP@g"), "", cell("fog", "FedAvg MLP@g"), ""],
    ["All layers, routine onboarding (first minutes)", cell("har70", "FT all layers, routine onboarding (1 min)@g"), cell("har70", "FT all layers, routine onboarding (5 min)@g"), cell("fog", "FT all layers, routine onboarding (1 min)@g"), cell("fog", "FT all layers, routine onboarding (5 min)@g")],
    ["All layers, guided onboarding (activity) / onboarding with a freeze (FoG)", cell("har70", "FT all layers, guided onboarding (1 min)@g"), cell("har70", "FT all layers, guided onboarding (5 min)@g"), cell("fog", "FT all layers, onboarding with a freeze (1 min)@g"), cell("fog", "FT all layers, onboarding with a freeze (5 min)@g")],
    ["Output layer, guided onboarding (activity) / onboarding with a freeze (FoG)", cell("har70", "FT last layer, guided onboarding (1 min)@g"), cell("har70", "FT last layer, guided onboarding (5 min)@g"), cell("fog", "FT last layer, onboarding with a freeze (1 min)@g"), cell("fog", "FT last layer, onboarding with a freeze (5 min)@g")],
  ];
  out.push(P(hl(forgetText(N, tab))));
  out.push(...Tab(hl("Can on-device personalisation avoid forgetting? Federated MLP of a new person after 1 or 5 minutes of labelled onboarding (leave-one-person-out; three seeds; pooled / median over people). Upper block: the same evaluation windows as " + tab("fed") + ". Lower block: guided activity sessions that cover every activity the person performs, or FoG sessions that start 30 s before the person's first freeze; these are evaluated on the windows outside every onboarding session, so their reference row differs; for FoG, this block includes the " + RV.fog.find((r) => r.method === "FedAvg MLP" && r.eval_set.startsWith("outside")).people + " people with at least one freeze."),
    ["Personalisation", "Activity macro-F1, 1 min", "Activity macro-F1, 5 min", "FoG AUROC, 1 min", "FoG AUROC, 5 min"], fRows, [40, 15, 15, 15, 15], { hl: true, key: "forget" }));
  out.push(...Fig(F.path("Fig9_federated_personalisation.png"), "Federated training and on-device personalisation for a new person. Lines: performance after 0–5 minutes of labelled onboarding; horizontal references: centralised, FedProx and partially connected (50%) federated training.", 6.5, "fed"));

  // ------------------------------------------------------------------ 4.5 alarms
  out.push(H2("4.5. False-Alarm Budget: Guarantee and Price"));
  const AF = T("T6_alarm_budget_fog"), AL = T("T6_alarm_budget_falls");
  const CI = T("T6e_coverage_ci"), DI = T("T6f_false_alarm_distribution"), SP = T("T6b_falls_split_conformal");
  const a = (t, p, b) => t.find((r) => r.policy === p && r.budget_per_h === b);
  const cic = (svc, p, b) => CI.find((r) => r.service === svc && r.policy === p && r.budget_per_h === b);
  const dic = (svc, p, b) => DI.find((r) => r.service === svc && r.policy === p && r.budget_per_h === b);
  const covTxt = (r) => `${r.covered}/${r.people} (${pct(r.coverage)}; ${f(100 * r.coverage_lo, 1)}–${f(100 * r.coverage_hi, 1)})`;
  const rateTxt = (d) => `${f(d.pooled_fa_per_h, 2)} (${f(d.pooled_lo_poisson, 2)}–${f(d.pooled_hi_poisson, 2)})`;
  const rowsA = [];
  const fogRows = [["fixed 0.5", 4], ["Youden (window-optimal)", 4], ["naive pooled calibration", 4], ["naive pooled calibration", 16], ["conformal (proposed)", 4], ["conformal (proposed)", 16]];
  fogRows.forEach(([p, b]) => {
    const r = a(AF, p, b), q = cic("FoG", p, b), d = dic("FoG", p, b);
    const lab = p.startsWith("fixed") || p.startsWith("Youden") ? p : `${p}, B = ${b}/h`;
    rowsA.push(["FoG", lab, hl(covTxt(q)), hl(`${d.people_with_fa}/${d.people}`), hl(rateTxt(d)), pct(r.episode_sensitivity, 1), f(r.median_latency_s, 1)]);
  });
  const fallRows = [["fixed 0.5", 0], ["Youden (window-optimal)", 0], ["conformal (proposed)", 0], ["conformal (proposed)", 2]];
  fallRows.forEach(([p, b]) => {
    const r = a(AL, p, b), q = cic("Falls", p, b), d = dic("Falls", p, b);
    const lab = p.startsWith("fixed") || p.startsWith("Youden") ? p : `${p}, B = ${b}/h`;
    rowsA.push(["Falls", lab, hl(covTxt(q)), hl(`${d.people_with_fa}/${d.people}`), hl(rateTxt(d)), pct(r.fall_sensitivity, 1), f(r.median_latency_s, 1)]);
  });
  [0, 2].forEach((b) => {
    const r = SP.find((x) => x.budget_per_h === b), q = cic("Falls (split conformal)", "split conformal (exact exchangeability)", b), d = dic("Falls (split conformal)", "split conformal (exact exchangeability)", b);
    rowsA.push([hl("Falls"), hl(`split conformal (A2 exact), B = ${b}/h`), hl(covTxt(q)), hl(`${d.people_with_fa}/${d.people}`), hl(rateTxt(d)), hl(pct(r.fall_sensitivity, 1)), hl("–")]);
  });
  const fogH = dic("FoG", "conformal (proposed)", 4), falH = dic("Falls", "conformal (proposed)", 0);
  out.push(P(hl(`${tab("alarm")} and ${fig("alarm")} compare the threshold policies on continuous streams; ${fig("streams")} shows examples.`)));
  out.push(...Tab(`Alarm management on continuous streams (leave-one-person-out). Coverage = share of held-out people whose false-alarm rate stayed within the budget B (target ≥ ${pct(1 - 0.1, 1)}){+, with exact Clopper–Pearson 95% CI; people with ≥ 1 FA = held-out people with at least one false alarm; pooled FA/h = all false alarms divided by all monitored hours, with exact Poisson 95% CI. Monitored hours per person: FoG ${f(fogH.hours_min, 2)}–${f(fogH.hours_max, 2)} h (median ${f(fogH.hours_median, 2)} h) of protocol; falls ${f(falH.hours_min, 2)}–${f(falH.hours_max, 2)} h (median ${f(falH.hours_median, 2)} h) of ADL after the onboarding trials. Split conformal: detector trained on 6 and calibrated on 10 of the 16 training people, so that (A2) holds exactly; evaluated on all ADL trials of the held-out person.+} Sensitivity = freezing episodes or falls detected; latency = median delay from event onset (FoG) or impact (falls).`,
    ["Service", "Threshold policy", hl("Coverage (95% CI)"), hl("People with ≥ 1 FA"), hl("Pooled FA/h (95% CI)"), "Sensitivity", "Latency (s)"], rowsA, [7, 25, 21, 9, 16, 10, 8], { key: "alarm" }));
  const fy = a(AF, "Youden (window-optimal)", 4), fn4 = a(AF, "naive pooled calibration", 4), fc4 = a(AF, "conformal (proposed)", 4);
  const fc16 = a(AF, "conformal (proposed)", 16), fn16 = a(AF, "naive pooled calibration", 16), f05 = a(AF, "fixed 0.5", 4);
  const qn4 = cic("FoG", "naive pooled calibration", 4), qc4 = cic("FoG", "conformal (proposed)", 4), qy4 = cic("FoG", "Youden (window-optimal)", 4);
  const minNaive = Math.min(...AF.filter((r) => r.policy === "naive pooled calibration").map((r) => r.coverage));
  const minConf = Math.min(...AF.filter((r) => r.policy === "conformal (proposed)").map((r) => r.coverage));
  const fyD = dic("FoG", "Youden (window-optimal)", 4);
  out.push(P(`**Freezing of gait.** Thresholds chosen for window accuracy were unusable as alert thresholds. The Youden threshold produced a median of ${f(fy.median_fa_per_h, 2)} false cues per hour{+ (every one of the ${fyD.people} people exceeded every budget)+}, and the default 0.5 still produced ${f(f05.median_fa_per_h, 2)}. With the subject-level conformal rule, the share of new people within budget was ${pct(minConf, 1)} {+(9 of 10 people)+} or more at every budget from 2 to 32 false alarms per hour{+, consistent with the 90% target. With 10 people, however, each person changes the estimate by 10 percentage points: the exact 95% CI of 9/10 is ${pct(qc4.coverage_lo)}–${pct(qc4.coverage_hi)}, and the difference from the naive pooled calibration+} (${pct(qn4.coverage, 1)} at B = 4/h and ${pct(fn16.coverage, 1)} at B = 16/h; ${pct(minNaive, 1)} at worst) {+was not statistically significant (exact McNemar p = ${pv(qn4.mcnemar_p_vs_conformal)} at B = 4/h), whereas the difference from the Youden threshold was (p = ${pv(qy4.mcnemar_p_vs_conformal)}). The FoG data therefore show that the conformal rule behaves as intended, not that it is better than a pooled threshold. At B = 4/h, ${fogH.people - fogH.people_with_fa} of 10 people had no false cue at all, and the ${fogH.total_fa} false cues of the other ${fogH.people_with_fa} people gave a pooled rate of ${f(fogH.pooled_fa_per_h, 2)} per hour (95% CI ${f(fogH.pooled_lo_poisson, 2)}–${f(fogH.pooled_hi_poisson, 2)}).+} The guarantee had a clear price. At B = 4/h the conformal threshold detected ${pct(fc4.episode_sensitivity, 1)} of freezing episodes (naive: ${pct(fn4.episode_sensitivity, 1)}), and at B = 16/h it detected ${pct(fc16.episode_sensitivity, 1)}. The median cue latency was ${f(fc16.median_latency_s, 1)} s after freeze onset.`));
  const l0 = a(AL, "conformal (proposed)", 0), ly = a(AL, "Youden (window-optimal)", 0), l05 = a(AL, "fixed 0.5", 0), l2 = a(AL, "conformal (proposed)", 2);
  const ql0 = cic("Falls", "conformal (proposed)", 0), qly = cic("Falls", "Youden (window-optimal)", 0), dl0 = dic("Falls", "conformal (proposed)", 0), dly = dic("Falls", "Youden (window-optimal)", 0);
  const sp0 = SP.find((x) => x.budget_per_h === 0), sp2 = SP.find((x) => x.budget_per_h === 2);
  const qs0 = cic("Falls (split conformal)", "split conformal (exact exchangeability)", 0);
  const MG = T("T6g_personal_margin_scenarios");
  const mgHalf = MG.find((r) => r.personal_margin && r.base_threshold.startsWith("conformal, delta = 0.50"));
  const qs2 = cic("Falls (split conformal)", "split conformal (exact exchangeability)", 2);
  const spWorst = T("S18_falls_split_per_subject").filter((r) => r.budget_per_h === 0).reduce((x, y) => (y.fa > x.fa ? y : x));
  const raised = MG.filter((r) => r.personal_margin).reduce((x, r) => x + r.people_threshold_raised, 0);
  out.push(P(`**Falls on continuous streams.** Scanning every trial with a window every 0.4 s exposes the detector to the calm and transitional parts of every activity, which the impact-centred benchmark never sees. Even so, the streaming detector produced few false alarms. With the window-optimal threshold, ${f(ly.adl_trials_with_alarm_pct, 1)}% of ADL trials raised an alarm (${f(ly.pooled_fa_per_h, 2)} false alarms per ADL hour), but these were concentrated in a few people: ${pct(ly.coverage, 1)} {+(${qly.covered}/${qly.people})+} of people had no false alarm at all. The fixed 0.5 threshold behaved similarly (${f(l05.pooled_fa_per_h, 2)} per hour). The conformal threshold for a zero-false-alarm budget kept {+${ql0.covered} of ${ql0.people} young volunteers (${pct(ql0.coverage, 1)}; 95% CI ${pct(ql0.coverage_lo)}–${pct(ql0.coverage_hi)})+} completely free of false alarms over their ADL protocol {+(McNemar p = ${pv(qly.mcnemar_p_vs_conformal)} vs. the Youden threshold)+}, and reduced the pooled rate to ${f(l0.pooled_fa_per_h, 2)} per ADL hour{+, which rests on only ${dl0.total_fa} false alarms in ${dl0.people_with_fa === 1 ? "one person" : dl0.people_with_fa + " people"} (95% CI ${f(dl0.pooled_lo_poisson, 2)}–${f(dl0.pooled_hi_poisson, 2)} per hour)+}. It detected ${pct(l0.fall_sensitivity, 1)} of falls with a median latency of ${f(l0.median_latency_s, 1)} s after impact, against ${pct(ly.fall_sensitivity, 1)} for the Youden threshold. Allowing two or six false alarms per ADL hour raised sensitivity to ${pct(l2.fall_sensitivity, 1)} and ${pct(a(AL, "conformal (proposed)", 6).fall_sensitivity, 1)}. {+These coverage figures must be read with the exposure in mind: each volunteer contributed only ${f(dl0.hours_min, 2)}–${f(dl0.hours_max, 2)} h of ADL (median ${f(dl0.hours_median, 2)} h, i.e. about ${Math.round(dl0.hours_median * 60)} min), and zero false alarms in ${Math.round(dl0.hours_median * 60)} min bound a person's rate only below ${f(dl0.upper95_rate_if_zero_median_person, 1)} per hour (one-sided 95%). A budget of 0 per hour therefore means 'no false alarm during the recorded ADL protocol', not 'no false alarm during a day'. When the detector and the calibration people were separated so that (A2) holds exactly (split conformal: detector trained on 6 people, 10 calibration people, evaluated on all ADL trials of the held-out person), coverage was ${qs0.covered}/${qs0.people} (${pct(qs0.coverage, 1)}; 95% CI ${pct(qs0.coverage_lo)}–${pct(qs0.coverage_hi)}) at B = 0 with a fall sensitivity of ${pct(sp0.fall_sensitivity, 1)}, and ${qs2.covered}/${qs2.people} (${pct(qs2.coverage, 1)}) at B = 2 per hour with ${pct(sp2.fall_sensitivity, 1)}. Exact exchangeability therefore did not cost sensitivity, but it made the threshold depend on which 10 people were drawn for calibration: in one fold, these people gave a threshold of only ${f(spWorst.threshold, 2)}, and the held-out person raised ${spWorst.fa} of the ${sp0.total_fa} false alarms at B = 0. The optional personal ADL margin did not raise the threshold of any volunteer on top of the conformal, fixed or Youden thresholds, and on top of a less conservative conformal threshold (δ = 0.5) it raised ${mgHalf.people_threshold_raised === 1 ? "one threshold" : mgHalf.people_threshold_raised + " thresholds"} without changing any false alarm, because almost no volunteer's first 5 minutes of activities scored above these thresholds (Table S23); it is therefore no longer presented as a contribution.+}${fallsActivitySentence(T("S9_falls_by_activity"), f, pct)}`));
  // revision: older adults (HAR70+) as an ADL-only source
  out.push(...olderAdults(T, f, pct, pv, tab));
  out.push(...Fig(F.path("Fig10_alarm_budget.png"), "False-alarm budget. Left of each pair: share of new people whose false-alarm rate stayed within the budget {+(whiskers: exact 95% CI; dashed line: 1 − δ = 90% target)+}. Right: sensitivity for freezing episodes or falls at that budget. {+Only the conformal rule reached the target at every budget, but the wide CIs show that 10 or 17 people cannot separate it from the alternatives with certainty.+}", 6.5, "alarm"));
  out.push(...Fig(F.path("Fig11_example_streams.png"), "Examples. Top: smoothed CARES freezing probability of a held-out person with frequent freezing (shaded: annotated freezes), with the conformal thresholds for budgets of 4 and 16 false alarms per hour and the resulting cues. Bottom: streaming fall probability around the acceleration peak for a fall (syncope) and two fall-like ADLs of another held-out person.", 6.5, "streams"));

  // ------------------------------------------------------------------ 4.6 cognition
  out.push(H2("4.6. Camera-Free Cognitive-Health Surveillance"));
  const CG = T("T7_cognitive_surveillance");
  const cr = (task, m) => CG.find((r) => r.task.startsWith(task) && r.model === m);
  const cogRows = [];
  ["A", "B", "C", "D"].forEach((t) => ["LR", "RF", "HGB", "CARES-Cog (ensemble)"].forEach((m) => {
    const r = cr(t, m);
    if (r) cogRows.push([r.task.replace(/^[A-D]: /, ""), disp(m === "HGB" ? "GBT" : m), `${r.n} (${r.n_impaired})`, `${f(r.AUROC)} (${f(r.AUROC_lo)}–${f(r.AUROC_hi)})`, f(r.AUPRC), f(r.Sens), f(r.Spec)]);
  }));
  out.push(...Tab("Cognitive-health surveillance from ambient smart-home sensors (ten repeats of stratified 5-fold cross-validation; AUROC mean and 2.5–97.5th percentile over repeats; sensitivity/specificity at the Youden point of the averaged cross-validated scores, descriptive).",
    ["Contrast", "Model", "n (cases)", "AUROC", "AUPRC", "Sens", "Spec"], cogRows, [28, 18, 10, 18, 9, 8, 9], { key: "cog" }));
  const best = (t) => CG.filter((r) => r.task.startsWith(t)).reduce((x, y) => (y.AUROC > x.AUROC ? y : x));
  const bA = best("A"), bB = best("B"), bC = best("C"), bD = best("D");
  const ord = T("S10b_cognitive_risk_ordering")[0];
  const tq = T("T7b_task_quality_regression")[0];
  const imp = T("S11_cognitive_importance_by_task");
  const tname = { t01: "sweeping and dusting (T1)", t02: "filling a medication dispenser (T2)", t03: "writing a birthday card (T3)", t04: "finding a DVD and watching a news clip (T4)", t05: "watering plants (T5)", t06: "answering questions by phone (T6)", t07: "preparing soup (T7)", t08: "choosing an outfit (T8)", t09: "replacing a light bulb (T9)", t10: "washing hands (T10)", t11: "cleaning countertops (T11)", t12: "noting a recipe from a phone call (T12)", t13: "sorting laundry (T13)", t14: "cooking oatmeal (T14)", t15: "filing bills (T15)", t16: "setting up a card game (T16)", t17: "planning a bus trip (T17)", t18: "microwaving a heat pack (T18)", t19: "selecting a magazine (T19)", t20: "counting bus fare (T20)", t21: "taking medication (T21)", t22: "gathering picnic ingredients (T22)", t23: "packing a picnic basket (T23)", t24: "leaving the apartment (T24)", uncued: "the uncued task block as a whole", cued: "the cued task block as a whole", dayout: "the interwoven day-out block as a whole", n_tasks: "the number of tasks completed" };
  const tk = (r) => r.task || r[Object.keys(r)[0]];
  const tl = imp.slice(0, 3).map((r) => tname[tk(r)] || tk(r));
  const topTasks = tl.slice(0, -1).join(", ") + " and " + tl[tl.length - 1];
  out.push(P(`Using only motion, door and item-use events recorded while participants carried out everyday tasks, the best model distinguished people living with cognitive impairment from cognitively healthy adults with AUROC ${f(bA.AUROC)} (${disp(bA.model === "HGB" ? "GBT" : bA.model)}; ${tab("cog")}). Restricting the healthy group to people aged 60+ reduced this to ${f(bB.AUROC)}, so part of the signal reflects age. Dementia was clearly more detectable (AUROC ${f(bC.AUROC)}) than MCI (${f(bD.AUROC)}). The cross-validated risk increased with clinical severity (Kruskal–Wallis p ${ord.kruskal_p < 0.001 ? "< 0.001" : "= " + f(ord.kruskal_p)}; ${fig("cog")}). The tasks that carried most information were ${topTasks}. The same sensor features also predicted the experimenters' error score of the uncued tasks (Spearman ρ = ${f(tq.spearman_rho)}, 95% range ${f(tq.rho_lo)}–${f(tq.rho_hi)}). This service is therefore suited to flagging people for clinical follow-up rather than to diagnosis.`));
  out.push(P(hl(mciText(CG, f, c))));
  out.push(...Fig(F.path("Fig12_cognitive.png"), "Cognitive-health surveillance. Left: cross-validated ROC curve of the ensemble (impaired vs. healthy 45+). Middle: cross-validated risk by clinical group. Right: everyday tasks with the largest permutation importance.", 6.5, "cog"));

  // ------------------------------------------------------------------ 4.7 robustness and explanations
  out.push(H2("4.7. Robustness, Sensor Placement and Explanations"));
  const R = { falls: T("T8_robustness_falls"), fog: T("T8_robustness_fog"), har70: T("T8_robustness_har70") };
  const rr = (s, an, st) => R[s].find((r) => r.analysis.startsWith(an) && r.setting === st);
  out.push(P(hl(`${tab("robust")} and ${fig("robust")} report sensor placement, run-time sensor dropout, sampling rate and noise; ${fig("expl")} reports the alert explanations.`)));
  const robRows = [];
  ["falls", "fog", "har70"].forEach((s) => R[s].forEach((r) => robRows.push([{ falls: "Falls", fog: "FoG", har70: "Activity 70+" }[s], r.analysis.replace(/^\([a-d]\) /, ""), r.setting, f(r.score, s === "falls" ? 4 : 3)])));
  out.push(...Tab("Sensor placement, run-time sensor dropout, sampling rate and test-time noise (leave-one-person-out, descriptor branch with lighter boosting: 150 iterations for binary and 80 for multi-class tasks; AUROC for falls and FoG, macro-F1 for activity). {+Because only the descriptor branch with fewer boosting iterations is used, scores are lower than those of the full CARES ensemble in " + tab("bench") + " (e.g., activity: " + f(T("T8_robustness_har70").find((r) => r.setting.startsWith("back+thigh")).score) + " here vs. " + f(T("T3_benchmark_har70").find((r) => r.model === "CARES (dual-view)").MacroF1) + "); the analysis compares settings with each other, not with " + tab("bench") + ".+}",
    ["Service", "Analysis", "Setting", "Score"], robRows, [14, 26, 44, 16], { key: "robust" }));
  const fallsPlace = R.falls.filter((r) => r.analysis.startsWith("(a)"));
  const bestSite = fallsPlace.filter((r) => r.setting.includes("(acc+gyr)")).reduce((x, y) => (y.score > x.score ? y : x));
  const worstSite = fallsPlace.filter((r) => r.setting.includes("(acc+gyr)")).reduce((x, y) => (y.score < x.score ? y : x));
  const fogDrop = R.fog.filter((r) => r.analysis.startsWith("(b)"));
  const miss = fogDrop.filter((r) => r.setting.includes("missing")), bank = fogDrop.filter((r) => r.setting.includes("bank"));
  const fogRate = R.fog.filter((r) => r.analysis.startsWith("(c)"));
  const fogNoise = R.fog.filter((r) => r.analysis.startsWith("(d)"));
  out.push(P(`At trial level, every single body site detected falls almost perfectly (AUROC ${f(worstSite.score, 4)}–${f(bestSite.score, 4)}). The ${bestSite.setting.split(" ")[0]} was marginally the best site and the ${worstSite.setting.split(" ")[0]} the worst, in line with earlier placement studies [@ozdemir2016]; the waist + wrist kit reached ${f(rr("falls", "(a)", "waist+wrist (CARES kit)").score, 4)}. For FoG (descriptor branch), the ankle sensor alone reached ${f(rr("fog", "(a)", "ankle").score)} and the thigh sensor alone only ${f(rr("fog", "(a)", "thigh").score)}; the best combination was ankle + trunk (${f(rr("fog", "(a)", "ankle+trunk").score)}), slightly above all three sensors (${f(rr("fog", "(a)", "all three (CARES kit)").score)}), so a two-sensor kit would suffice. When a worn sensor dropped out at run time, feeding the unchanged model with missing inputs gave FoG AUROCs between ${f(Math.min(...miss.map((r) => r.score)))} and ${f(Math.max(...miss.map((r) => r.score)))}, whereas switching to a model of the CARES model bank trained without that sensor gave ${f(Math.min(...bank.map((r) => r.score)))}–${f(Math.max(...bank.map((r) => r.score)))}. For falls, losing the waist unit cost more with the unchanged model (${f(rr("falls", "(b)", "waist lost: same model, missing inputs").score, 4)}) than with the model bank (${f(rr("falls", "(b)", "waist lost: CARES model bank").score, 4)}). Halving the FoG sampling rate to 32 Hz gave an AUROC of ${f(fogRate.find((r) => r.setting.startsWith("32")).score)} and 16 Hz gave ${f(fogRate.find((r) => r.setting.startsWith("16")).score)} (native ${f(fogRate.find((r) => r.setting.includes("native")).score)}). Falls at 12.5 Hz (${f(rr("falls", "(c)", "12.5 Hz (from 25 Hz)").score, 4)}) were as good as at 25 Hz, which leaves room for battery savings. Test-time Gaussian noise with a standard deviation {+(SD)+} of 10–50% of the signal SD reduced FoG AUROC to ${f(Math.min(...fogNoise.map((r) => r.score)))}–${f(Math.max(...fogNoise.map((r) => r.score)))} and fall AUROC to ${f(Math.min(...R.falls.filter((r) => r.analysis.startsWith("(d)")).map((r) => r.score)), 4)}–${f(Math.max(...R.falls.filter((r) => r.analysis.startsWith("(d)")).map((r) => r.score)), 4)}.${harRobust(R, f)}`));
  const EX = { falls: T("T9_alert_explanations_falls"), fog: T("T9_alert_explanations_fog") };
  const FB = { falls: T("T9b_explanation_faithfulness_falls")[0], fog: T("T9b_explanation_faithfulness_fog")[0] };
  const top = (s) => { const t = EX[s]; const tot = t.reduce((a, r) => a + r.alerts, 0); const r0 = t[0]; return `${r0.family} features of the ${r0.site} sensor (${pct(r0.alerts / tot, 1)} of alerts)`; };
  out.push(P(`Alert explanations concentrated on a few interpretable sources. The most frequent dominant evidence was ${top("falls")} for falls and ${top("fog")} for FoG (${fig("expl")}). The explanations were faithful: removing the top-attributed group reduced the alert probability by ${f(FB.falls.mean_drop_top)} on average for falls and ${f(FB.fog.mean_drop_top)} for FoG, compared with ${f(FB.falls.mean_drop_random)} and ${f(FB.fog.mean_drop_random)} for a randomly chosen group (Wilcoxon p ${FB.falls.wilcoxon_p < 0.001 ? "< 0.001" : "= " + f(FB.falls.wilcoxon_p)} and ${FB.fog.wilcoxon_p < 0.001 ? "< 0.001" : "= " + f(FB.fog.wilcoxon_p)}).`));
  out.push(...Fig(F.path("Fig13_robustness.png"), "Robustness: sensor placement, run-time sensor dropout (unchanged model with missing inputs vs. the CARES model bank), battery-saving sampling rates and test-time noise.", 6.5, "robust"));
  out.push(...Fig(F.path("Fig14_explanations.png"), "Alert explanations. Heat maps: share of alerts whose dominant evidence came from each body site and feature family. Right: deletion faithfulness (drop in alert probability when the top-attributed group or a random group is removed).", 6.5, "expl"));

  // ------------------------------------------------------------------ 4.8 edge / IoT
  out.push(H2("4.8. Edge Cost, Data Minimisation and Alert Delivery"));
  const LT = T("T10_gateway_latency"), BW = T("T11_bandwidth"), SIM = T("T12_alert_delivery_simulation");
  const lt = (s) => LT.find((r) => r.service === s), bw = (s) => BW.find((r) => r.service === s);
  const edgeRows = ["fog", "har70", "falls"].map((s) => [{ fog: "FoG", har70: "Activity 70+", falls: "Falls" }[s], f(lt(s).median_ms_per_window, 1), f(lt(s).p95_ms, 1),
    f(lt(s).duty_cycle_pct, 1) + "%", hl(f(lt(s).model_kB / 1024, 2)), hl(f(bw(s).raw_MB_per_day, 2)), hl(f(bw(s).features_MB_per_day, 2)), f(bw(s).cares_MB_per_day, 2), `${f(bw(s).reduction_vs_raw, 0)}×`]);
  edgeRows.push(["ECG patch", "–", "–", "–", "–", hl(f(bw("ecg").raw_MB_per_day, 2)), hl(f(bw("ecg").features_MB_per_day, 2)), f(bw("ecg").cares_MB_per_day, 2), `${f(bw("ecg").reduction_vs_raw, 0)}×`]);
  out.push(...Tab("Gateway cost (complete CARES chain for one window on one CPU core: descriptors, MiniRocket-lite, both classifiers) and data leaving the home per person-day.",
    ["Service", "Median ms", "p95 ms", "CPU duty cycle", "Model (MB)", "Raw MB/day", "Features MB/day", "IoT-CARES MB/day", "Reduction"], edgeRows, [12, 9, 9, 10, 9, 10, 12, 13, 10], { key: "edge" }));
  const pa = (x) => (x >= 0.999999 ? "all" : pct(x, 1));
  const s5 = SIM.find((r) => r.scenario === "packet loss 5%"), s0 = SIM.find((r) => r.scenario === "packet loss 0%"), s20 = SIM.find((r) => r.scenario === "packet loss 20%");
  const so = SIM.find((r) => r.scenario.startsWith("cloud outage"));
  out.push(P(`The complete CARES chain needed a median of ${f(lt("fog").median_ms_per_window, 1)} ms per FoG window, ${f(lt("har70").median_ms_per_window, 1)} ms per activity window and ${f(lt("falls").median_ms_per_window, 1)} ms per fall window on one CPU core. This corresponds to a CPU duty cycle of ${f(lt("falls").duty_cycle_pct, 1)}% even for the fall service, which scores a window every 0.4 s (${tab("edge")}). Because only alerts, hourly summaries and one model update per day are transmitted, the data leaving the home fell from ${f(bw("fog").raw_MB_per_day, 2)} MB to ${f(bw("fog").cares_MB_per_day, 2)} MB per person-day for FoG (${f(bw("fog").reduction_vs_raw, 0)}-fold) and from ${f(bw("ecg").raw_MB_per_day, 2)} MB to ${f(bw("ecg").cares_MB_per_day, 2)} MB for the ECG patch. Streaming descriptors instead would save little or even increase traffic (FoG ${f(bw("fog").features_MB_per_day, 2)} MB and falls ${f(bw("falls").features_MB_per_day, 2)} MB per day), because the descriptors of overlapping windows can be larger than the signal itself. In the delivery simulation (${tab("delivery")}, ${fig("edge")}), the median time from fall impact to caregiver notification was ${f(s0.median_s, 1)} s without packet loss and ${f(s5.median_s, 1)} s at 5% loss. At 5% loss, ${pct(s5.within_10s, 1)} of alerts arrived within 10 s and ${pa(s5.within_60s)} within 60 s; at 20% loss, ${pct(s20.within_10s, 1)} arrived within 10 s, ${pct(s20.within_30s, 1)} within 30 s and ${pa(s20.within_60s)} within 60 s. During a cloud outage of up to 60 s, ${pct(so.within_60s, 1)} of alerts still arrived within 60 s, and the gateway's local siren works regardless of connectivity.`));
  const simRows = SIM.map((r) => [r.scenario, f(r.median_s, 1), f(r.p95_s, 1), f(r.p99_s, 1), pct(r.within_10s, 1), pct(r.within_30s, 1), pct(r.within_60s, 1)]);
  out.push(...Tab("Monte-Carlo simulation of fall-alert delivery (20,000 alerts per scenario; measured detection latency plus assumed network delays, see Table S12).",
    ["Scenario", "Median (s)", "p95 (s)", "p99 (s)", "≤ 10 s", "≤ 30 s", "≤ 60 s"], simRows, [32, 11, 11, 11, 11, 12, 12], { key: "delivery" }));
  out.push(...Fig(F.path("Fig15_edge_iot.png"), "Left: data leaving the home per person-day (log scale) for raw streaming, descriptor streaming and IoT-CARES. Right: cumulative distribution of the simulated time from fall impact to caregiver notification under packet loss and a cloud outage.", 6.5, "edge"));
  out.push(...Fig(F.path("Fig16_confusion.png"), "Row-normalised confusion matrices: activity recognition of adults aged 70+ (CARES, leave-one-person-out) and ECG beat classification on DS2 {+(beats after the first 5 minutes) for (b) global and (c) patient-adaptive CARES; n = number of beats per class. The Q class contains only 7 evaluation beats. Confusion matrices of all global ECG models are given in Figure S1+}.", 6.5, "confusion"));
  out.push(...Fig(F.path("Fig17_fedavg_curves.png"), "FedAvg learning curves for each held-out person (grey) and their median (orange), recorded for illustration only; no model selection was performed on the held-out person.", 6.3, "curves"));
  return out;
};

function harRobust(R, f) {
  const h = R.har70;
  const g = (a, st) => h.find((r) => r.analysis.startsWith(a) && r.setting.startsWith(st));
  const kit = g("(a)", "back+thigh"), back = g("(a)", "lower back"), thigh = g("(a)", "thigh");
  const r25 = g("(c)", "25 Hz"), r12 = g("(c)", "12.5 Hz");
  let s = ` For activity monitoring of older adults, macro-F1 was ${f(kit.score)} with both sensors, ${f(back.score)} with the lower-back sensor only and ${f(thigh.score)} with the thigh sensor only`;
  if (r25 && r12) s += `, and ${f(r25.score)} and ${f(r12.score)} at 25 and 12.5 Hz`;
  return s + ".";
}

function convergence(T) {
  const med = (name) => {
    const rows = T(name), by = {};
    rows.forEach((r) => { if (r.score !== null && r.score !== undefined) (by[r.round] = by[r.round] || []).push(r.score); });
    return Object.keys(by).map(Number).sort((a, b) => a - b).map((k) => { const v = by[k].sort((a, b) => a - b); const n = v.length; return [k, n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2]; });
  };
  const fog = med("S6_fedavg_convergence_fog"), har = med("S6_fedavg_convergence_har70");
  const peak = (m) => m.reduce((x, y) => (y[1] > x[1] ? y : x));
  const pf = peak(fog), ph = peak(har), lf = fog[fog.length - 1], lh = har[har.length - 1];
  const r95 = (m) => { const mx = peak(m)[1], mn = m[0][1]; const t = m.find((x) => x[1] >= mn + 0.95 * (mx - mn)); return t ? t[0] : m[m.length - 1][0]; };
  return ` Median learning curves (${fig("curves")}; recorded for illustration only, never used for model selection) reached 95% of their maximum after ${r95(fog)} rounds for FoG and ${r95(har)} rounds for activity. For FoG the median then declined slightly (${pf[1].toFixed(3)} at round ${pf[0]} vs. ${lf[1].toFixed(3)} at round ${lf[0]}), a mild sign of over-training that a validation-based stopping rule could avoid.`;
}

function fallsActivitySentence(rows, f, pct) {
  const names = { 801: "walking forward", 802: "walking backward", 803: "jogging", 804: "squatting", 805: "bending", 806: "bending to pick something up", 807: "limping", 808: "stumbling", 809: "tripping over", 810: "coughing or sneezing", 811: "sitting down on a chair", 812: "sitting down on a sofa", 813: "sitting in the air", 814: "sitting down on a bed", 815: "lying down on a bed", 816: "rising from a bed" };
  const fnames = { 901: "front-lying", 902: "front-protecting", 903: "falling onto the knees", 904: "falling onto the knees then lying", 905: "front fall with quick recovery", 906: "front fall with slow recovery", 907: "front fall to the right", 908: "front fall to the left", 909: "backward fall ending sitting", 910: "backward fall ending lying", 911: "backward fall to the right", 912: "backward fall to the left", 913: "sideways fall to the right", 914: "right sideways fall with recovery", 915: "sideways fall to the left", 916: "left sideways fall with recovery", 917: "rolling out of bed", 918: "falling from a podium", 919: "syncope", 920: "syncope sliding down a wall" };
  const adl = rows.filter((r) => !r.fall && r.trials_with_false_alarm > 0);
  const falls = rows.filter((r) => r.fall).sort((a, b) => a.detected / a.trials - b.detected / b.trials);
  let s = "";
  const nfa = adl.reduce((x, r) => x + r.trials_with_false_alarm, 0);
  if (adl.length) s += ` At the zero budget, the only ADL trials with an alarm were ${nfa} trial${nfa > 1 ? "s" : ""} of ${adl.map((r) => names[r.activity] || r.activity).join(" and ")}.`;
  if (falls.length) { const l = falls.slice(0, 3).map((r) => (fnames[r.activity] || ("type " + r.activity)) + ` (${(100 * r.detected / r.trials).toFixed(0)}%)`); s += ` The least detected fall types were ${l.slice(0, -1).join(", ")} and ${l[l.length - 1]}. Rolling out of bed starts close to the floor and sliding down a wall slows the descent, so both produce weaker impacts; these types deserve priority when more training data are collected.`; }
  return s;
}

function mciText(CG, f, c) {
  const best = (t) => CG.filter((r) => r.task.startsWith(t)).reduce((x, y) => (y.AUROC > x.AUROC ? y : x));
  const d = best("D"), cdem = best("C");
  return `**Why was MCI so difficult?** Several factors probably explain why MCI (best AUROC ${f(d.AUROC)}, 95% range ${f(d.AUROC_lo)}–${f(d.AUROC_hi)}) was much harder to detect than dementia (${f(cdem.AUROC)}). First, by definition, people with MCI remain largely independent in everyday activities, whereas dementia impairs them [@petersen2004;@albert2011]; scripted household tasks are therefore completed almost normally in MCI, and the remaining differences (hesitation, small sequencing errors, slower completion) are subtle. Second, the ambient sensors record where and when a person moves and which items are touched, but not the quality of the action, so many of the errors that experimenters score are invisible to them. Third, the MCI group (${c.cog.mci} people) is clinically heterogeneous, and its age overlaps with that of the healthy group, so part of any difference reflects age rather than cognition (impaired vs. healthy fell from AUROC ${f(best("A").AUROC)} to ${f(best("B").AUROC)} when the healthy group was restricted to people aged 60+). Fourth, each person contributed a single session in a laboratory apartment, whereas studies that linked in-home sensor data to MCI followed people for months and looked at changes in walking speed and activity over time [@hayes2008;@kaye2011]. Longitudinal monitoring of within-person change, rather than a single comparison with other people, is therefore the more promising route for MCI.`;
}

function forgetText(N, tab) {
  return require("./revision_numbers")(N).forgetText(tab);
}

function olderAdults(T, f, pct, pv, tab) {
  const { P, Tab, hl } = require("./lib");
  const R = require("./revision_numbers")(T.__N);
  const OC = R.OC, OD = R.OD, CI = T("T6e_coverage_ci"), DI = T("T6f_false_alarm_distribution");
  const SVO = "Falls, magnitude detector: older adults (HAR70+)", SVY = "Falls, magnitude detector: young volunteers (falls dataset)";
  const ci = (svc, cal, B) => CI.find((r) => r.service === svc && r.policy === cal && r.budget_per_h === B);
  const di = (svc, cal, B) => DI.find((r) => r.service === svc && r.policy === cal && r.budget_per_h === B);
  const covTxt = (r) => `${r.covered}/${r.people} (${pct(r.coverage)}; ${f(100 * r.coverage_lo, 1)}–${f(100 * r.coverage_hi, 1)})`;
  const rateTxt = (d) => `${f(d.pooled_fa_per_h, 2)} (${f(d.pooled_lo_poisson, 2)}–${f(d.pooled_hi_poisson, 2)})`;
  const defs = [
    ["young volunteers (17)", SVO, "Young volunteers (17) → older adults", (B) => R.od("young", B).fall_sensitivity_on_volunteers],
    ["young volunteers (17) + personal ADL margin", SVO, "Young volunteers (17) + personal ADL margin → older adults", () => null],
    ["older adults, leave-one-out (17)", SVO, "Older adults (17, leave-one-out) → older adults", (B) => R.od("older", B).fall_sensitivity_on_volunteers],
    ["young volunteers, leave-one-out (16)", SVY, "Young volunteers (16, leave-one-out) → young volunteers (reference)", (B) => R.oc("young volunteers, leave-one-out (16)", B).fall_sensitivity],
  ];
  const rows = [];
  R.budgetsOA.forEach((B) => defs.forEach(([cal, svc, lab, sens]) => {
    const q = ci(svc, cal, B), d = di(svc, cal, B), o = R.oc(cal, B), se = sens(B);
    rows.push([lab, f(B, 0), f(o.median_threshold, 3), covTxt(q), `${d.people_with_fa}/${d.people}`, rateTxt(d), se === null ? "–" : pct(se, 1)]);
  }));
  const out = [];
  out.push(P(hl(R.olderText ? R.olderText(tab) : "OLDER_ADULT_TEXT_PENDING")));
  out.push(...Tab(hl(`How far does the fall false-alarm guarantee reach? A fall detector restricted to the magnitude of the waist (lower-back) acceleration was trained on the young volunteers and streamed over the free-living recordings of the 18 adults aged 70–95 years of HAR70+ (no falls, so every alert is a false alarm). B: false-alarm budget per hour. Thresholds were set by the subject-level conformal rule (δ = 0.1) from the calibration people shown. Threshold: median over folds. Coverage: share of monitored people within the budget B (exact 95% CI). Pooled FA/h: all false alarms divided by all monitored hours (exact Poisson 95% CI). Fall sensitivity: simulated falls of the volunteers (held-out-person scores) detected at the same threshold; it cannot be measured in older adults.`),
    ["Calibration → monitored people", "B", "Threshold", "Coverage (95% CI)", "People with ≥ 1 FA", "Pooled FA/h (95% CI)", "Fall sensitivity (volunteers)"], rows, [27, 6, 10, 19, 9, 16, 13], { hl: true, key: "older" }));
  return out;
}
