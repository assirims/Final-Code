// Sentences of the response letter that report the newest analyses (scripts 18, 20 and 21).
module.exports = function (N) {
  const U = require("./results_util")(N);
  const { T, f, pct, pv } = U;
  const R = require("./revision_numbers")(N);
  const CI = T("T6e_coverage_ci"), DI = T("T6f_false_alarm_distribution");
  const ci = (svc, p, b) => CI.find((r) => r.service === svc && r.policy === p && r.budget_per_h === b);
  const di = (svc, p, b) => DI.find((r) => r.service === svc && r.policy === p && r.budget_per_h === b);
  const cov = (r) => `${r.covered}/${r.people} (${pct(r.coverage)}; 95% CI ${pct(r.coverage_lo)}–${pct(r.coverage_hi)})`;
  const X = {};

  // split conformal
  const SPN = "Falls (split conformal)", SPP = "split conformal (exact exchangeability)";
  const s0 = R.SP.find((r) => r.budget_per_h === 0), s2 = R.SP.find((r) => r.budget_per_h === 2);
  const m0 = T("T6_alarm_budget_falls").find((r) => r.policy === "conformal (proposed)" && r.budget_per_h === 0);
  const worst = T("S18_falls_split_per_subject").filter((r) => r.budget_per_h === 0).reduce((x, y) => (y.fa > x.fa ? y : x));
  X.split = `In this variant, coverage was ${cov(ci(SPN, SPP, 0))} at B = 0/h with a fall sensitivity of ${pct(s0.fall_sensitivity)} (main analysis ${pct(m0.fall_sensitivity)}), and ${ci(SPN, SPP, 2).covered}/${ci(SPN, SPP, 2).people} at B = 2/h with ${pct(s2.fall_sensitivity)}. Exact exchangeability did not cost sensitivity, but it made the threshold depend on which 10 people were drawn for calibration (in one fold the threshold was only ${f(worst.threshold, 2)}, and that person raised ${worst.fa} of the ${s0.total_fa} false alarms).`;

  // older adults
  X.older = R.olderResponse || "";
  X.margin = R.marginResponse || "";

  // personalisation remedies
  const H = R.h, G = R.g;
  X.forget = `Fine-tuning only the output layer avoided the loss: after 1 and 5 minutes, activity macro-F1 was ${H.last1} and ${H.last5} (federated model ${H.ref}; all layers ${H.all1} and ${H.all5}) and FoG AUROC was ${G.last1} and ${G.last5} (federated model ${G.ref}; all layers ${G.all1} and ${G.all5}), and the rare activity classes were preserved (stairs F1 ${H.stairsLast5} vs. ${H.stairsAll5} with all layers). These gains over the federated model were not statistically significant (Holm-adjusted p ≥ ${R.lastMinHolm}). Class-balanced replay from federated statistics reduced the loss of all-layer fine-tuning after 5 minutes (activity ${H.stat5}, FoG ${G.stat5}) but not after 1 minute (activity ${H.stat1} vs. ${H.all1}), and replay of real windows gave ${H.real5} (activity, 5 min). Guided onboarding sessions that contained every activity, or a freeze, reduced the per-person loss of routine fine-tuning but did not make the models better than the federated model (${tab_("forget")}). We report this as a partly negative result: forgetting can be avoided by adapting only the output layer, but with a few minutes of data, personalisation of the gait and activity classifiers brought no significant gain.`;

  // fusion alternatives (script 18 retrain)
  const FA = N.tables.T3c_fusion_alternatives || [];
  const alt = (s, m) => FA.find((r) => r.service === s && r.method.startsWith(m));
  const sc = (s, r) => (r ? f(s === "har70" ? r.MacroF1 : r.AUROC, s === "falls" ? 4 : 3) : "–");
  const W = T("T3b_fusion_weights");
  const w5 = (s) => W.find((r) => r.service === s && Math.abs(r.w_gbt - 0.5) < 1e-9).pooled;
  X.fusion = FA.length ? `We also added the two simple alternatives: a single GBT trained on both views (early fusion) reached ${sc("fog", alt("fog", "GBT on concatenated"))} (FoG) and ${sc("falls", alt("falls", "GBT on concatenated"))} (falls), and the average of one GBT per view reached ${sc("fog", alt("fog", "GBT(descriptors) + GBT"))} and ${sc("falls", alt("falls", "GBT(descriptors) + GBT"))}, compared with ${f(w5("fog"))} and ${f(w5("falls"), 4)} for the CARES ensemble${alt("har70", "GBT on concatenated") ? `; for activity, ${sc("har70", alt("har70", "GBT on concatenated"))} and ${sc("har70", alt("har70", "GBT(descriptors) + GBT"))} vs. ${f(w5("har70"))}` : " (for activity, a six-class GBT on the MiniRocket-lite features in each of 18 folds was computationally prohibitive on our server; the script is included and can be run)"}. The ensemble was numerically highest for FoG, but none of the differences was significant (Holm-adjusted p ≥ ${pv(Math.min(...FA.filter((x) => x.service === "fog" && x.p_holm !== null && x.p_holm !== undefined).map((x) => x.p_holm)))}); it performs on a par with other reasonable ways of combining the two views, which is why we no longer present it as a new method.` : "";
  function tab_(k) { return require("./lib").tab(k); }
  return X;
};
