// Numbers and sentences of the revision analyses (scripts 18-22) that are quoted in several places (results,
// discussion, abstract, response to reviewers). Everything is computed from paper_numbers.json.
module.exports = function (N) {
  const U = require("./results_util")(N);
  const { T, f, pct, pv } = U;
  const R = {};

  // ---------------------------------------------------------------- personalisation remedies (script 21)
  const RV = { fog: T("T5b_personalisation_revision_fog"), har70: T("T5b_personalisation_revision_har70") };
  const after = (s, m) => RV[s].find((r) => r.method === m && r.eval_set.startsWith("after"));
  const outside = (s, m) => RV[s].find((r) => r.method === m && r.eval_set.startsWith("outside"));
  const k = (s) => (s === "har70" ? "MacroF1" : "AUROC");
  const v = (s, m, o = false) => { const r = o ? outside(s, m) : after(s, m); return r ? f(r[k(s)]) : "–"; };
  R.rv = { after, outside, v };
  const lastRows = ["FT last layer (1 min)", "FT last layer (5 min)"].flatMap((m) => [after("har70", m), after("fog", m)]);
  R.lastMinHolm = pv(Math.min(...lastRows.map((r) => r.p_holm)));
  R.lastMinRaw = pv(Math.min(...lastRows.map((r) => r.p_vs_FedAvg)));
  const h = (m) => after("har70", m);
  R.h = {
    ref: v("har70", "FedAvg MLP"), all1: v("har70", "FT all layers (1 min)"), all5: v("har70", "FT all layers (5 min)"),
    last1: v("har70", "FT last layer (1 min)"), last5: v("har70", "FT last layer (5 min)"),
    stat1: v("har70", "FT all + statistics replay (1 min)"), stat5: v("har70", "FT all + statistics replay (5 min)"),
    lstat1: v("har70", "FT last + statistics replay (1 min)"), lstat5: v("har70", "FT last + statistics replay (5 min)"),
    real1: v("har70", "FT all + real-data replay (1 min)"), real5: v("har70", "FT all + real-data replay (5 min)"),
    stairsRef: f(h("FedAvg MLP").F1_stairs), stairsAll5: f(h("FT all layers (5 min)").F1_stairs), stairsLast5: f(h("FT last layer (5 min)").F1_stairs),
    lyingAll5: f(h("FT all layers (5 min)").F1_lying), lyingLast5: f(h("FT last layer (5 min)").F1_lying), lyingRef: f(h("FedAvg MLP").F1_lying),
    gRef: v("har70", "FedAvg MLP", true), gAll1: v("har70", "FT all layers, guided onboarding (1 min)", true), gAll5: v("har70", "FT all layers, guided onboarding (5 min)", true),
    gLast1: v("har70", "FT last layer, guided onboarding (1 min)", true), gLast5: v("har70", "FT last layer, guided onboarding (5 min)", true),
    gRout1: v("har70", "FT all layers, routine onboarding (1 min)", true), gRout5: v("har70", "FT all layers, routine onboarding (5 min)", true),
    gMedRef: f(outside("har70", "FedAvg MLP").subject_median), gMedAll5: f(outside("har70", "FT all layers, guided onboarding (5 min)").subject_median),
    gMedRout5: f(outside("har70", "FT all layers, routine onboarding (5 min)").subject_median), gMedLast5: f(outside("har70", "FT last layer, guided onboarding (5 min)").subject_median),
    gStairsRef: f(outside("har70", "FedAvg MLP").F1_stairs),
  };
  const g = (m) => after("fog", m);
  R.g = {
    ref: v("fog", "FedAvg MLP"), all1: v("fog", "FT all layers (1 min)"), all5: v("fog", "FT all layers (5 min)"),
    last1: v("fog", "FT last layer (1 min)"), last5: v("fog", "FT last layer (5 min)"),
    stat1: v("fog", "FT all + statistics replay (1 min)"), stat5: v("fog", "FT all + statistics replay (5 min)"),
    real1: v("fog", "FT all + real-data replay (1 min)"), real5: v("fog", "FT all + real-data replay (5 min)"),
    medRef: f(g("FedAvg MLP").subject_median), medLast5: f(g("FT last layer (5 min)").subject_median), medLast1: f(g("FT last layer (1 min)").subject_median),
    nFreeze: outside("fog", "FedAvg MLP").people,
    fRef: v("fog", "FedAvg MLP", true), fAll1: v("fog", "FT all layers, onboarding with a freeze (1 min)", true), fAll5: v("fog", "FT all layers, onboarding with a freeze (5 min)", true),
    fLast1: v("fog", "FT last layer, onboarding with a freeze (1 min)", true), fLast5: v("fog", "FT last layer, onboarding with a freeze (5 min)", true),
    fRout1: v("fog", "FT all layers, routine onboarding (1 min)", true), fRout5: v("fog", "FT all layers, routine onboarding (5 min)", true),
    fMedRef: f(outside("fog", "FedAvg MLP").subject_median), fMedLast5: f(outside("fog", "FT last layer, onboarding with a freeze (5 min)").subject_median),
  };
  const H = R.h, G = R.g;
  R.forgetText = (tab) => `**Can forgetting be avoided?** ${tab("forget")} tests the remedies on the same folds and seeds. Fine-tuning only the output layer avoided the loss: activity macro-F1 was ${H.last1} and ${H.last5} after 1 and 5 minutes (federated model without onboarding ${H.ref}; all layers ${H.all1} and ${H.all5}), and FoG AUROC was ${G.last1} and ${G.last5} (without onboarding ${G.ref}; all layers ${G.all1} and ${G.all5}). The rare classes were preserved: after 5 minutes, stairs F1 was ${H.stairsLast5} with output-layer fine-tuning, ${H.stairsAll5} with all layers and ${H.stairsRef} without onboarding, and lying F1 was ${H.lyingLast5}, ${H.lyingAll5} and ${H.lyingRef}. None of the differences from the federated model was statistically significant, however (all Holm-adjusted p ≥ ${R.lastMinHolm}), and the median FoG AUROC per person was slightly lower (${G.medLast1} and ${G.medLast5} vs. ${G.medRef}), so output-layer fine-tuning is a safe default rather than a demonstrated improvement. Class-balanced replay of pseudo-windows drawn from federated class statistics reduced the loss of all-layer fine-tuning after 5 minutes (activity ${H.stat5} vs. ${H.all5}; FoG ${G.stat5} vs. ${G.all5}) but not after 1 minute (activity ${H.stat1} vs. ${H.all1}), and it lowered the output-layer variant (activity ${H.lstat1} and ${H.lstat5}). Replay of real training windows, which would require storing other people's data on the device, gave ${H.real1} and ${H.real5} (activity) and ${G.real1} and ${G.real5} (FoG). Onboarding sessions that contained the target events did not make the models better than the federated model either. A guided activity session that covered every activity of the person gave macro-F1 ${H.gAll1} and ${H.gAll5} (all layers) and ${H.gLast1} and ${H.gLast5} (output layer) on the windows outside every onboarding session, compared with ${H.gRef} without onboarding and ${H.gRout1} and ${H.gRout5} after routine onboarding; pooled macro-F1 is low for every method in this evaluation set because the guided sessions used up most stairs windows of people who rarely climbed stairs (stairs F1 ${H.gStairsRef} without onboarding), but the guided session avoided the per-person loss of routine fine-tuning (median ${H.gMedAll5} vs. ${H.gMedRout5} after 5 minutes; ${H.gMedRef} without onboarding). For the ${G.nFreeze} people with freezes, onboarding that started just before the first freeze gave AUROC ${G.fAll1} and ${G.fAll5} (all layers) and ${G.fLast1} and ${G.fLast5} (output layer), compared with ${G.fRef} without onboarding and ${G.fRout1} and ${G.fRout5} after routine onboarding (person-level median ${G.fMedLast5} vs. ${G.fMedRef}). Thus, forgetting can be avoided by adapting only the output layer, but with a few minutes of data none of the procedures made the gait and activity models clearly better than the federated model.`;
  R.remedyDiscussion = `Fine-tuning only the output layer avoided this loss (activity ${H.last5}, FoG ${G.last5} after 5 minutes), and class-balanced replay from federated statistics reduced it, but no remedy, including guided onboarding sessions that contained stairs, shuffling or a freeze, was clearly better than the federated model without personalisation.`;
  R.remedyAbstract = `fine-tuning on routine activity lowered activity macro-F1 (${H.ref} to ${H.all5}) unless only the output layer was adapted (${H.last5})`;

  // ---------------------------------------------------------------- older adults as ADL-only source (script 20 har70)
  const Ts = (n) => (N.tables[n] ? N.tables[n] : []);
  const OC = Ts("T6c_falls_older_adults"), OD = Ts("T6d_falls_threshold_sources");
  const oc = (cal, B) => OC.find((r) => r.calibration === cal && r.budget_per_h === B);
  R.oc = oc; R.od = (src, B) => OD.find((r) => r.calibration_source.startsWith(src) && r.budget_per_h === B);
  R.OC = OC; R.OD = OD;
  R.budgetsOA = [...new Set(OC.map((r) => r.budget_per_h))].sort((a, b) => a - b);

  if (OC.length && N.tables.S19b_magnitude_detector) {
    const MD = N.tables.S19b_magnitude_detector;
    const CI = T("T6e_coverage_ci"), DI = T("T6f_false_alarm_distribution");
    const SVO = "Falls, magnitude detector: older adults (HAR70+)", SVY = "Falls, magnitude detector: young volunteers (falls dataset)";
    const ci = (svc, cal, B) => CI.find((r) => r.service === svc && r.policy === cal && r.budget_per_h === B);
    const di = (svc, cal, B) => DI.find((r) => r.service === svc && r.policy === cal && r.budget_per_h === B);
    const Y = "young volunteers (17)", O = "older adults, leave-one-out (17)", Mg = "young volunteers (17) + personal ADL margin";
    const y0 = ci(SVO, Y, 0), y2 = ci(SVO, Y, 2), dy0 = di(SVO, Y, 0), o0 = ci(SVO, O, 0), o2 = ci(SVO, O, 2), do0 = di(SVO, O, 0);
    const sy0 = R.od("young", 0).fall_sensitivity_on_volunteers, so0 = R.od("older", 0).fall_sensitivity_on_volunteers;
    const sy2 = R.od("young", 2).fall_sensitivity_on_volunteers, so2 = R.od("older", 2).fall_sensitivity_on_volunteers;
    const ty0 = oc(Y, 0).median_threshold, to0 = oc(O, 0).median_threshold, to2 = oc(O, 2).median_threshold;
    const raised = oc(Mg, 0).people_margin_raised + oc(Mg, 2).people_margin_raised;
    const hrs = f(MD.har70_hours, 1);
    const people = (k) => (k === 1 ? "one person" : `${k} people`);
    R.oa = { y0, y2, dy0, o0, o2, do0, sy0, so0, sy2, so2, ty0, to0, to2, raised, hrs };
    R.olderText = (tab) => `**How far does the fall guarantee reach?** ${tab("older")} reports the analysis with older adults' free-living data as an ADL-only source. The magnitude-only detector separated the volunteers' falls from their activities almost as well as the full sensor kit (window AUROC ${f(MD.window_auroc_young_lopo, 4)}, leave-one-person-out) and was streamed over ${Number(MD.har70_windows).toLocaleString("en-US")} windows of HAR70+ (${hrs} h; ${f(dy0.hours_min, 2)}–${f(dy0.hours_max, 2)} h per person). The threshold calibrated on the young volunteers did not raise a single false alarm in any older adult, at B = 0 or 2 per hour (${y0.covered}/${y0.people}; 95% CI ${pct(y0.coverage_lo)}–${pct(y0.coverage_hi)}; pooled rate ${f(dy0.pooled_fa_per_h, 2)} per hour, upper 95% limit ${f(dy0.pooled_hi_poisson, 2)}). For this detector, the volunteers' ADL protocol, which includes abrupt, fall-like activities, was therefore a conservative rather than an optimistic calibration source for older adults' everyday activities. The conservatism had a price. Calibrated on the older adults' own activity data (leave-one-older-adult-out), the threshold was much lower (median ${f(to0)} vs. ${f(ty0)} at B = 0), still kept ${o0.covered} of ${o0.people} older adults free of false alarms (${do0.total_fa} false alarm in ${hrs} h; pooled ${f(do0.pooled_fa_per_h, 2)} per hour, 95% CI ${f(do0.pooled_lo_poisson, 2)}–${f(do0.pooled_hi_poisson, 2)}), and detected ${pct(so0)} of the volunteers' simulated falls, compared with ${pct(sy0)} for the young-volunteer threshold (B = 2 per hour: ${pct(so2)} vs. ${pct(sy2)}). The personal ADL margin, computed from the first 5 minutes of each older adult's recording, raised ${raised === 0 ? "none of the thresholds" : people(raised) + "'s threshold"} in this scenario either. Three limits apply. Fall sensitivity could be measured only on the simulated falls of young volunteers, which are faster than real falls of older people; the HAR70+ recordings are short (median ${f(dy0.hours_median, 2)} h per person) and follow a semi-structured free-living protocol rather than days of everyday life; and the analysis used a reduced detector. It therefore does not establish the guarantee for frail older adults. It shows that the false-alarm calibration, which needs activity data only, can and should be performed on data from the target population.`;
    R.oaDiscussion = `In free-living data of ${y0.people} adults aged 70–95 years, the threshold of a reduced, accelerometer-only fall detector calibrated on the volunteers raised no false alarm in ${hrs} h, but it was conservative: calibrating on the older adults' own activity data allowed a lower threshold that detected ${pct(so0)} instead of ${pct(sy0)} of the volunteers' simulated falls. The guarantee itself has been verified only for young volunteers.`;
    R.oaAbstract = `In free-living data of ${y0.people} adults aged 70–95 years, a fall threshold calibrated on the volunteers raised no false alarm in ${hrs} h, but calibration on the older adults' own data detected more of the volunteers' simulated falls (${pct(so0)} vs. ${pct(sy0)}).`;
    R.olderResponse = `The magnitude detector separated the volunteers' falls from their activities with a window AUROC of ${f(MD.window_auroc_young_lopo, 4)}. Calibrated on the 17 young volunteers, its threshold raised no false alarm in any of the ${y0.people} older adults over ${hrs} h of recordings (coverage ${y0.covered}/${y0.people}; 95% CI ${pct(y0.coverage_lo)}–${pct(y0.coverage_hi)}; pooled rate ${f(dy0.pooled_fa_per_h, 2)} per hour, upper 95% limit ${f(dy0.pooled_hi_poisson, 2)}), so for this detector the volunteers' ADL protocol was a conservative rather than an optimistic calibration source. Calibrated on the older adults' own activity data (leave-one-older-adult-out), the threshold was much lower (median ${f(to0)} vs. ${f(ty0)}), ${o0.covered} of ${o0.people} older adults remained free of false alarms, and the threshold detected ${pct(so0)} of the volunteers' simulated falls instead of ${pct(sy0)}. We therefore state in the revised Discussion that the guarantee itself has been verified only for young volunteers, that the false-alarm calibration needs activity data only and should be performed on data from the target population, and that fall sensitivity in older adults remains untested because no real falls were available. We also list the short HAR70+ recordings (median ${f(dy0.hours_median, 2)} h per person) and the reduced detector as limitations.`;
    R.marginResponse = `We also built the scenario the reviewer suggested with older adults' data, in which the young-volunteer threshold is transferred to a different population: the margin computed from the first 5 minutes of each older adult's recording raised ${raised === 0 ? "none" : raised} of the ${y0.people} thresholds.`;
  }

  // ---------------------------------------------------------------- split conformal (script 20 split)
  R.SP = Ts("T6b_falls_split_conformal");
  return R;
};
