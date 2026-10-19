#!/usr/bin/env sh
# Runs the complete IoT-CARES pipeline in the required order (Linux / macOS).
set -e
cd "$(dirname "$0")"
python 01_download_data.py
python 02_prepare_fog_parkinson.py
python 03_prepare_har70_older_adults.py
python 04_prepare_falls.py
python 05_prepare_ecg_mitbih.py
python 06_prepare_cognitive_smarthome.py
python 07_extract_edge_features.py
python 08_benchmark_models.py falls
python 08_benchmark_models.py fog
python 08_benchmark_models.py har70
python 08b_benchmark_ecg.py
python 09_federated_personalisation.py fog
python 09_federated_personalisation.py har70
python 09_federated_personalisation.py falls
python 10_calibrated_alarms.py fog
python 10_calibrated_alarms.py falls
python 11_cognitive_surveillance.py
python 12_explainability_robustness.py falls
python 12_explainability_robustness.py fog
python 12_explainability_robustness.py har70
python 13_edge_iot_simulation.py
# revision analyses (peer review)
python 18_revision_fusion_calibration.py offline
python 18_revision_fusion_calibration.py retrain falls
python 18_revision_fusion_calibration.py retrain fog
# python 18_revision_fusion_calibration.py retrain har70   # optional: several hours on 2 cores; not run for the paper
python 20_revision_falls_guarantee.py split
python 20_revision_falls_guarantee.py har70
python 21_revision_personalisation.py fog
python 21_revision_personalisation.py har70
python 22_revision_ecg.py
python 19_revision_alarm_statistics.py
python 14_make_figures.py
python 15_make_diagrams.py
python 16_collect_results.py
echo "done - results in ./results ; build the paper with: node manuscript/build_manuscript.js"
