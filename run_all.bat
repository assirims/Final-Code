@echo off
REM Runs the complete IoT-CARES pipeline in the required order (Windows).
cd /d %~dp0
python 01_download_data.py || exit /b 1
python 02_prepare_fog_parkinson.py || exit /b 1
python 03_prepare_har70_older_adults.py || exit /b 1
python 04_prepare_falls.py || exit /b 1
python 05_prepare_ecg_mitbih.py || exit /b 1
python 06_prepare_cognitive_smarthome.py || exit /b 1
python 07_extract_edge_features.py || exit /b 1
python 08_benchmark_models.py falls || exit /b 1
python 08_benchmark_models.py fog || exit /b 1
python 08_benchmark_models.py har70 || exit /b 1
python 08b_benchmark_ecg.py || exit /b 1
python 09_federated_personalisation.py fog || exit /b 1
python 09_federated_personalisation.py har70 || exit /b 1
python 09_federated_personalisation.py falls || exit /b 1
python 10_calibrated_alarms.py fog || exit /b 1
python 10_calibrated_alarms.py falls || exit /b 1
python 11_cognitive_surveillance.py || exit /b 1
python 12_explainability_robustness.py falls || exit /b 1
python 12_explainability_robustness.py fog || exit /b 1
python 12_explainability_robustness.py har70 || exit /b 1
python 13_edge_iot_simulation.py || exit /b 1
REM revision analyses (peer review)
python 18_revision_fusion_calibration.py offline || exit /b 1
python 18_revision_fusion_calibration.py retrain falls || exit /b 1
python 18_revision_fusion_calibration.py retrain fog || exit /b 1
REM python 18_revision_fusion_calibration.py retrain har70   (optional: several hours on 2 cores; not run for the paper)
python 20_revision_falls_guarantee.py split || exit /b 1
python 20_revision_falls_guarantee.py har70 || exit /b 1
python 21_revision_personalisation.py fog || exit /b 1
python 21_revision_personalisation.py har70 || exit /b 1
python 22_revision_ecg.py || exit /b 1
python 19_revision_alarm_statistics.py || exit /b 1
python 14_make_figures.py || exit /b 1
python 15_make_diagrams.py || exit /b 1
python 16_collect_results.py || exit /b 1
echo done - results in .\results ; build the paper with: node manuscript\build_manuscript.js
