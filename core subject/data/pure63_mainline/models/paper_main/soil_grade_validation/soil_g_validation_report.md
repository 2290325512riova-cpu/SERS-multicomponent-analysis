# Soil-G validation report

Run id: `soil_g_extratrees_p1_20260622_142841`

## Soil-G definition

Soil-G used ternary spiked-soil spectra only. Grades were assigned from the total molar loading of Thiram, MG, and 4-MBA.

| soil_g_grade   |   soil_g_total_molar |   n_spectra |
|:---------------|---------------------:|------------:|
| low            |              2.1e-05 |          24 |
| middle         |              3e-05   |          11 |
| high           |              0.00012 |          21 |

## Input-set comparison

| run_id                               | feature_set              |   n_spectra |   n_features |   macro_f1_mean |   macro_f1_std |   balanced_accuracy_mean |   balanced_accuracy_std |   accuracy_mean |   accuracy_std |   oof_macro_f1 |   oof_balanced_accuracy |   oof_accuracy |
|:-------------------------------------|:-------------------------|------------:|-------------:|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|---------------:|------------------------:|---------------:|
| soil_g_extratrees_p1_20260622_142841 | mg1616_single_band_area  |          56 |            1 |        0.368754 |      0.147828  |                 0.417778 |               0.162892  |        0.375758 |      0.120452  |       0.402778 |                0.416486 |       0.375    |
| soil_g_extratrees_p1_20260622_142841 | manual_marker_band_areas |          56 |           16 |        0.855291 |      0.0932497 |                 0.89     |               0.0572519 |        0.857576 |      0.0798874 |       0.848405 |                0.886905 |       0.857143 |
| soil_g_extratrees_p1_20260622_142841 | shap_global_top10pct     |          56 |          141 |        0.89963  |      0.108822  |                 0.913333 |               0.0988826 |        0.909091 |      0.0909091 |       0.899655 |                0.914141 |       0.910714 |
| soil_g_extratrees_p1_20260622_142841 | full_p1_spectrum         |          56 |         1401 |        0.931481 |      0.0795435 |                 0.946667 |               0.0557773 |        0.927273 |      0.07606   |       0.926768 |                0.944444 |       0.928571 |

## Per-grade recall

| run_id                               | feature_set              |   grade_label | grade   |   recall |   support |
|:-------------------------------------|:-------------------------|--------------:|:--------|---------:|----------:|
| soil_g_extratrees_p1_20260622_142841 | mg1616_single_band_area  |             0 | low     | 0.375    |        24 |
| soil_g_extratrees_p1_20260622_142841 | mg1616_single_band_area  |             1 | middle  | 0.636364 |        11 |
| soil_g_extratrees_p1_20260622_142841 | mg1616_single_band_area  |             2 | high    | 0.238095 |        21 |
| soil_g_extratrees_p1_20260622_142841 | manual_marker_band_areas |             0 | low     | 0.708333 |        24 |
| soil_g_extratrees_p1_20260622_142841 | manual_marker_band_areas |             1 | middle  | 1        |        11 |
| soil_g_extratrees_p1_20260622_142841 | manual_marker_band_areas |             2 | high    | 0.952381 |        21 |
| soil_g_extratrees_p1_20260622_142841 | shap_global_top10pct     |             0 | low     | 0.833333 |        24 |
| soil_g_extratrees_p1_20260622_142841 | shap_global_top10pct     |             1 | middle  | 0.909091 |        11 |
| soil_g_extratrees_p1_20260622_142841 | shap_global_top10pct     |             2 | high    | 1        |        21 |
| soil_g_extratrees_p1_20260622_142841 | full_p1_spectrum         |             0 | low     | 0.833333 |        24 |
| soil_g_extratrees_p1_20260622_142841 | full_p1_spectrum         |             1 | middle  | 1        |        11 |
| soil_g_extratrees_p1_20260622_142841 | full_p1_spectrum         |             2 | high    | 1        |        21 |

## Permutation test

| run_id                               | feature_set              | metric       |   observed_metric |   p_value |   n_permutations |   null_mean |   null_std | permutation_mode                             |
|:-------------------------------------|:-------------------------|:-------------|------------------:|----------:|-----------------:|------------:|-----------:|:---------------------------------------------|
| soil_g_extratrees_p1_20260622_142841 | mg1616_single_band_area  | OOF_macro_F1 |          0.402778 | 0.130187  |            10000 |    0.334106 |  0.0622029 | OOF_prediction_label_permutation_within_fold |
| soil_g_extratrees_p1_20260622_142841 | manual_marker_band_areas | OOF_macro_F1 |          0.848405 | 9.999e-05 |            10000 |    0.325769 |  0.0645004 | OOF_prediction_label_permutation_within_fold |
| soil_g_extratrees_p1_20260622_142841 | shap_global_top10pct     | OOF_macro_F1 |          0.899655 | 9.999e-05 |            10000 |    0.333643 |  0.0659936 | OOF_prediction_label_permutation_within_fold |
| soil_g_extratrees_p1_20260622_142841 | full_p1_spectrum         | OOF_macro_F1 |          0.926768 | 9.999e-05 |            10000 |    0.330858 |  0.0646671 | OOF_prediction_label_permutation_within_fold |

## Feature definitions

| feature_set              |   n_rows_in_definition_file |
|:-------------------------|----------------------------:|
| full_p1_spectrum         |                           1 |
| manual_marker_band_areas |                          17 |
| mg1616_single_band_area  |                           2 |
| shap_global_top10pct     |                         142 |

Interpretation note: this is a matrix-matched fivefold assessment within spiked soil spectra, not an external unknown-soil validation or continuous concentration calibration.