# Soil-G 加标土壤总负荷分级实验记录

更新时间：2026-06-22

## 实验定位

Soil-G 用于回应论文主线中的一个关键问题：三元共存和土壤基质同时存在时，MG 1616 cm^-1 单一标志峰是否仍能支撑浓度档判断。该实验不是连续浓度回归，也不是未知真实土壤外部验证，而是加标土壤基质内五折评估。

## 数据与标签

输入数据位于：

`core subject/data/pure63_mainline/models/paper_main/soil_validation/`

使用 56 条三元加标土壤光谱，标签按 Thiram、MG 和 4-MBA 的总摩尔负荷划分：

| 档位 | 总摩尔负荷 | 光谱数 |
|---|---:|---:|
| low | 2.1e-5 mol/L | 24 |
| middle | 3.0e-5 mol/L | 11 |
| high | 1.2e-4 mol/L | 21 |

## 脚本与输出

脚本：

`core subject/scripts/analysis/run_soil_grade_validation.py`

输出目录：

`core subject/data/pure63_mainline/models/paper_main/soil_grade_validation/`

主要输出文件：

| 文件 | 内容 |
|---|---|
| `soil_g_input_comparison_summary.csv` | 四类输入集的五折均值、标准差和 OOF 指标 |
| `soil_g_fold_metrics.csv` | 每折 macro-F1、balanced accuracy 和 accuracy |
| `soil_g_oof_predictions.csv` | 每条三元土壤谱的 OOF 预测结果 |
| `soil_g_confusion_matrix.csv` | 四类输入集的三档混淆矩阵 |
| `soil_g_per_class_recall.csv` | low/middle/high 各档召回率 |
| `soil_g_permutation_test.csv` | OOF macro-F1 置换检验 |
| `soil_g_feature_sets.csv` | 单峰、人工峰组、SHAP 波数和全谱输入定义 |
| `soil_g_validation_report.md` | 自动生成的实验报告 |

最终 run id：

`soil_g_extratrees_p1_20260622_142841`

## 输入集设计

四类输入均使用 ExtraTrees/p1 和相同分层五折划分：

| 输入集 | 特征数 | 定义 |
|---|---:|---|
| MG 1616 cm^-1 峰面积 | 1 | p1 光谱中 1616 cm^-1 ±5 cm^-1 窗口积分 |
| 人工特征峰面积 | 16 | Thiram、MG 和 4-MBA 主要标志峰 ±5 cm^-1 窗口积分 |
| SHAP 前 10% 波数 | 141 | 六项混合体系任务平均 mean \|SHAP\| 排序前 10% 波数 |
| 完整 p1 光谱 | 1401 | 400-1800 cm^-1 全部 p1 预处理波数 |

## 主要结果

| 输入集 | Macro-F1 | Balanced accuracy | Accuracy | OOF macro-F1 | 置换检验 p |
|---|---:|---:|---:|---:|---:|
| MG 1616 cm^-1 峰面积 | 0.369 ± 0.148 | 0.418 ± 0.163 | 0.376 ± 0.120 | 0.403 | 0.130 |
| 人工特征峰面积 | 0.855 ± 0.093 | 0.890 ± 0.057 | 0.858 ± 0.080 | 0.848 | <1.0e-4 |
| SHAP 前 10% 波数 | 0.900 ± 0.109 | 0.913 ± 0.099 | 0.909 ± 0.091 | 0.900 | <1.0e-4 |
| 完整 p1 光谱 | 0.931 ± 0.080 | 0.947 ± 0.056 | 0.927 ± 0.076 | 0.927 | <1.0e-4 |

## 论文写法

正文中应将 Soil-G 写成竞争吸附主线的应用验证：MG 1616 cm^-1 单峰输入在三元加标土壤总负荷分级中未形成显著分级能力，而人工特征峰、SHAP 高贡献波数和全谱输入显著提高性能。这说明土壤基质中的离散档判别依赖多分子指纹峰和共存敏感谱区，而不是单个 MG 核心峰。

边界表述必须保留：Soil-G 属于加标土壤基质内五折评估，不能写成未知真实土壤外部验证，也不能写成连续浓度定量模型。
