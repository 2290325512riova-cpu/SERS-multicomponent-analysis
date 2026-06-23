# 项目交接提示词（v3.2 · 2026-06-01 D021/D022强叙述与SHAP归属收束）

## 一句话概述

SERS（表面增强拉曼）+ 机器学习论文项目，目标期刊 **Spectrochimica Acta Part A / Talanta / Analyst（Q1-Q2 分析化学）**。三组分农药混合物（Thiram / Malachite Green / 4-MBA）于自制柠檬酸 AgNPs 上的同时定性+半定量筛查。

## 当前主线（锁定，必须记住）

**三组分 SERS 混合体系中存在竞争吸附诱导的 MG 1616 cm⁻¹ 谱峰压制；可解释 ML 将该混合物谱干扰定位到化学峰区，并转化为紧凑谱区筛查流程。**

核心叙事 = **混合物谱干扰发现 + 可解释ML闭环**：
1. **头牌发现**：MG 1616 cm⁻¹ 峰在共存下被显著压制（中位强度比 0.355 MG+MBA / 0.206 MG+Thiram / 0.129 三元；283/358 三元谱 <0.5×纯MG），主文强写为**竞争吸附诱导的混合物谱干扰**。1616 是 clean core；1172/1394 写成 MG 相关重叠敏感峰区和峰包络重塑证据。
2. **系统 benchmark**：13模型 × 6预处理 × 6任务 = 468 实验点，ExtraTrees 实际最优。
3. **SHAP 解释**：TreeSHAP 归因落到化学峰和干扰敏感区；当前正式输出为 **22/29 峰簇落在已知化学谱带，覆盖 91.2% total SHAP mass**。其中 18 个为自身分析物峰，2 个为 cross-component 干扰带，2 个为 curated Thiram 860/1444 文献峰。约 24/29 的共吸附/参考峰扩展只放 SI 备选。
4. **SHAP-guided 特征选择**：保留 ~10%（141/1401）波数，macro-F1 仍 >0.95。
5. **加标土壤基质筛查验证**：soil-matrix CV AUC 0.956-0.996 + 空白特异 + 置换检验；`same-matrix CV` 只放方法细节。

干扰发现是灵魂；benchmark/SHAP/FS/土壤为其铺垫、解释、延伸、落地。

## 必须先读的文件（按顺序）

```
project_memory/DECISIONS.md          — 冻结决策（D020=干扰主线 + 行对齐已硬验）
project_memory/AGENTS.md             — Paper Claim Guardrails
project_memory/EXPERIMENT_TRACKER.md — 实验历史与数据位置
reports/图表与表格清单.md             — v3 八图方案（图表设计权威来源）
reports/论文写作路线图.md             — 正文结构
reports/全局进度看板.md               — 当前进度
```

## 关键冻结决策（违反会被批评）

- Grouped CV **不入论文**（D017）；Random CV 不能叫 "external validation"（D001）
- RamanNet-Lite 是 DL 代表模型，不是 star model（D013）；**ExtraTrees 实际最优，经典>DL 是结论之一**（中性陈述，引 Grinsztajn 2022）
- SHAP 归属按 D022 写：正文和主图只写正式输出 22/29 + 91.2% SHAP-mass consistency；20/29 + 88.1% 仅留 ST1/internal audit，不进正文叙述或主图；不要把 24/29 或更高写成正文主结果
- 增强叫 "composition-constrained spectral mixing"，不用 "Beer-Lambert augmentation"
- 主文/图题写“加标土壤基质筛查验证”，`same-matrix CV` 只作方法细节；非 "external validation"
- 浓度只作 3 离散档（序数筛查），**全程无连续标准曲线**

## 已澄清的旧"遗留问题"（2026-05-29 实地核查，全部不再是问题）

旧版 HANDOFF 列的遗留问题，逐条核查后大半已解决，记录免得 GPT 误判：

- ❌→✅ **RamanNet-Lite 未丢失**：`benchmark_full_matrix.csv` 实测 **13 模型 468 行完整**（1D-CNN/1D-ResNet/ExtraTrees/HistGradientBoosting/KAN-CNN/KNN/LDA/PLS-DA/RF/RamanNet-Lite/SVM/Spectrum-KAN/XGBoost），RamanNet-Lite 36 行齐全。Fig4 可直接做。
- ❌→✅ **`附/` 目录仍在**：师兄论文.pdf、参考文献/、skills/ 均存在，未删除。
- ❌→✅ **EXPERIMENT_TRACKER.md 存在**（22KB）。
- ✅ **增强消融失败暴露问题**：v3 已把增强消融降到 **SI S2**，正文不再暴露 no_aug 失败。
- ✅ **"5 tasks"→"6 tasks" 口径已统一**：G1 已补跑，全 6 任务就绪；`实验结果摘要.md`/`EXPERIMENT_TRACKER.md`/`CLAIM_EVIDENCE_MATRIX.md` 的旧 "5 tasks" 及对应行数（top20 100→120、mean|SHAP| 7005→8406、cluster 24→29、elimination 50→60）已于 2026-05-29 全部改正。

## 当前状态（v3）

- **图表方案与正式输出**：v3 正文图底稿已生成到 `figures/paper_main/`（Fig2-Fig8，PDF+PNG），正文表/SI表已生成到 `tables/paper_main/`。Fig1 仍待 user 手绘骨架；Fig2(a)(b) 的 TEM/UV-Vis 仍待 user 素材替换。
- **旧图旧脚本清理**：v2 旧编号图、早期 Origin 迭代图、临时参考图和 pycache 已在 2026-06-03 清理；当前保留 `scripts/figures/make_paper_figures.py`、`make_paper_tables.py`、`paper_style.py` 与 `build_origin_adjusted_project.py`。最终 Fig5/Fig8 复合图依赖 `figures/paper_main_origin/origin_clean_fig5_mg_suppression.png`、`origin_clean_fig8_soil_spectra.png` 和 `sers_ml_origin_clean.opju`。
- **行对齐**：X_p4[i] ↔ split.csv[i] 已硬验（D020，15/15 corr=1.0000）。Fig5 头牌地基可靠。
- **SHAP 归属新口径**：Fig6 正文采用“22/29 峰簇落在已知化学谱带 + 91.2% SHAP 加权一致性”；20/29 不在正文图或正文叙述中出现，仅留 ST1/internal audit，约 24/29 只作 SI 扩展。
- **出图交接单**：正由出图技法 workflow 产出（每图 matplotlib 配方 + 对标真实论文 + 全局样式），完成后落地。

## 图表数据状态速查（v3 编号）

| 图 | 数据 | 备注 |
|---|---|---|
| Fig1 流程 | user-drawn | 手绘骨架 |
| Fig2 基底 | (a)(b) user 提供 TEM+UV-Vis；(c)(d) 自有复测谱 | RSD 用同浓度10^-4 M纯组分 marker-band area 口径，正文不画全量异常波动 |
| Fig3 纯谱 | generated | 结构 inset 待 user 提供/正式排版时再补 |
| Fig4 benchmark | generated | 13模型完整；Fig4b 已改为 per-task model spread |
| Fig5 干扰头牌 | generated | 条件均值谱+差谱+Origin-clean 主谱嵌入 |
| Fig6 SHAP | generated | 正文只呈现 22/29 + 91.2% 正式口径 |
| Fig7 FS | generated | 10%/141 波数为正文 operating point |
| Fig8 土壤 | generated | ROC/blank/置换/土壤谱均已成图 |

## 后续任务清单（中文初稿阶段）

1. 下载并整理核心参考文献，开始中文初稿；写作按 D020/D021/D022 主线收束。
2. Fig1 与 Fig2(a)(b) 等 user 素材到位后再正式替换；其余当前图先作为中文初稿配图底稿。
3. 后续正式排版时可继续用 Origin/高质量绘图工具微调 Fig3/Fig5/Fig8 的谱图细节，但不改变数据和叙事结构。
4. 按 D021/D022 的发表型强叙述写 Methods/Results：头牌发现强、方法边界放 Methods/Discussion、SHAP 归属写成 count + SHAP-mass consistency，不要正文自我削弱。

## Python 环境

- 路径：`/d/anaconda3/python.exe`（Python 3.13.9，matplotlib/pandas/numpy 已装）
- Windows Store 的 `python` 不可用（exit 49）——脚本一律用 anaconda 全路径。

## 用户沟通风格（极其重要）

### 用户画像
- 化学/材料方向研究生，论文是毕业要求；对 ML/代码不熟但学习能力强，对化学实验和期刊规范非常熟悉。
- 有判断力，不盲从 AI；导师和师兄论文是最重要参考标准。

### 沟通特点
- **直接严厉**：犯错会直接说，认错改正即可，别玻璃心。
- **要求证据**：不接受"我觉得""一般来说"，必须拿数据/文献说话，且会逐篇去看引用的文献。
- **重视一致性**：每个文件/数字/表述都要对齐主线。
- **全程中文**，技术术语可用英文。
- **不要反射性附和**："你说得对"式的肯定他反感；直接给判断和证据。
- **RSD/多天测量**：正文按已发表 SERS 论文常用的代表性重复性口径展示；当前 Fig2 RSD 固定为同浓度10^-4 M纯组分 marker-band area RSD，不把全量逐folder异常波动当主图证据；默认实验合规，不要反复劝或加"若可恢复"之类对冲措辞。

### 容易踩的坑（血泪教训）
1. 不要用不相关文献做背书——会被逐篇核。
2. **不要把 Nature 等其他领域规范套到 SAA 化学期刊**。
3. **不要建议把化学表征（SEM/TEM/UV-vis）放 SI**——化学期刊必须在正文。
4. 不要随意放弃数据——丢了要找回，不要改口。
5. 不要在正文图暴露自己方法的失败。
6. **不要失忆**——操作前先读 DECISIONS.md；不要凭记忆下结论（曾误判"附/删了""只有12模型"，均不实）。
7. 不要过度谨慎反复确认。
8. 文献只找 SERS+ML+农药/食品安全 高度相关，不要生物/临床方向。

### 正确工作方式
- 先读文件/查数据再说话，不凭记忆。
- 犯错直接认错+给修正方案。
- 给建议附文献/数据证据。
- 做完一步汇报一步。
- 给论文叙述建议时按 D021/D022：中文优先，主文强写，不默认列自我削弱风险清单；SHAP 写 22/29 + 91.2% SHAP-mass consistency；以已发表 SERS/ML 农药文章的实际话术强度为参照。

## 中文稿逐节写作协议（2026-06-07，必须继承）

- 中文稿重修必须**一段一段或一节一节**推进，不能一次生成大段再事后解释。
- 每段/每节先找最贴合的已发表文章背书，再设计我们自己的文字。优先顺序：
  1. `附/参考文献/正式写作参考背书/`
  2. `附/参考文献/新文献/`
  3. `附/参考文献/A组`-`H组`
  4. `附/师兄论文_full.txt`
  5. 本地不足时再联网找同路线文章；开源可直接看，非开源则把 DOI/链接告诉用户下载。
- 给用户设计段落前，先说明选用的文献和对应段落。段落翻译/译述应尽量保持原文结构，保留原文中出现的引用标号、图表引用、公式、变量定义和章节位置；原文没有的不要补。
- 不要摘几句拼接成“背书”。Methods 对 Methods，Results 对 Results，引言对引言，结论对结论。
- 写作不能凭 Codex 自己的老套模板。先观察真实文章怎么写，再改成适配本项目 D020/D021/D022 主线的中文稿。
- 章节边界必须记住：第二章写方法定义（样品、采集、预处理、任务、模型实现、验证策略、指标公式、峰强比、SHAP计算）；第三章写结果比较与解释（模型优劣、主模型选择、MG压制、SHAP 22/29和91.2%、10%特征筛选、土壤AUC）。
- 如果用户质疑，不要机械附和；回到文献段落和项目证据，给出独立判断。

## 参考资料位置（均存在）

```
附/师兄论文_full.txt          — 同课题组、同基底、同方向已投稿论文
附/参考文献/                  — 按 A-H 组分类的 PDF
附/skills/绘图与数据分析/     — matplotlib/seaborn/nature-figure/shap 等 skill
                               （⚠️ nature-figure 风格≠SAA，化学期刊出图别直接套）
```

## 项目核心矛盾（定位用）

我们没有新模型、新基底、新化学创新，但有**竞争吸附诱导的 MG 1616 压制发现** + 系统性方法学闭环。说服力来自：① 干扰发现（MG 1616 竞争压制，三元中位仅 0.129× pure MG）；② 468 实验点系统性；③ SHAP 从"解释"到"10%紧凑筛查"的闭环；④ 加标土壤基质筛查验证。不要包装成"提出新模型"——贡献是发现+方法学框架+系统验证。
