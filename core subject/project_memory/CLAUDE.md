@AGENTS.md

# Claude-Specific Reminder

When proposing research strategy or project cleanup, cite the relevant entries in `project_memory/DECISIONS.md`, check `project_memory/COLLABORATION_PROTOCOL.md`, and update `reports/全局进度看板.md` after the phase is complete.

## Publication-Style Narrative Calibration

Follow D021/D022. Do not default to self-limiting "risk list" prose. The manuscript tone is Chinese-first and publication-forward. When the user asks for strategy, first answer in Chinese with project data and same-field publication practice; only translate to English later.

- 主文强写：本研究揭示三组分 SERS 混合体系中竞争吸附诱导的 MG 1616 cm⁻¹ 峰压制与混合物谱干扰。
- 1172/1394 cm⁻¹ 写成重叠敏感的 MG 相关峰区，用来证明峰包络重塑，不写成正文缺点。
- 随机五折 CV 是系统模型筛选；土壤结果写成加标土壤/土壤基质筛查验证。`same-matrix CV` 只在方法细节中出现。
- SHAP 写成定位化学峰区和干扰相关谱区，不要把正文写成 SHAP 免责声明。
- SHAP 峰归属按 D022 写成“数量匹配 + SHAP 贡献加权一致性”：正式输出为 22/29 个峰簇落在已知化学谱带，覆盖 91.2% total SHAP mass；20/29、88.1% 只作为 18 自身峰 + 2 cross-component 干扰带的分解层级，不再作为正式 baseline；更宽的共吸附/参考峰归属约 24/29、92.3% 只作为 SI 扩展。
- 10% SHAP 保留是主文压缩结果；5% 只作补充敏感性。
- 不要用“道德洁癖/审稿理想主义”的方式自断武功：正文按已发表 SERS/ML 农药论文的实际话术强度写，弱验证细节放在 Methods 或必要的 Discussion，不在摘要、图题和结果首句主动拆台。
- 强叙述不是编造证据：不能把 random CV 或 soil-only CV 写成 external validation，不能声称 Langmuir 常数、热力学证明或连续定量曲线。边界留在方法细节，主线保持强。

## Publication-Forward Strategy Rule

When the user asks for manuscript strategy, do not answer from an abstract "perfect validation" standard. Calibrate against what closely related SERS/ML pesticide-mixture papers actually publish and how they phrase their evidence.

- 先给中文判断，再给必要英文术语；不要先甩英文框架让用户翻译。
- 不要把方法细节主动包装成正文缺点。例如 random five-fold CV、matrix-matched soil validation、DL 未胜出、1172/1394 重叠区，都应放在对应的发表型叙事位置，而不是开头自我削弱。
- 不要因为追求"严谨姿态"把能支撑主线的数据写软。默认写法是：头牌发现强、方法边界放 Methods/Discussion、补充材料承接细节。
- 但强叙述不等于编造。不能写 external validation、cross-matrix transfer success、Langmuir isotherm、吸附常数、连续定量标准曲线，除非项目数据真的支持。
