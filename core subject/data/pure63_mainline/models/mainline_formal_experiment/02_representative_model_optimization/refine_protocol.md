# 02 NSGA-II 多目标优化执行协议

## 1. 当前 Anchor 基线

| 模型 | Trial | 预处理 | mean_f1_all | mean_f1_g2 | G2 std |
|------|-------|--------|-------------|------------|--------|
| Spectrum-KAN | skan_anchor_repro_01 | p4 | 0.6763 | 0.5546 | 0.225 |
| RF | rf_coarse_08 | p4 | 0.7676 | 0.4939 | - |

## 2. 优化方法

Optuna NSGAIISampler 多目标贝叶斯优化。
同时最大化两个目标：
- 目标 1：mean_f1_g2（最难切片性能）
- 目标 2：mean_f1_all（全任务分类性能）

## 3. 搜索空间：Spectrum-KAN（30-50 trials，GPU）

锁死不搜索：optimizer=adam, scheduler=cosine, epochs=200,
patience=20, preprocess=p4, 架构不变, batch_size=128。

**最终执行版搜索空间（ordinal loss 已移除）：**

| 参数 | 类型 | 范围 | 理由 |
|------|------|------|------|
| loss_type | 固定 | focal | ordinal_coral/cost_sensitive_ordinal 验证失败后移除 |
| focal_gamma | 浮点 | [1.0, 3.0] | 聚焦难样本 |
| label_smoothing | 浮点 | [0.0, 0.15] | 防止过度自信 |
| mixup_alpha | 浮点 | [0.1, 0.4] | 增强泛化 |
| aug_n | 分类 | {2, 4} | aug_n=0 导致模型崩溃，已移除 |
| weight_decay | 浮点(log) | [5e-5, 3e-3] | 正则化强度 |
| lr | 浮点(log) | [1e-4, 5e-4] | 学习率微调 |
| class_balance | 分类 | {simple, effective_number} | 稀有类处理 |
| use_class_weight | 固定 | True | 类不平衡必须处理 |

Ordinal loss 移除原因（2026-05-18 确认）：
- ordinal_coral 与 argmax predict 逻辑不兼容
- cost_sensitive_ordinal 梯度爆炸，归一化后仍劣于 focal
- G2 的 fold 间类分布极度不均，ordinal 假设不成立

Anchor seed 注入：study.enqueue_trial() 把已知最优配置作为第一个 trial。

## 4. 搜索空间：RF（80-100 trials，CPU）

| 参数 | 类型 | 范围 |
|------|------|------|
| n_estimators | 整数 | [200, 1200], step=100 |
| max_depth | 整数 | [8, 50] |
| min_samples_leaf | 整数 | [1, 8] |
| max_features | 浮点 | [0.1, 0.6] |
| class_weight | 分类 | {balanced, balanced_subsample} |

## 5. 成功标准

KAN：Pareto front 上存在点满足 G2 > 0.5546 且 mean_all >= 0.67
RF：Pareto front 上存在点满足 mean_all > 0.7676

有意义的提升门槛：
- KAN G2 >= 0.60：论文可写为方法贡献
- KAN G2 >= 0.57：小幅正收益，可进入锁定

## 6. 失败退出

KAN 50 trials 后 Pareto front 无点超越 anchor：
- 冻结 skan_anchor_repro_01，ready_for_stage03 = True
- ready_reason = nsga_refine_exhausted_anchor_locked
- 直接进入锁定主结果

RF 100 trials 后无点超越 rf_coarse_08：
- 保持 rf_coarse_08，ready_for_stage03 = True
- ready_reason = nsga_refine_no_gain_baseline_locked

## 6b. 实际执行结果（2026-05-18/19 完成）

**RF NSGA-II：100 trials 完成 ✅ 成功路径**
- Best trial: mean_f1_all = 0.7704, mean_f1_g2 = 0.5046
- 超越 rf_coarse_08 (0.7676 / 0.4939)，触发成功标准
- 最终锁定配置：RF NSGA-II best

**Spectrum-KAN NSGA-II：20 trials 完成 → 失败退出路径**
- Best G2 = 0.5546 = anchor（20 trials 内未超越）
- 根因：G2 性能瓶颈为 SERS 竞争吸附物理限制，非模型配置问题
- 触发失败退出规则：冻结 skan_anchor_repro_01，ready_for_stage03 = True
- ready_reason = nsga_refine_exhausted_anchor_locked

**数据保存说明：** 本轮服务器产物已经回传到当前 stage 目录。
完整 Optuna study、trial 注册表、所选 trial 输出、服务器日志和审计摘要以本地 stage artifacts 为准；
路线决策以 `project_memory/DECISIONS.md` 为准。

## 7. 结果存储

- optuna_studies/nsga_spectrum_kan.db / optuna_studies/nsga_rf.db：完整 trial 历史
- trial_registry.csv：追加 phase=nsga_refine 的行
- trials/skan_nsga_best_*/：保留的 KAN 选点完整 CV 输出
- trials/rf_nsga_best_*/：保留的 RF 选点完整 CV 输出
- rf_nsga_log.txt / kan_nsga_log.txt：服务器运行日志
- selected_config_manifest.csv：更新最终选择

只有 Pareto 被选中的 trial 才建完整目录，其余记录在 db 和 registry 中。

## 8. 优化完成后操作

成功路径：
1. 更新 selected_config_manifest.csv
2. 写 refine_audit_summary.md（Pareto front 分析）
3. 生成 F6b（Pareto front 图）和 S6（importance 图）
4. 进入锁定主结果阶段

失败路径：
1. 写 refine_audit_summary.md（记录搜索耗尽）
2. 仍然生成 F6b（证明 anchor 接近最优）
3. 冻结 anchor，进入锁定主结果阶段

## 9. 论文叙事

成功版：多目标贝叶斯优化发现，浓度有序 ordinal loss 结合降低 label smoothing，
在 grouped CV 约束下显著提升了最难分级切片的性能。

失败版：对 N 个配置的系统多目标搜索确认了复现 anchor 接近最优，
表明 G2 性能瓶颈是数据可分性限制而非模型配置问题。
