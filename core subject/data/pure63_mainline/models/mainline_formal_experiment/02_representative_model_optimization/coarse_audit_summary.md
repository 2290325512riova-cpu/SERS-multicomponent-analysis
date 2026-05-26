# 02 阶段 coarse 审计摘要

本文件同时记录两件事：

- RF 这条当前仍保留在现役 02 里的 coarse tuned baseline。
- 2026-05-13 那版现已归档的旧 Spectrum-KAN 8-trial coarse 审计结果。

## 1. 运行状态

- 服务器运行时间：2026-05-13
- 运行配置：RF + Spectrum-KAN 单模型 coarse search
- trial 数：16
- 总耗时：92.3 分钟
- 当前本地已同步：stage-root 的 trial_registry.csv、selected_config_manifest.csv，以及 16 个 trials/<trial_id>/ 目录
- 其中 `skan_coarse_01` 到 `skan_coarse_08` 现已在 active trial_registry 中标记为 archived，并镜像归档到 archive_历史归档/20260513_SpectrumKAN旧coarse锚点误配归档

## 2. 当前 best trial

- RF：rf_coarse_08，preprocess = p4，mean_f1_all = 0.7675537092403149，mean_f1_g2 = 0.4938579034426762
- Spectrum-KAN（legacy archived route）：skan_coarse_06，preprocess = p4，mean_f1_all = 0.6104924650212021，mean_f1_g2 = 0.39113253909379

## 3. 与 01 阶段 apples-to-apples 对比

- RF p4：01 阶段 mean_f1_all = 0.752304335792848，02 最优 = 0.7675537092403149；01 阶段 G2 = 0.49074138573308385，02 最优 = 0.4938579034426762。
- Spectrum-KAN p4：01 阶段 mean_f1_all = 0.676292096178226，02 最优 = 0.6104924650212021；01 阶段 G2 = 0.5546015196818799，02 最优 = 0.39113253909379。
- Spectrum-KAN p1：01 阶段 mean_f1_all = 0.692097715379434，02 p1 best = 0.6038302468519331；01 阶段 G2 = 0.4017997995281288，02 p1 best = 0.3496180018703755。
- Spectrum-KAN p4 anchor_repro：`skan_anchor_repro_01` 已把 02 的 p4 结果逐项复现回 01 阶段，mean_f1_all = 0.6762920961782264，G2 = 0.5546015196818799。
- Spectrum-KAN p1 anchor_repro：`skan_anchor_repro_02` 也已把 02 的 p1 结果逐项复现回 01 阶段，mean_f1_all = 0.6920977153794335，G2 = 0.4017997995281288。

## 4. 当前判断

- RF 属于小幅正优化，可以保留为现役 02 tuned baseline。
- 旧 Spectrum-KAN 路线属于明确负优化，但它当前只说明那版 step/adamw/no-aug 的优化制度不成立，不再作为现役 KAN selection 依据。
- 新的 Spectrum-KAN anchor_repro 已经成功，说明旧 KAN 负结果的主因是训练制度失配，而不是架构层先验失败。
- 因此 RF 不需要在这一步继续 broad optimize；KAN 的现役下一步则从“等待 reproduction”切换到“是否做窄范围 local refine”的决策，而不是直接推进 03。

## 5. 当前唯一推荐下一步

- 只在已复现成功的 `skan_anchor_repro_01` p4 anchor 周围决定是否开一个非常窄的 local refine；如果不开 refine，则应显式把 reproduced anchor 写成 02 的停机结论，而不是重新回到旧 coarse 路线。

## 6. 同步备注

- 服务器端 /hy-tmp/sers_project/data/pure63_mainline/models/mainline_formal_experiment/02_representative_model_optimization/trials/ 已完成回传，本地当前拥有完整的 02 coarse trial 目录镜像。
- stage-root 的 trial_registry.csv 和 selected_config_manifest.csv 仍然是最权威、最适合先读的 02 摘要索引；需要逐 trial 细节时再进入对应 trials/<trial_id>/ 目录。
- 想追溯旧 KAN 8-trial route 时，优先看 archive_历史归档/20260513_SpectrumKAN旧coarse锚点误配归档，而不是继续把它当成 active 02 的现役最优配置。