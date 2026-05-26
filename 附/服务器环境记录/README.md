# 服务器环境记录

本目录记录本项目在不同租用服务器上的运行环境，避免后续复现实验时把硬件、Python、PyTorch、CUDA 或包版本混在一起。

## 记录边界

- 每台实际用于实验或 smoke/profile 的服务器单独建一份记录。
- 服务器记录只保存复现信息，不保存账号、密码、token、SSH 私钥或平台支付信息。
- 论文主结果仍以 `core subject/data/pure63_mainline/models/paper_main/` 下回传并校验后的 stage artifacts 为准。
- 每次远程实验完成后，服务器记录应能指向本地代码 commit、实验 stage、回传目录和环境快照。

## 建议记录内容

每份服务器记录至少写明：

1. 日期、服务器别名、平台和用途。
2. GPU、CPU、内存、磁盘、系统镜像。
3. Python、PyTorch、CUDA、cuDNN 如可得、XGBoost 及关键包版本。
4. 本地 Git commit、远程 checkout commit 和运行命令范围。
5. 为该服务器做过的工程调整，例如分批跑法、DataLoader 调整、XGBoost profile、线程限制。
6. 输出回传位置、日志位置、环境快照位置和是否完成本地校验。

## 与项目记忆的关系

- 服务器回传硬约束在 `core subject/project_memory/project_architecture.md`。
- 路线决策在 `core subject/project_memory/DECISIONS.md`。
- 实验状态在 `core subject/project_memory/EXPERIMENT_TRACKER.md`。

本目录用于复现实验环境，不替代以上事实源。
