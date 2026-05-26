# Skills 索引

更新时间：2026-05-22

本目录用于存放项目可参考的外部 skills。这里的内容不等于已经被 Codex/Claude 自动启用；真正启用前要检查 `SKILL.md`、适用范围和是否与本项目记忆规则冲突。

## 已修复的本地 skill 仓库

| 分类 | 本地路径 | 远端 | 当前用途 |
| --- | --- | --- | --- |
| 论文写作类 | `论文写作类/academic-research-skills/` | `https://github.com/Imbad0202/academic-research-skills.git` | 论文写作、审稿模拟、学术 pipeline、deep research |
| 论文写作类 | `论文写作类/deep-ai-research/` | `https://github.com/dair-ai/deep-ai-research.git` | AI research 资料参考；不作为当前项目主 skill |
| 实验设计与规划类 | `实验设计与规划类/AI-research-SKILLs/` | `https://github.com/Orchestra-Research/AI-research-SKILLs.git` | 大型 AI/ML research skills 库；仅筛选使用 |
| 实验设计与规划类 | `实验设计与规划类/Auto-claude-code-research-in-sleep/` | `https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git` | 实验计划、论文写作、claim audit、rebuttal 等候选 skills |

四个仓库的工作区已恢复为 clean。当前能检索到 267 个 `SKILL.md`。

## 当前项目推荐优先使用的本地 skills

### 论文写作与审稿

- `academic-research-skills/academic-paper`
- `academic-research-skills/academic-paper-reviewer`
- `academic-research-skills/academic-pipeline`
- `academic-research-skills/deep-research`
- `Auto-claude-code-research-in-sleep/skills/paper-plan`
- `Auto-claude-code-research-in-sleep/skills/paper-write`
- `Auto-claude-code-research-in-sleep/skills/paper-claim-audit`
- `Auto-claude-code-research-in-sleep/skills/rebuttal`
- `Auto-claude-code-research-in-sleep/skills/result-to-claim`

### 实验设计与结果审计

- `Auto-claude-code-research-in-sleep/skills/experiment-plan`
- `Auto-claude-code-research-in-sleep/skills/experiment-audit`
- `Auto-claude-code-research-in-sleep/skills/ablation-planner`
- `Auto-claude-code-research-in-sleep/skills/analyze-results`
- `Auto-claude-code-research-in-sleep/skills/monitor-experiment`
- `AI-research-SKILLs/20-ml-paper-writing/academic-plotting`
- `AI-research-SKILLs/20-ml-paper-writing/ml-paper-writing`

### 文献检索与引用

- `Auto-claude-code-research-in-sleep/skills/research-lit`
- `Auto-claude-code-research-in-sleep/skills/comm-lit-review`
- `Auto-claude-code-research-in-sleep/skills/citation-audit`
- `Auto-claude-code-research-in-sleep/skills/openalex`
- `Auto-claude-code-research-in-sleep/skills/semantic-scholar`

## 暂时不建议启用的类型

- 大模型微调、分布式训练、LLM 推理部署、RAG 数据库、agent 框架等 skills，和当前 SERS+ML 论文主线关系弱。
- 专利、海报、幻灯片、会议演讲类 skills，等论文结果稳定后再考虑。
- 自动循环类 skills，例如 `auto-review-loop`、`auto-paper-improvement-loop`，除非先明确输入/输出边界，否则容易覆盖项目已有判断。

## 已安装到 Codex 用户 skill 目录的官方 skills

安装位置：`D:\CodexMigrated\.codex\skills\`

- `pdf`：用于读取、检查和处理 PDF 文献或论文。
- `jupyter-notebook`：用于创建或整理 notebook，适合 EDA、结果检查、图表验证。

注意：安装后通常需要重启 Codex 才会在可用 skills 列表中出现。

## 使用规则

1. 项目方向仍以 `core subject/project_memory/AGENTS.md` 和 `DECISIONS.md` 为准。
2. 外部 skill 只能辅助写作、审稿、实验审计，不能覆盖 frozen decisions。
3. 使用任何外部 skill 前，先读对应 `SKILL.md` 的前半部分，确认不会触发不必要的自动循环或外部服务。
4. 如果某个 skill 会生成新论文 claim，必须同步更新 `core subject/project_memory/CLAIM_EVIDENCE_MATRIX.md`。
