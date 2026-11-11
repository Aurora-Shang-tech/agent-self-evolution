
# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution via Counterfactual Verification, Memory Evolution, Hybrid Retrieval, Utility Learning and State Tracking**

基于环境反馈的大语言模型智能体自进化研究项目。

本项目以 **ALFWorld / TextWorld** 为交互环境，在**不更新 LLM 参数**的前提下，研究如何从失败轨迹中获得可复用经验、动态维护记忆、检索有用经验并改进 Agent 的任务执行。

> **项目定位**：以端到端 Success Rate、Average Decision Steps、Improved / Regressed 为最终优化指标。Lesson 是帮助决策的经验，而不是需要逐条证明绝对正确的规则。所有方法的增益均以实际环境评估为准。
>
> **历史阶段（2026-11-10）**：Counterfactual Verification、Memory Evolution、Hybrid BM25、Utility-Aware Retrieval、Contextual Utility 原型、稳定 Memory ID、Memory Pipeline 和 State Tracking 均已实现或完成本地实验。最新集成实验 Hybrid + State 与 Utility + State 均为 11/11，Utility 平均多用 2.73 步；不应将工程集成成功解读为稳定的性能提升。

## 目录

- [1. 研究目标与系统架构](#1-研究目标与系统架构)
- [2. 核心模块](#2-核心模块)
- [3. 环境与实验设置](#3-环境与实验设置)
- [4. 反事实经验学习](#4-反事实经验学习)
- [5. Memory Scaling 与 State Tracking](#5-memory-scaling-与-state-tracking)
- [6. Memory Evolution 全量实验](#6-memory-evolution-全量实验)
- [7. 检索与效用学习实验](#7-检索与效用学习实验)
- [8. 统一 Memory Pipeline](#8-统一-memory-pipeline)
- [9. 快速开始与复现命令](#9-快速开始与复现命令)
- [10. 项目结构与研究结论](#10-项目结构与研究结论)
- [11. 连续自进化闭环](#11-连续自进化闭环2026-11-11)

## 1. 研究目标与系统架构

### 1.1 研究问题

- **Experience Generation**：从失败轨迹、关键决策与环境反事实续跑中提炼可复用 Lesson。
- **Memory Evolution**：通过 `ADD / KEEP / EDIT / REMOVE` 等策略维护长期经验库。
- **Memory Retrieval**：由 BM25 扩展到 Source Task + Lesson 双路 Hybrid BM25，并探索任务感知过滤。
- **Utility Learning**：用 ALFWorld 成功/失败反馈估计经验的历史使用价值，进行 Top-K 重排序。
- **State-Aware Execution**：跟踪 clean / cool / heat 操作和目标物体实例，减少状态错误。
- **Empirical Evaluation**：以配对任务的成功率、决策步数和负迁移检验实际效果。

### 1.2 整体流程

```text
ALFWorld Train Tasks
        |
        v
ReAct -> Trajectories -> Failure Analysis -> Critical Decisions
                                            |
                                            v
                                Counterfactual Replay
                                            |
                                            v
                              Verified Continuations
                                            |
                                            v
                                Candidate Lessons
                                            |
                                            v
                    Memory Evolution: ADD / EDIT / REMOVE / KEEP
                                            |
                                            v
                                Experience Catalog
                              (stable Memory IDs)
                                            |
                      +---------------------+------------------+
                      |                                        |
                      v                                        v
                Hybrid BM25                           Utility Statistics
             (source + lesson)                      (Q, uses, feedback)
                      |                                        |
                      +---------------------+------------------+
                                            |
                                            v
                          Candidate Recall -> Utility Rerank
                                            |
                                            v
                               Top-3 Lessons -> ReAct
                                             + State Tracking
                                            |
                                            v
                               ALFWorld Environment
                                            |
                           Success / Failure / Decision Steps
                                            |
                                            v
                             Feedback stored by Memory ID
                             (affects future retrieval only)
```

**注意**：当前闭环的 Memory Evolution 依赖已有反事实候选数据；Utility Feedback 会更新 Q 值，**不会自动触发新一轮 LLM 候选经验生成或调用 Updater**。生成/更新记忆和在线效用学习是已连接、但调度上独立的阶段。当前 Q 值是对已使用经验的**弱反馈估计**，不是严格的因果贡献。

## 2. 核心模块

### 2.1 ReAct、失败分析和反事实验证

`ReActAgent` 根据任务、当前 Observation、完整交互历史和 `admissible_actions` 逐步执行 ALFWorld 任务。

`Trajectory` 记录 `observation`、`admissible_actions`、`thought`、`action`、`valid_format`、`admissible`、`next_observation`、`score`、`done`、`won`，以及整个 Episode 的 `success` 和 `truncated`。

`FailureAnalyzer` 识别失败类型、关键决策和替代动作候选；Counterfactual Verifier 重放关键步骤之前的环境，尝试替代动作并继续执行。

**验证依据为真实环境的 `won` 信号，而不是单纯依赖 LLM 对经验的判断。**

### 2.2 长期经验与 Memory Evolution

原始 `ExperienceStore` 中的经验主要包含：

```json
{
  "failure_type": "missing_state_transition",
  "lesson": "Ensure the required object state is confirmed before final placement.",
  "source_task": "put a cool potato in microwave"
}
```

在 Memory Evolution 快照中，经验使用稳定 ID，例如：

```json
{
  "id": "evo_episode_138_c001",
  "lesson": "When a task requires placing a clean object, prioritize cleaning the object before returning it to its destination to avoid redundant retrieval and movement cycles.",
  "source_task": "put a clean spatula in countertop."
}
```

上面的 ID 和 Lesson 文本来自本地实际记录；`source_task` 在示例中仅说明数据结构，具体来源请以 `working_memory.json` 为准。

| 更新策略 | 操作 | 含义 |
|---|---|---|
| **Direct Update** | ADD / KEEP / EDIT / REMOVE | 比较候选和现有记忆，动态增删改 |
| Rule Strength Voting | ADD / AGREE | 匹配已有规则并累积支持 |
| Source-Aware Revision | CREATE / CORROBORATE / REFINE | 引入来源信息，持续修订和细化 |

这三套方法已使用同一候选池完成对照；后续的统一 Pipeline 当前以 **Direct Update 18 条经验**为主要集成对象，并未删除另外两套实验实现。

### 2.3 Retrieval：BM25、Hybrid、Task-Aware

- **BM25**：基于 `source_task` 的词项匹配，检索相关历史任务。
- **Hybrid BM25**：同时计算任务描述与 `source_task`、`lesson` 的相关性，先召回候选，再选择 Top-3；当前原型先在两个通道内归一化、取平均。
- **Task-Aware**：依据任务条件限制或调整候选，已完成探索性实验，当前不作为统一 Pipeline 默认检索器。

Hybrid 的候选排序不是 Utility 的最终排序。

**先相关性召回 Top-8，再由效用分数调整 Top-3 的优先级。**

这样不会在完全不相关的全库经验中只按 Q 值选出“历史成功次数最高”的经验。

### 2.4 Utility-Aware Retrieval

`UtilityTracker` 为每条经验维护 `q`、`uses`、`successes`、`failures`。

```text
reward = +1  if ALFWorld task succeeded
reward = -1  otherwise

Q_new = (1 - alpha) * Q_old + alpha * reward
```

现有实验配置使用：

| 参数 | 值 |
|---|---:|
| alpha | 0.3 |
| weight | 0.35 |
| pool_size | 8 |
| top_k | 3 |

`HybridUtilityStore` 将 Hybrid 相关性与历史效用加分相结合。

效用值根据经验在 Episode 中是否被检索到及最终任务结果更新，属于**共享奖励 / 弱信用分配**；不能把一次成功直接归因给本次使用的每条 Lesson。

`ContextualCredit` / `ContextualHybridStore` 是任务类型条件效用的探索性实现：仅从配对差异中提取较保守的反馈，按 `task_type` 存储。

当前数据量较少，该原型尚未展现可观察的排序变化。

### 2.5 State Tracking

`TaskStateTracker` 跟踪 clean / cool / heat，区分目标对象和实例编号，并用环境反馈确认状态变化；Agent 可在每步后获得状态提醒。

它不会读取 ALFWorld 隐藏目标状态，也不能解决所有物体搜索、目标选择或行动规划错误。

## 3. 环境与实验设置

| 配置 | 值 |
|---|---|
| Environment | ALFWorld / TextWorld |
| LLM | `qwen3.8-chat` |
| Python | 3.11；使用 `uv` |
| 经验构建 Train | 1,000 Episodes |
| Valid Seen | 140 Episodes |
| Valid Unseen | 134 Episodes |
| 原始数据随机种子 | 42 |
| Max Decisions | 50 |
| Retrieval Top-K | 3 |
| 参数微调 | 无 |

实验使用固定 Manifest 与逐 Episode 轨迹。

主要指标：

- **Success Rate**：任务完成率。
- **Average Decision Steps**：平均决策步数。
- **Improved**：对照失败、新方法成功的任务数。
- **Regressed**：对照成功、新方法失败的任务数。
- **Net**：Improved − Regressed。
- **Exact McNemar p**：配对成功/失败状态的统计检验。

不同实验可能采用不同脚本版本、状态跟踪设置或任务子集，故不能把不同批次中的微小差异直接归因于某个单一模块。

尤其应区分完整 Seen/Unseen 测试与仅用于算法筛选的少量 Train 留出任务。

## 4. 反事实经验学习

### 4.1 Train 阶段

在 1,000 个 Train 任务中，ReAct 完成 867 个，失败 133 个。

| 指标 | 结果 |
|---|---:|
| Train 成功 | 867/1000 (86.70%) |
| 平均决策步数 | 20.26 |
| 失败 Episodes | 133 |
| 反事实候选分支 | 371 |
| 成功的反事实续跑 | 197 |
| 反事实候选成功率 | 53.10% |
| 有成功反事实的失败任务 | 91/133 |

经验库去重后规模：

| 方法 | 经验数量 |
|---|---:|
| Ordinary Memory | 132 |
| Unverified Memory | 132 |
| CF-Verified Memory | 130 |

### 4.2 原始方法对照

| 方法 | Seen 成功 | Seen 平均步数 | Unseen 成功 | Unseen 平均步数 |
|---|---:|---:|---:|---:|
| ReAct | 120/140 (85.71%) | 19.97 | 122/134 (91.04%) | 18.24 |
| Ordinary Memory | 122/140 (87.14%) | 18.59 | 127/134 (94.78%) | 15.32 |
| Unverified Memory | 127/140 (90.71%) | 17.01 | 129/134 (96.27%) | 14.55 |
| **CF-Verified Memory** | **128/140 (91.43%)** | **15.54** | **131/134 (97.76%)** | **13.22** |
| Reflexion | 133/140 (95.00%) | 25.19 | 130/134 (97.01%) | 21.78 |

相较 ReAct，CF-Verified Memory 在这组历史实验中提升了成功率并降低平均决策步数。

但相较 Unverified 的额外收益较小。

Reflexion 最多允许两次尝试，因此不属于严格等预算比较。

## 5. Memory Scaling 与 State Tracking

### 5.1 Memory Scaling

基于原始 130 条 CF-Verified Memory 按固定随机顺序构造嵌套子集。

| 经验规模 | 条数 | Seen 成功 | Seen 平均步数 | Unseen 成功 | Unseen 平均步数 |
|---|---:|---:|---:|---:|---:|
| M0 | 0 | 120/140 | 19.97 | 122/134 | 18.24 |
| M25 | 32 | 127/140 | 16.15 | 125/134 | 15.80 |
| M50 | 65 | **132/140** | **14.72** | 128/134 | 15.05 |
| M75 | 98 | 130/140 | 16.00 | 127/134 | 13.87 |
| M100 | 130 | 128/140 | 15.54 | **131/134** | **13.22** |

**结论：** 本次固定抽样中，记忆数量与性能不呈单调关系。

Seen 的最高成功数出现在 M50，Unseen 出现在 M100，不能据此宣称某个规模普遍最优。

### 5.2 Memory × State Tracking（历史 2×2 实验）

| 方法 | Seen 成功 | Seen 平均步数 | Unseen 成功 | Unseen 平均步数 |
|---|---:|---:|---:|---:|
| ReAct | 120/140 | 19.97 | 122/134 | 18.24 |
| ReAct + State | 128/140 | 17.43 | 126/134 | 15.62 |
| CF-Verified | 128/140 | 15.54 | 131/134 | 13.22 |
| **CF-Verified + State** | **133/140** | **13.79** | **134/134** | **12.11** |

在这组历史实验中，CF-Verified + State 的 Seen 为 95%，Unseen 为 100%。

但 100% 为固定任务集上的观测，不等于任意新任务都能成功。

| 配对比较 | Split | Improved | Regressed | 双侧 Exact McNemar p |
|---|---|---:|---:|---:|
| Verified → Verified + State | Seen | 8 | 3 | 0.2266 |
| Verified → Verified + State | Unseen | 3 | 0 | 0.2500 |
| ReAct + State → Verified + State | Seen | 7 | 2 | 0.1797 |
| ReAct + State → Verified + State | Unseen | 8 | 0 | 0.0078 |

多项比较未做多重检验校正，需谨慎解释显著性。

## 6. Memory Evolution 全量实验

### 6.1 全量记忆更新

Train 133 个失败 Episode、371 条反事实分支，经筛选进入记忆更新对照。

| Direct Update 过程 | 数量 |
|---|---:|
| SKIPPED | 177 |
| ABSTAIN | 21 |
| KEEP | 151 |
| PROVISIONAL | 22 |
| **合计** | **371** |

其中 173 条候选 Lesson 进入统一更新策略对照。

| 更新方法 | 操作统计 | 最终规模 |
|---|---|---:|
| Direct Update | ADD 19、KEEP 151、EDIT 2、REMOVE 1 | **18** |
| Rule Strength Voting | ADD 21、AGREE 152 | **21** |
| Source-Aware Revision | CREATE 22、CORROBORATE 137、REFINE 14 | **22** |

### 6.2 全量下游对照

以下是完整 **16 组 Memory Evolution × State Tracking** 实验。

表内表示 `成功数 / 总数 (平均步数)`。

| Memory | Seen / 无 State | Seen + State | Unseen / 无 State | Unseen + State |
|---|---|---|---|---|
| **Original CF-Verified (130)** | 128/140 (15.14) | **133/140 (13.75)** | **130/134 (12.80)** | **134/134 (11.93)** |
| Direct Update (18) | 126/140 (16.48) | 133/140 (14.35) | 130/134 (13.65) | 128/134 (13.92) |
| Rule Strength Voting (21) | **129/140 (15.51)** | 133/140 (14.40) | 130/134 (13.84) | 130/134 (13.49) |
| Source-Aware Revision (22) | 126/140 (16.36) | 133/140 (14.46) | 129/134 (14.46) | 133/134 (13.52) |

第 5 节与本节的部分平均步数和成功数不同，原因是它们来自不同的历史评估批次；此处保留原始报告，不强行合并结果。

| 配对比较 | 设置 | Improved | Regressed | p-value |
|---|---|---:|---:|---:|
| Original → Direct Update | Seen / 无 State | 7 | 9 | 0.8036 |
| Original → Rule Strength Voting | Seen / 无 State | 7 | 6 | 1.0000 |
| Original → Source-Aware Revision | Seen / 无 State | 5 | 7 | 0.7744 |
| Original → Direct Update | Unseen / State | 0 | 6 | 0.0312 |
| Original → Rule Strength Voting | Unseen / State | 0 | 4 | 0.1250 |
| Original → Source-Aware Revision | Unseen / State | 0 | 1 | 1.0000 |
| Direct Update → Source-Aware Revision | Unseen / State | 5 | 0 | 0.0625 |

**结论：** 进化策略将经验从 130 条压缩至 18～22 条，但在这批 Seen/Unseen 全量对照中，尚无证据证明它们稳定优于原始 CF-Verified 130 条记忆。

压缩与可维护性不自动等于更高的 Agent 成功率。

## 7. 检索与效用学习实验

本节为 **2026-11-10 新增**。

下面的 Utility / Contextual / Pipeline 实验使用额外选取的 Train 留出任务或复用配对轨迹，**不是**对第 6 节 Valid Seen/Unseen 的重新评估。小样本差异为探索性结果。

### 7.1 Hybrid 与 Task-Aware

Hybrid BM25 结合 `source_task` 和 Lesson 文本相关性。

已有探索性评估记录显示：

- Direct Update 18 的 Hybrid 在 `valid_seen` 的一次历史评估为 **131/140，平均 14.91 步**。
- Task-Aware v2 在对应探索性设置下也为 **131/140，平均 15.44 步**，配对有 3 个改善、3 个退步。

上述指标属于已有独立实验记录，不意味着新实现统一 Pipeline 的默认配置获得了同样的分数。

当前 Task-Aware 保留为消融选项，默认集成仍采用 Hybrid。

### 7.2 单条经验效用探查（Lesson 08）

选定清洗类 Train 留出任务，比较**无经验**与**只注入 Lesson 08**，每组 6 个 Episode：

| 任务 | Control | With Lesson 08 |
|---|---|---|
| Spatula | 21 步成功 | 21 步成功 |
| Cloth | 50 步失败 | 17 步成功 |
| Mug | 13 步成功 | 13 步成功 |
| Kettle | 24 步成功 | 19 步成功 |
| Bowl | 27 步成功 | 27 步成功 |
| Pan | 49 步成功 | 50 步成功 |

结果为：

- 成功数：**5/6 → 6/6**
- 平均步数差：**-6.17**
- Improved / Regressed：**+1 / -0**

这一小样本仅用于探索经验注入的效用，不代表 Lesson 08 对所有 Clean 任务有效，也不代表实际 Top-3 检索中的边际贡献。

### 7.3 Utility v1：在线 Q 学习 + Hybrid 重排序（无 State）

使用 12 个未用于原始 1,000 个经验构建 Episode 的 Train 游戏学习 Q 值，另 12 个不同游戏做 Hybrid / Utility 配对。

评估阶段冻结 Q。

| 指标 | Hybrid | Utility v1 |
|---|---:|---:|
| 成功 | **10/12** | **11/12** |
| 成功率 | 83.33% | 91.67% |
| 平均决策步数 | 18.83 | 17.25 |

- 检索排序变化：**12/12**
- Improved / Regressed：**+1 / -0**
- 净成功数：**+1**
- 平均步数变化：**-1.58**
- 双侧 Exact McNemar：**p=1.0**

v1 学习阶段：12 个 Episode，**11/12 成功**。

由于采用共享奖励机制，Q 值可能偏向经常出现在容易成功任务中的 Lesson。

**解释**：首次得到正向探索性结果，但样本过小且差异未显著，不能宣称 Utility 在总体任务分布上稳定胜过 Hybrid。

### 7.4 Contextual Utility v2：条件信用（离线原型）

复用 v1 的 12 对历史评估轨迹，从经验集合差异中提取弱配对反馈，不再调用 LLM 学习。

实验结果：

- 产生非零信用的配对：**2/12**。
- `look_at_obj_in_light`：Lesson 03 的条件信用 **+0.303**，Lesson 10 **-0.303**。
- 在另 12 个 Train 游戏的预检查中：**0/12 Top-3 排序变化**。
- 因检索无变化，脚本跳过新 LLM 对照，**没有报告 Contextual v2 的下游增益**。

这说明现有证据信号不足以驱动该批任务的重排序，不能由此得出条件信用策略普遍无效的结论。

Contextual v2 保留为原型，不替换已可运行的 Utility v1。

### 7.5 统一 Pipeline + State：Hybrid vs Utility

将 Direct Update 18、稳定 ID、Hybrid + Q、ReAct + State Tracking 集成。

Q 值导入 v1 已学到的 `frozen_utility.json`，随后完成了一次独立 Train Episode 的反馈更新：

- Agent 执行：**9 步成功**
- Utility Feedback：**Feedback applied: True**

在后续评估中排除该用于学习的 Episode 0，对剩余 11 个 Train 游戏运行两组配对。

两组均启用 State Tracking，评估期间冻结 Q 值和经验快照。

| 指标 | Hybrid + State | Utility + State |
|---|---:|---:|
| 完成任务 | **11/11** | **11/11** |
| 成功率 | 100% | 100% |
| Top-3 排序变化 | — | **11/11** |
| Improved / Regressed | — | **0 / 0** |
| 平均步数变化 | — | **+2.73 步/任务** |
| 总步数变化 | — | **+30 步** |
| Exact McNemar p | — | 1.0 |

**结论**：本批次任务上，Utility 确实改变了检索排序，但**没有带来额外成功，反而增加平均决策步数**。

Hybrid 已完成 11/11，成功率存在天花板效应。

结果说明端到端集成可运行，不足以证明加上 State Tracking 后 Utility 仍有正向净收益。

### 7.6 当前证据汇总

| 方法或阶段 | 已观察结果 | 研究定位 |
|---|---|---|
| Original 130 + State | 历史 Seen 133/140；Unseen 134/134 | 目前最强的完整 Seen/Unseen 基线之一 |
| Memory Evolution | 18 / 21 / 22 条 | 动态更新能力已实现，尚未稳定超过 Original 130 |
| Hybrid BM25 | 已完成全量/探索性评估 | 当前检索核心基线 |
| Utility v1（无 State） | 10/12 → 11/12；步数 -1.58 | 初步正向、样本有限 |
| Contextual v2 | 0/12 检索变化 | 可保留的研究原型，未产生新下游证据 |
| Integrated + State | 11/11 → 11/11；步数 +2.73 | 工程闭环完成，当前无额外性能增益 |

**研究优先级**：保留已实现的更新、检索和效用学习模块；以成功率、平均步数和负迁移决定后续是否采用某种可选机制。

无需把每条 Lesson 修改到“完美”后才能推进下游实验。

## 8. 统一 Memory Pipeline

### 8.1 稳定 ID 与经验更新同步

`src/memory/pipeline.py` 将 Memory Evolution 快照转换为稳定 ID Catalog，与 `UtilityTracker` 和 `HybridUtilityStore` 连接。

| 快照操作 | 稳定 ID / Q 的处理 |
|---|---|
| ADD | 新 ID，新 Q 统计从 0 开始 |
| EDIT | ID 保留，Lesson 内容变化时重置旧 Q |
| REMOVE | 从当前索引移除，旧统计归档 |
| KEEP | 保留已有统计 |
| 列表顺序变化 | 通过 Memory ID 匹配，避免 Q 错位 |

`record_feedback()` 需要显式传入：

- `event_id`
- 任务描述
- 成功或失败状态
- 决策步数
- 本次检索的 Memory ID 列表

同一事件重复提交不会重复更新 Q。

该实现适合单进程顺序执行，当前没有提供跨进程并发写保护。

### 8.2 当前实现位置

| 文件 | 作用 |
|---|---|
| `src/memory/utility.py` | `UtilityTracker`，维护 Q / uses / success / failure |
| `src/memory/utility_store.py` | 双路 Hybrid BM25 召回与 Utility 重排序 |
| `src/memory/contextual_utility.py` | 按任务类型的条件信用统计（实验原型） |
| `src/memory/contextual_store.py` | Contextual Utility 重排序（实验原型） |
| `src/memory/pipeline.py` | 稳定 ID、旧 Q 导入、快照同步与反馈持久化 |
| `scripts/memory/manage_pipeline.py` | `sync`、`inspect`、单 Episode `run`，可选 `--learn` |
| `scripts/evaluation/verify_memory_utility.py` | 单条 Lesson 的配对探索 |
| `scripts/evaluation/evaluate_utility_memory.py` | Utility v1 的反馈学习与对照 |
| `scripts/evaluation/evaluate_contextual_utility.py` | Contextual v2 离线反馈与实验 |
| `scripts/evaluation/evaluate_pipeline_pair.py` | 冻结 Q 后比较 Hybrid + State 与 Utility + State |

**在将本 README 提交到 GitHub 前，应确保对应本地 `.py` 文件也已暂存并提交**，否则文档会领先远程代码。

### 8.3 实验产物

```text
outputs/memory_evolution/train_full_v1/working_memory.json
outputs/memory_utility/holdout_train_v1/
outputs/memory_utility/feedback_v1/qwen3.8-chat/seed_20261110/
outputs/memory_utility/contextual_v2/qwen3.8-chat/seed_20261112/
outputs/memory_pipeline/state.json
outputs/memory_pipeline/executions/
outputs/memory_pipeline/paired_state_v1/qwen3.8-chat/v2_seed_20261112/
```

`outputs/` 被 `.gitignore` 排除。

这些路径为**本地实验产物位置**，并不表示其内容已经公开上传到 GitHub。

## 9. 快速开始与复现命令

### 9.1 环境配置

项目采用 Python 3.11 和 uv 管理依赖。

```bash
uv sync
uv run alfworld-download
```

使用 OpenAI-compatible 模型服务，需配置：

```text
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL=<your-compatible-api-base-url>
OPENAI_MODEL=qwen3.8-chat
```

真实模型运行前确认环境变量正确，不要将 `.env` 或密钥提交到 GitHub。

### 9.2 Memory Evolution

基于 Train 数据提取候选经验并完成 Direct Update：

```bash
uv run python -m scripts.memory.run_evolution \
  --run-name train_full_v1 \
  --top-k 8
```

对相同候选池运行其他 Memory Evolution 方法：

```bash
uv run python -m scripts.memory.compare_updates \
  --source-run train_full_v1 \
  --output-run train_full_comparison \
  --limit 371 \
  --top-k 8
```

这两条命令可能涉及大量数据处理和 LLM 调用。

已有输出完整时**无需重新执行**。

### 9.3 全量 Memory Evolution × State 对照

Valid Seen：

```bash
uv run python -m scripts.evaluation.evaluate_memory_evolution \
  --split valid_seen \
  --state both
```

Valid Unseen：

```bash
uv run python -m scripts.evaluation.evaluate_memory_evolution \
  --split valid_unseen \
  --state both
```

这些是历史全量评估命令，不是完成 README 更新所必需的步骤。

### 9.4 Utility v1 与 Contextual v2

冻结 12 个学习任务和 12 个评估任务：

```bash
uv run python -m scripts.evaluation.evaluate_utility_memory \
  --stage plan
```

查看已经完成的 Utility v1 实验：

```bash
uv run python -m scripts.evaluation.evaluate_utility_memory \
  --stage report
```

Contextual v2 离线计算信用：

```bash
uv run python -m scripts.evaluation.evaluate_contextual_utility \
  --stage prepare
```

检查条件效用是否改变检索：

```bash
uv run python -m scripts.evaluation.evaluate_contextual_utility \
  --stage inspect
```

以上命令不会新增 LLM 评估调用。

### 9.5 统一 Pipeline：导入旧 Q、同步、查看检索

首次在**尚未创建** `outputs/memory_pipeline/state.json` 时，可以从 v1 冻结快照迁移：

```bash
uv run python -m scripts.memory.manage_pipeline \
  --stage sync \
  --import-v1
```

**已有 `state.json` 时不要再次使用 `--import-v1`**，否则会拒绝导入。

正常同步：

```bash
uv run python -m scripts.memory.manage_pipeline \
  --stage sync
```

查看 Hybrid 与 Utility Top-3：

```bash
uv run python -m scripts.memory.manage_pipeline \
  --stage inspect \
  --task "put a clean ladle in countertop."
```

本地运行已成功导入 18 条 Direct Update 的 Q 值，并确认稳定 ID 对齐。

### 9.6 单 Episode 在线反馈

`manage_pipeline.py --stage run` 使用包含 `{"games": [...]}` 的 Manifest。

新学习任务须属于 Train，并且不在记忆构建的原始 1,000 个游戏中。

```bash
uv run python -m scripts.memory.manage_pipeline \
  --stage run \
  --manifest outputs/memory_utility/contextual_v2/qwen3.8-chat/seed_20261112/experiment_plan.json \
  --episode 0 \
  --run-name integrated_train \
  --state-tracking \
  --learn
```

上面这条是**已完成实验的复现示例**。

当前已有相同 `run-name` 和 Episode 时，脚本复用缓存，并通过 `event_id` 防止重复 Q 更新。

不需要为了文档更新而再次运行。

**用过的任务不能继续作为新的独立评估样本。**

### 9.7 批量对照（冻结 Q）

先扫描任务并比较两组 Top-3：

```bash
uv run python -m scripts.evaluation.evaluate_pipeline_pair \
  --stage plan
```

查看已经完成的配对统计：

```bash
uv run python -m scripts.evaluation.evaluate_pipeline_pair \
  --stage report
```

如确实需要重新开展实验：

```bash
uv run python -m scripts.evaluation.evaluate_pipeline_pair \
  --stage eval \
  --max-new 11
```

本地最新结果已完成 11 对。

评估过程中必须冻结经验快照、代码和 Q 状态，**不要同时执行 `--learn`**。

脚本会对输入文件的哈希进行检查，避免混入不同版本。

## 10. 项目结构与研究结论

### 10.1 主要目录

```text
agent-self-evolution/
├── src/
│   ├── agents/                  # ReAct / Reflexion
│   ├── analysis/                # Failure analysis
│   ├── core/                    # Prompt protocol / Trajectory
│   ├── counterfactual/          # Replay / Verification
│   ├── data/                    # Game discovery / Manifest
│   ├── envs/                    # ALFWorld wrapper
│   ├── llm/                     # LLM API client
│   └── memory/
│       ├── experience.py
│       ├── store.py
│       ├── updater.py
│       ├── state_tracker.py
│       ├── utility.py
│       ├── utility_store.py
│       ├── contextual_utility.py
│       ├── contextual_store.py
│       └── pipeline.py
├── scripts/
│   ├── analysis/
│   ├── evaluation/
│   │   ├── evaluate_memory_evolution.py
│   │   ├── verify_memory_utility.py
│   │   ├── evaluate_utility_memory.py
│   │   ├── evaluate_contextual_utility.py
│   │   └── evaluate_pipeline_pair.py
│   └── memory/
│       ├── run_evolution.py
│       ├── compare_updates.py
│       └── manage_pipeline.py
├── docs/
│   └── experiment_results.md
├── outputs/                    # .gitignore 忽略，本地实验产物
├── pyproject.toml
└── README.md
```

### 10.2 目前可以确认的结论

**1. Training-Free Experience Memory 具有下游价值**

原始 CF-Verified Memory 与 State Tracking 在历史完整 Seen/Unseen 任务上取得较高成功率。

**2. 动态记忆更新已实现，但压缩不等于性能提高**

三套 Memory Evolution 方法将记忆库压缩至 18～22 条，目前尚未稳定超过原始 130 条的完整基线。

**3. Hybrid Retrieval 与 Utility Feedback 已集成**

完成稳定 Memory ID、Q 值持久化、检索记录和 ALFWorld 反馈回写。

**4. Utility 的收益具有设置依赖性**

无 State 的小样本实验为 +1 成功、-1.58 步；有 State 的小样本实验成功数不变且 +2.73 步。

**5. 下一步应以系统性能为主导**

保留所有完成的研究模块，以新任务成功率、平均步数和负迁移判断是否启用或继续改进可选机制，而不是要求 Lesson 完美。

### 10.3 结果与复现的边界

- 所有“成功率 100%”均针对固定任务集，不是普遍保证。
- 训练反馈任务与确认性评估任务必须分离；使用评估轨迹学习新的 Q 后，不得再将同批数据声称为独立测试。
- 使用成功/失败给多条经验共享奖励并非无偏的信用分配；Q 值不是因果效果。
- `outputs/` 默认不提交，GitHub README 仅报告指标与路径，不代表原始轨迹已公开。
- 本 README 汇总了已提供的本地执行日志和仓库历史内容；**文档更新本身没有重新运行任何 ALFWorld 实验**。

---

**研究方向：Training-Free LLM Agent Self-Evolution / Memory Evolution / Hybrid Retrieval / Utility-Aware Feedback / State-Aware Execution。**


## 11. 连续自进化闭环（2026-11-11）

### 11.1 系统实现

新增批次式 `EvolutionController`，将以下模块连接为统一流程：

ReAct 执行 → 失败分析 → 反事实验证 → Lesson 提取 →
ADD/EDIT/REMOVE/KEEP → 候选记忆版本 → 配对验证 → 版本选择。

主要新增与修改：

- `src/evolution/controller.py`：连续调度、候选版本、验证门控和断点恢复。
- `src/evolution/version_store.py`：正式版本加载、历史版本读取与完整性校验。
- `scripts/run_continual_evolution.py`：统一命令行入口。
- `src/core/trajectory.py`：保存完整 LLM 对话上下文。
- `src/agents/react_agent.py`：支持恢复上下文和 State Tracking 后续跑。
- `src/counterfactual/verifier.py`：在原始 Memory + State 上下文中进行反事实验证。

不更新底座模型参数。本实验固定使用 Original CF-Verified
130、Source-task BM25 Top-3、State Tracking 和最大 50 决策步。

### 11.2 连续三轮实验

采用固定 Train 任务划分：72 个学习任务、12 个版本验证任务、
30 个最终留出任务。三轮各使用 24 个不同学习任务。

| 指标 | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| 成功任务 | 24/24 | 23/24 | 23/24 |
| 平均决策步数 | 12.54 | 15.04 | 14.96 |
| 反事实分支 | 0 | 2 | 2 |
| 成功反事实 | 0 | 2 | 0 |
| 记忆处理 | 无失败 | KEEP | NO_CHANGE |
| 正式版本 | v000 | v000 | v000 |

**总体结果：70/72 成功（97.22%），平均 14.18 决策步。**

三轮自动执行、失败处理和无更新时的版本保持已验证。
第二轮产生两条成功反事实分支，但 Updater 决定 KEEP；
第三轮没有成功反事实分支。

因此尚未产生新的正式记忆版本，候选版本配对验证与自动晋升
尚未在真实候选变更下得到验证，也不能声称动态进化已优于固定记忆。

### 11.3 使用方式

查看已完成实验：

```bash
uv run python -m scripts.run_continual_evolution \
  --stage status --run-name dynamic_v1 --mode dynamic
```

加载当前正式记忆：

```python
from src.evolution.version_store import load_active_memory

store, info = load_active_memory(
    "outputs/continual_v1/runs/dynamic_v1"
)
```

本次实验结果保存在
`outputs/continual_v1/runs/dynamic_v1/run_summary.json`。

`outputs/` 默认不提交 GitHub，最终留出任务尚未用于本次评估。
