# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution via Counterfactual Verification, Experience Memory and State Tracking**

基于强化反馈的大语言模型智能体自进化系统。

本项目研究如何在**不更新大语言模型参数**的情况下，通过环境反馈、反事实验证和长期经验记忆，提高 LLM Agent 的任务成功率与决策效率。

项目以 **ALFWorld** 为实验环境，基于 ReAct Agent 实现任务执行、失败分析、关键决策定位、反事实验证、经验提炼、动态记忆更新和状态跟踪，并通过系统化的消融实验评估不同机制对 Agent 行为的影响。

核心研究内容包括：

- **Counterfactual Experience Learning**：通过真实环境反事实验证，从失败轨迹中提炼可复用经验。
- **State-Aware Execution**：显式跟踪任务要求的物体状态，减少状态遗漏和错误推断。
- **Memory Evolution**：通过经验强化、合并、修订和删除，实现长期记忆的动态更新。
- **Empirical Evaluation**：通过成功率、平均决策步数、负迁移和配对统计检验衡量实际效果。

## 1. 核心方法

### 1.1 系统整体流程

```text
                    ALFWorld Task
                          |
                          v
                     ReAct Agent
                          |
                          v
                  Environment Feedback
                          |
                          v
                   Episode Trajectory
                          |
                   Failed Episodes
                          |
                          v
                    Failure Analyzer
                 / Critical Step
                /  Failure Type
               /   Alternative Actions
                          |
                          v
                  Counterfactual Replay
                          |
                          v
                 Verified Rollout Results
                          |
                          v
                  Experience Extraction
                          |
                          v
                    Memory Evolution
                 / Direct Update
                /  Rule Strength Voting
               /   Source-Aware Revision
                          |
                          v
                    Experience Store
                          |
                          v
                     BM25 Retrieval
                       (Top-K=3)
                          |
                          v
                ReAct + State Tracking
                          |
                          v
                    ALFWorld Evaluation
               Success / Steps / Regression
```

系统通过环境反馈积累经验，再将检索到的经验注入后续任务的 Agent Prompt，实现 **Training-Free / Non-Parametric Self-Evolution**。

### 1.2 基于反事实验证的经验学习

#### ReAct 任务执行

Agent 根据当前任务、环境观察和可执行动作生成下一步决策，并与 ALFWorld 环境交互。

系统记录完整执行轨迹，包括：

- `observation`：当前环境观察。
- `admissible_actions`：环境允许的动作。
- `thought`：模型的推理内容。
- `action`：模型选择的动作。
- `next_observation`：执行后的环境反馈。
- `done` / `won`：任务终止与完成状态。

#### 失败分析与关键决策定位

`FailureAnalyzer` 对失败轨迹进行分析，识别失败类型、关键决策步骤和可能的替代动作。

反事实验证器通过环境重放恢复关键决策前的状态，执行替代动作，并让 Agent 从新状态继续运行。

通过真实 ALFWorld 环境的 `won` 信号判断反事实续跑是否完成任务，而不是仅依赖 LLM 对策略有效性的判断。

#### 长期经验提炼与检索

实现三种经验提炼方式：

| 方法 | 核心机制 |
|---|---|
| Ordinary Memory | 直接从完整失败轨迹总结经验 |
| Unverified Memory | 基于失败关键步骤及 LLM 提出的替代动作提炼经验 |
| CF-Verified Memory | 结合反事实环境执行结果及续跑轨迹提炼经验 |

每条经验主要包含以下字段：

```json
{
  "failure_type": "missing_state_transition",
  "lesson": "Ensure the required object state is confirmed before final placement.",
  "source_task": "put a cool potato in microwave"
}
```

`ExperienceStore` 使用 **BM25** 根据 `source_task` 检索相关经验，选择 Top-K Lesson 注入后续任务的 ReAct Prompt。

经验存储和检索不需要修改底层 LLM 参数。

### 1.3 状态跟踪（State Tracking）

针对 ALFWorld 中涉及物体状态转换的任务，实现 `TaskStateTracker`。

主要关注三种状态：

- `clean`：物体清洁状态。
- `cool`：物体冷却状态。
- `heat`：物体加热状态。

核心功能包括：

- 根据实际环境交互记录物体状态转换。
- 区分不同物体实例，避免状态混淆。
- 根据环境反馈确认状态，而非仅根据物体所在位置推断状态。
- 处理后续操作可能导致的状态变化或失效。
- 在 Agent 决策过程中提供状态提醒。

例如，任务要求将冷却后的 Potato 放入 Microwave，Agent 需要确认目标 Potato 已完成冷却，不能仅因为物体位于 Fridge 中就推断冷却条件已经满足。

State Tracking 可以独立启用，也可以与长期经验记忆组合使用。

### 1.4 长期经验自进化（Memory Evolution）

在原始经验提炼机制的基础上，进一步实现长期记忆的动态更新。

系统根据历史失败轨迹与反事实续跑结果提炼候选 Lesson，并针对已有记忆执行更新操作。

实现三种更新机制：

| 方法 | 更新操作 | 核心思想 |
|---|---|---|
| Direct Update | ADD / KEEP / EDIT / REMOVE | 根据候选经验与已有经验的关系动态增删改 |
| Rule Strength Voting | ADD / AGREE | 对已有规则进行匹配和强度强化 |
| Source-Aware Revision | CREATE / CORROBORATE / REFINE | 积累来源信息，持续细化和修订经验内容 |

三种方法使用相同的候选 Lesson，从空记忆库开始独立更新，以比较不同 Memory Update Policy 的下游任务效果。

## 2. 实验设置

| 配置 | 设置 |
|---|---|
| 交互环境 | ALFWorld / TextWorld |
| LLM | `qwen3.8-chat` |
| Train Episodes | 1,000 |
| Valid Seen | 140 |
| Valid Unseen | 134 |
| Random Seed | 42 |
| Max Steps | 50 |
| Experience Retrieval | BM25 (`source_task`) |
| Experience Top-K | 3 |
| 参数更新 | 无（Training-Free） |

使用固定任务 Manifest 保证各方法在相同任务集合上进行评估。

主要评估指标包括：

- **Success Rate**：任务完成率。
- **Average Decision Steps**：平均决策步数。
- **Improved / Regressed**：相较对照方法修复和退步的任务数。
- **Exact McNemar Test**：基于任务级配对结果的统计检验。

## 3. 原始经验学习与反事实验证实验

### 3.1 Train 阶段

在 1,000 个 Train 任务上，ReAct Agent 成功完成 867 个任务，失败 133 个。

| 指标 | 结果 |
|---|---:|
| Train Success | 867/1000 |
| Train Success Rate | 86.70% |
| Average Decision Steps | 20.26 |
| Failed Episodes | 133 |
| Counterfactual Candidates | 371 |
| Successful Counterfactual Rollouts | 197 |
| Counterfactual Success Rate | 53.10% |
| Episodes with Successful Counterfactual | 91/133 |

反事实验证在 371 条候选分支中产生 197 条成功续跑，表明部分原始失败可以通过替代关键决策及后续行动得到修复。

三种经验提炼方式生成的去重后 Memory 数量如下：

| Memory | 数量 |
|---|---:|
| Ordinary Memory | 132 |
| Unverified Memory | 132 |
| CF-Verified Memory | 130 |

### 3.2 ReAct / Reflexion / Experience Memory 对照

| 方法 | Seen Success | Seen Avg Steps | Unseen Success | Unseen Avg Steps |
|---|---:|---:|---:|---:|
| ReAct | 120/140 (85.71%) | 19.97 | 122/134 (91.04%) | 18.24 |
| Ordinary Memory | 122/140 (87.14%) | 18.59 | 127/134 (94.78%) | 15.32 |
| Unverified Memory | 127/140 (90.71%) | 17.01 | 129/134 (96.27%) | 14.55 |
| **CF-Verified Memory** | **128/140 (91.43%)** | **15.54** | **131/134 (97.76%)** | **13.22** |
| Reflexion | 133/140 (95.00%) | 25.19 | 130/134 (97.01%) | 21.78 |

与 ReAct 相比，CF-Verified Memory：

- Seen 成功率提升 **5.72 个百分点**，平均决策步数降低约 **22.2%**。
- Unseen 成功率提升 **6.72 个百分点**，平均决策步数降低约 **27.5%**。

这说明外部经验记忆能够改善 ReAct Agent 的整体任务表现。

反事实验证经验在该实验中取得更好的成功率和决策效率，但相较 Unverified Memory 的额外收益较小。

Reflexion 最多允许两次任务尝试，因此与单次执行的 ReAct 和 Memory 方法并非严格等预算比较。

## 4. Memory Scaling 消融实验

基于原始 130 条 CF-Verified Memory，构建不同规模的经验子集，研究经验数量对 Agent 成功率和决策效率的影响。

| Memory | 经验数量 | Seen Success | Seen Avg Steps | Unseen Success | Unseen Avg Steps |
|---|---:|---:|---:|---:|---:|
| M0 | 0 | 120/140 | 19.97 | 122/134 | 18.24 |
| M25 | 32 | 127/140 | 16.15 | 125/134 | 15.80 |
| M50 | 65 | **132/140** | **14.72** | 128/134 | 15.05 |
| M75 | 98 | 130/140 | 16.00 | 127/134 | 13.87 |
| M100 | 130 | 128/140 | 15.54 | **131/134** | **13.22** |

**实验发现：**

- Valid Seen 在 M50（65 条经验）时达到最高成功率 **94.29%**。
- Valid Unseen 在 M100（130 条经验）时达到最高成功率 **97.76%**。
- 经验数量增加并不一定带来更高成功率。
- 经验内容、任务覆盖范围和检索结果共同影响 Agent 的实际表现。

实验使用固定随机顺序构造的嵌套经验子集，因此结果反映该次记忆规模消融，不代表普遍最优的记忆数量。

## 5. Memory × State Tracking 消融实验

通过独立控制是否使用 CF-Verified Memory，以及是否启用 State Tracking，开展 2×2 消融实验。

| 方法 | Memory | State | Seen Success | Seen Avg Steps | Unseen Success | Unseen Avg Steps |
|---|:---:|:---:|---:|---:|---:|---:|
| ReAct | — | — | 120/140 | 19.97 | 122/134 | 18.24 |
| ReAct + State | — | ✓ | 128/140 | 17.43 | 126/134 | 15.62 |
| CF-Verified Memory | ✓ | — | 128/140 | 15.54 | 131/134 | 13.22 |
| **CF-Verified + State** | ✓ | ✓ | **133/140** | **13.79** | **134/134** | **12.11** |

CF-Verified + State 在该实验中达到：

- **Valid Seen：133/140（95.00%）**
- **Valid Unseen：134/134（100.00%）**

### 配对统计检验

| 对比 | Split | Improved | Regressed | Exact McNemar p |
|---|---|---:|---:|---:|
| Verified → Verified + State | Seen | 8 | 3 | 0.2266 |
| Verified → Verified + State | Unseen | 3 | 0 | 0.2500 |
| ReAct + State → Verified + State | Seen | 7 | 2 | 0.1797 |
| ReAct + State → Verified + State | Unseen | 8 | 0 | 0.0078 |

实验表明，经验记忆与状态跟踪具有一定互补作用，组合后取得了较高的任务成功率和较低的平均决策步数。

同时，状态跟踪不能消除所有失败，目标物体选择、搜索规划和非状态类任务仍可能出现负迁移。

上述显著性检验为探索性配对比较，未进行多重比较校正。

## 6. Memory Evolution 全量实验

### 6.1 经验生成与动态更新

使用 Train 阶段的 133 个失败 Episode 和 371 条反事实候选分支，执行完整的候选经验提炼与更新过程。

原始 Direct Update 的分支处理结果：

| 处理结果 | 数量 |
|---|---:|
| SKIPPED | 177 |
| ABSTAIN | 21 |
| KEEP | 151 |
| PROVISIONAL | 22 |
| **合计** | **371** |

其中 173 条候选 Lesson 进入统一的更新算法对照实验。

三种方法使用相同候选池，生成三套独立的 Memory。

| 方法 | 更新操作统计 | 最终 Memory |
|---|---|---:|
| Direct Update | ADD 19、KEEP 151、EDIT 2、REMOVE 1 | **18** |
| Rule Strength Voting | ADD 21、AGREE 152 | **21** |
| Source-Aware Revision | CREATE 22、CORROBORATE 137、REFINE 14 | **22** |

Direct Update 通过动态增删改得到 18 条经验；Rule Strength Voting 主要通过规则匹配和强度更新得到 21 条经验；Source-Aware Revision 通过来源补充及内容修订得到 22 条经验。

三组结果均来自完整候选池，而非小规模 Pilot。

### 6.2 Memory Evolution × State Tracking 完整效果验证

对原始 CF-Verified Memory 和三套进化 Memory，在 Valid Seen / Valid Unseen 上分别启用与关闭 State Tracking，共完成 **16 组评估实验**。

所有方法使用相同的任务 Manifest、执行模型、最大步数和经验检索数量。

每个单元格表示：

**成功任务数 / 总任务数（平均决策步数）**

| Memory | Seen 无 State | Seen + State | Unseen 无 State | Unseen + State |
|---|---|---|---|---|
| Original CF-Verified (130) | 128/140 (15.14) | **133/140 (13.75)** | **130/134 (12.80)** | **134/134 (11.93)** |
| Direct Update (18) | 126/140 (16.48) | 133/140 (14.35) | 130/134 (13.65) | 128/134 (13.92) |
| Rule Strength Voting (21) | **129/140 (15.51)** | 133/140 (14.40) | 130/134 (13.84) | 130/134 (13.49) |
| Source-Aware Revision (22) | 126/140 (16.36) | 133/140 (14.46) | 129/134 (14.46) | 133/134 (13.52) |

本表来自最新统一评估批次。第 3、5 节保留历史实验批次结果；不同批次的部分成功数和平均步数存在差异，分别按各自原始记录报告。

### 6.3 配对统计结果

为分析经验更新带来的任务级变化，使用 Exact McNemar Test 比较相同任务上的成功与失败转换。

| 对比 | 实验设置 | Improved | Regressed | p-value |
|---|---|---:|---:|---:|
| Original → Direct Update | Seen / 无 State | 7 | 9 | 0.8036 |
| Original → Rule Strength Voting | Seen / 无 State | 7 | 6 | 1.0000 |
| Original → Source-Aware Revision | Seen / 无 State | 5 | 7 | 0.7744 |
| Original → Direct Update | Unseen / State | 0 | 6 | 0.0312 |
| Original → Rule Strength Voting | Unseen / State | 0 | 4 | 0.1250 |
| Original → Source-Aware Revision | Unseen / State | 0 | 1 | 1.0000 |
| Direct Update → Source-Aware Revision | Unseen / State | 5 | 0 | 0.0625 |

以上为双侧 Exact McNemar 检验结果，p 值未经多重比较校正。

### 6.4 实验发现

**1. 进化记忆显著压缩了经验库规模**

从原始 130 条经验缩减为 18～22 条，但这种压缩没有稳定转化为任务性能提升。

**2. 较强的 Agent 存在成功率天花板效应**

原始 CF-Verified Memory 在 Unseen + State 上达到 134/134（100%），因此无法继续通过成功率体现额外收益。

**3. 不同记忆更新机制存在任务级差异**

在 Seen、无 State 条件下，Rule Strength Voting 达到 129/140，比原始经验多成功一个任务，但统计检验不显著。

在 Unseen + State 上，Source-Aware Revision 达到 133/134，优于另外两种进化方法，但仍未超过原始 Memory。

**4. 新记忆没有稳定改善决策效率**

在四种评估条件下，三套进化 Memory 的平均决策步数均高于原始 130 条经验库。

**5. 记忆进化可能产生负迁移**

动态规则更新和经验压缩能够改变 Agent 行为，但可能减少任务覆盖能力，或使检索结果与实际任务需求不匹配。

**总体结论：** 本次全量实验完成了三种 Memory Evolution 机制的构建与对照，但尚无证据证明它们能在下游 ALFWorld 任务上稳定优于原始 CF-Verified Memory。

这一结果说明，长期记忆系统的实际效果不仅取决于更新策略，还与经验数量、任务覆盖、检索方式和 Agent 本身的能力有关。

## 7. 项目结构

```text
agent-self-evolution/
├── src/
│   ├── agents/
│   │   ├── react_agent.py
│   │   └── reflexion_agent.py
│   ├── analysis/
│   │   └── failure_analyzer.py
│   ├── core/
│   │   ├── protocol.py
│   │   └── trajectory.py
│   ├── counterfactual/
│   │   ├── replay.py
│   │   └── verifier.py
│   ├── data/
│   │   ├── alfworld_data.py
│   │   └── manifest.py
│   ├── envs/
│   │   └── environment.py
│   ├── llm/
│   │   └── client.py
│   └── memory/
│       ├── experience.py
│       ├── store.py
│       └── state_tracker.py
├── scripts/
│   ├── analysis/
│   ├── evaluation/
│   │   ├── evaluate_state_tracking.py
│   │   ├── evaluate_memory_scaling.py
│   │   └── evaluate_memory_evolution.py
│   └── memory/
│       ├── run_evolution.py
│       └── compare_updates.py
├── experiments/
├── docs/
│   └── experiment_results.md
├── pyproject.toml
└── README.md
```

主要代码模块：

| 模块 | 功能 |
|---|---|
| `ReActAgent` | 执行 ALFWorld 任务并记录完整轨迹 |
| `FailureAnalyzer` | 分析失败原因和关键决策 |
| `Counterfactual Verifier` | 环境重放、替代动作执行和反事实续跑 |
| `ExperienceStore` | 长期经验存储与 BM25 检索 |
| `TaskStateTracker` | 维护物体状态转换与确认信息 |
| `Memory Evolution` | 候选经验提炼及动态更新 |
| `Evaluation Scripts` | 成功率、步数、消融和配对统计 |

## 8. 环境与运行

### 8.1 环境配置

项目使用 Python 3.11，采用 `uv` 管理依赖。

安装项目依赖：

```bash
uv sync
```

下载 ALFWorld 环境数据：

```bash
uv run alfworld-download
```

使用 OpenAI-compatible API 调用 LLM，需配置以下环境变量：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

实验执行模型为 `qwen3.8-chat`。

### 8.2 Memory Evolution

基于 Train 数据生成完整候选经验并执行 Direct Update：

```bash
uv run python -m scripts.memory.run_evolution \
  --run-name train_full_v1 \
  --top-k 8
```

基于相同候选 Lesson，生成另外两套进化 Memory：

```bash
uv run python -m scripts.memory.compare_updates \
  --source-run train_full_v1 \
  --output-run train_full_comparison \
  --limit 371 \
  --top-k 8
```

### 8.3 完整效果评估

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

`--state both` 会分别运行启用和关闭 State Tracking 的四套 Memory 对照实验。

评估程序支持逐 Episode 保存轨迹及断点恢复，记录：

- `summary.json`：成功率、平均步数和逐任务结果。
- `comparison.json`：相同 State 设置下不同 Memory 的配对比较。
- `state_comparison.json`：相同 Memory 在不同 State 设置下的配对比较。
- `episode_XXX.json`：完整执行轨迹。

实验结果保存至：

```text
outputs/memory_evolution/evaluations/
```

`outputs/` 默认通过 `.gitignore` 排除，不提交完整实验轨迹和中间数据至 GitHub。

## 9. 项目总结

本项目实现了一套 **Training-Free LLM Agent Self-Evolution** 研究系统，在不更新底层语言模型参数的条件下，通过环境反馈、反事实验证、长期经验记忆和状态跟踪改善 Agent 的任务执行行为。

已完成的主要工作与结果：

1. **反事实经验学习**：针对 1,000 个 Train Episode 中的 133 个失败任务，完成 371 条反事实候选验证，并构建 130 条 CF-Verified Memory。
2. **经验学习效果验证**：CF-Verified Memory 在原始实验中相较 ReAct 提升 Seen / Unseen 成功率，并降低平均决策步数。
3. **记忆规模消融**：完成 M0、M25、M50、M75、M100 五组对照，发现记忆数量与性能并非单调关系。
4. **状态跟踪机制**：实现 `TaskStateTracker`，与 CF-Verified Memory 结合后，在 Seen 上达到 133/140，在 Unseen 上达到 134/134。
5. **长期记忆进化**：实现 Direct Update、Rule Strength Voting、Source-Aware Revision 三种更新算法，将统一候选池构建为 18、21、22 条经验。
6. **完整下游评估**：完成 16 组 Memory Evolution × State Tracking 实验及配对统计分析，发现动态更新和经验压缩未必能够稳定改善任务成功率与执行效率。

研究表明，**LLM Agent 的经验自进化不仅取决于是否能够产生和更新经验，还取决于经验的可靠性、任务覆盖能力、检索机制及其与 Agent 决策能力的适配关系。**

