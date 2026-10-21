# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution with Counterfactual Verification, State Tracking, and Evidence-Grounded Experience Memory**

本项目研究如何让大语言模型智能体（LLM Agent）在**不更新模型参数**的前提下，利用环境反馈积累可复用经验，并在后续任务中改进决策。以 **ALFWorld** 文本交互环境为实验平台，项目构建了从任务执行、失败分析、反事实验证、经验提炼与检索，到状态追踪、主张级证据审计和受控干预的完整研究流程。

项目的核心问题不是简单地“让 Agent 记住经验”，而是进一步研究：**哪些经验真正有用？哪些经验有证据支持？环境中的成功续跑能否证明经验中的因果主张？如何降低错误经验和状态推断造成的负迁移？**

> **实验进度（2026-10-20）**：已完成 ReAct / Reflexion 对照、三种经验记忆方法、seen/unseen 评测、经验规模消融、Memory × State Tracking 的 2×2 消融、Claim-Level Evidence Matching 的 Dev10 对照，以及 episode_921 的 Potato 2×2 受控反事实实验。**Claim-Level 验证尚未形成经过下游 ALFWorld 评测的新经验库**，因此本 README 不宣称该部分已经提高任务成功率。
>
> 完整实验记录、统计口径和复现说明见 [docs/experiment_results.md](docs/experiment_results.md)。

## 1. 研究目标

本项目围绕四个相互联系的研究问题展开：

1. **经验学习（Experience Learning）**：从历史失败轨迹提炼的经验，能否改善 ReAct Agent 在新任务上的成功率与执行效率？
2. **反事实验证（Counterfactual Verification）**：让 Agent 回到关键决策状态，替换动作并观察环境结果，是否能获得比纯语言反思更可靠的经验反馈？
3. **状态感知决策（State-Aware Execution）**：显式跟踪物体的 `clean`、`cool`、`heat` 等状态，能否减少状态遗漏、错误推断与经验带来的负迁移？
4. **证据驱动的经验可靠性（Evidence-Grounded Experience Validation）**：将自然语言经验拆为原子主张，追溯执行证据、识别混杂因素，并通过受控实验检验其因果有效性。

这里的 **Self-Evolution** 指 Agent 通过历史交互、外部经验记忆及可验证环境反馈改进后续行为，属于 **Training-Free / Non-Parametric Adaptation**；本项目并未使用 PPO、GRPO 等方法更新模型权重，也不将普通上下文反思表述为参数训练。

## 2. 系统整体流程

```text
                         ALFWorld Tasks
                               │
                               ▼
                  ReAct Agent (qwen3.8-chat)
                               │
                      Environment Feedback
                               │
                         Episode Trajectory
                               │
                        Failed Episodes
                               │
                               ▼
                     Failure Analyzer
               (critical step / failure type /
                  counterfactual candidates)
                               │
                               ▼
                     Environment Replay
                   + Counterfactual Rollout
                   (actual ALFWorld won)
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
       Experience Extraction          Evidence Investigation
     Ordinary / Unverified /          Atomic Claim Extraction
           CF-Verified                Evidence Packet Matching
                 │                    Controlled Interventions
                 ▼                            │
           ExperienceStore               Claim Reliability
         (lesson deduplication)       (Dev research and audit)
                 │
            BM25 Retrieval
             (Top-K = 3)
                 │
                 ▼
          Retrieved Lessons
                 │
                 ▼
          ReAct / State-Aware Agent
        (optional state tracking of
           clean / cool / heat)
                 │
                 ▼
             ALFWorld Evaluation
       success / decision steps / regressions
```

**实施状态**：基于 Ordinary / Unverified / CF-Verified 的经验库检索和 State-Aware 执行已经完成下游评测；Claim-Level 证据审计与受控干预已完成开发阶段案例实验，尚未完成“基于 Claim 可靠性筛选记忆 → 下游任务评测”的最终闭环。上图表示项目的统一研究体系，不意味着所有研究模块均已在生产式 Agent 中联动。

### 2.1 ReAct 与失败分析

ReAct Agent 在 ALFWorld 中逐步选择合法动作，保存每步的 `observation`、`admissible_actions`、`thought`、`action`、`next_observation`、`done` 和 `won` 等信息。`FailureAnalyzer` 从失败轨迹中抽取：

- `critical_step`：可能影响结果的关键决策位置；
- `failure_type` / `failure_reason`：失败类型与候选原因；
- `counterfactual_actions`：拟在关键状态尝试的替代动作；
- `expected_effect`：对候选动作的预期效果。

失败原因与候选动作由 LLM 提议，不直接当作环境真值。

### 2.2 反事实环境验证

验证器使用 `replay_to_step()` 恢复到关键动作**执行之前**的 ALFWorld 状态，执行候选替代动作，再让 Agent 在新状态下继续运行，记录真实的任务完成结果。候选成功是 `won=True`，而 `done=True` 仅代表 Episode 终止。

这种机制可以回答“某个替代动作及其后续决策能否完成任务”，**但尚不能单独证明该动作是成功的唯一原因**，也不能直接保证提炼出的通用 Lesson 正确。因此项目同时研究严格控制变量的 Targeted Counterfactual Verification。

### 2.3 长期经验记忆

经验以 `failure_type`、`lesson`、`source_task` 为主要字段保存。`ExperienceStore` 对完全相同的 `lesson` 文本去重，并使用 **BM25 对 `source_task` 检索 Top-3**，将相关经验注入新任务的初始 ReAct Prompt。

三种经验提炼方法使用同一检索协议，但提炼依据不同：

| 方法 | 提炼依据 | 环境验证 |
|---|---|---|
| **Ordinary Memory** | 完整失败轨迹的直接总结 | 否 |
| **Unverified Memory** | 失败关键步骤与 LLM 建议的反事实动作 | 否 |
| **CF-Verified Memory** | 失败分析、反事实环境执行结果和续跑轨迹 | 是 |

### 2.4 状态追踪（State Tracking）

`TaskStateTracker` 根据交互轨迹维护目标操作、物体实例、状态转换动作及确认结果，主要针对 `clean`、`cool`、`heat` 等任务要求。State-Aware Agent 可利用这些信息约束行动规划，例如：

- 不因物体放在 Fridge 中就直接推断它已被冷却；
- 将 `potato 2` 与 `potato 3` 的状态分别维护，避免实例混淆；
- 不把移动、存储或观察动作直接等同于满足任务要求的状态转换；
- 注意后续的加热、冷却等动作可能使先前状态判断失效。

状态追踪来自可观测轨迹及规则解析，**不是对 ALFWorld 隐藏目标状态的直接读取**，也不能自动替代环境成功判据。

### 2.5 主张级证据验证（Claim-Level Evidence Verification）

自然语言 Lesson 往往包含多个断言：对象类型是否等价、某种动作顺序是否必要、某个策略是否总是有效。一次反事实续跑成功通常不足以同时验证这些断言。

为此，项目在开发阶段实现并评估：

1. **Atomic Claim Extraction**：将 Lesson 拆解成尽量独立且保留原逻辑强度的 Claims；
2. **Evidence Packet Construction**：关联 Baseline、反事实候选、步骤反馈和证据 ID；
3. **Evidence Matching**：判断证据支持、反驳或不足，并比较不同验证模型；
4. **Targeted Controlled Replay**：当物体实例、动作和后续路径同时变化时，通过同状态重放及控制变量补充因果证据。

后续方法将区分 **Case-Level Support、Causal Verification、Generalization Evidence**：当前任务中得到支持，不等于已在其他任务中证明普适有效。

## 3. 实验设置

| 设置 | 取值 |
|---|---|
| 交互环境 | ALFWorld（TextWorld 文本任务） |
| 执行模型 | `qwen3.8-chat` |
| 固定随机种子 | `42` |
| 训练任务 | 1000 |
| `valid_seen` | 140 |
| `valid_unseen` | 134 |
| Agent 决策上限 | 50 |
| 环境交互上限 | 50 |
| 检索方法 | BM25(`source_task`) |
| 检索经验数 | Top-K = 3 |
| 经验去重 | 完全相同 `lesson` 字符串 |
| 任务成功信号 | ALFWorld `won=True` |

任务 manifest 固定各个 split 的 Episode 集合，所有同协议对照在相同任务上进行比较。`valid_seen` / `valid_unseen` 的任务组成不同；由于研究中已多次使用这些数据分析错误和选择方法，**它们不能被视为完全未接触的最终测试集**。

## 4. 主要实验结果

### 4.1 训练数据与反事实采集

在 Train 1000 个任务上，ReAct 成功 **867/1000（86.70%）**，失败 133 个，平均决策步数 **20.26**。失败分析产生 **371** 个反事实候选，其中 **197** 个在替换动作并续跑后成功（53.10%），91/133 个失败 Episode 至少存在一条成功续跑。

三种经验库的去重后规模分别为：Ordinary **132**、Unverified **132**、CF-Verified **130** 条。训练阶段的反事实成功统计并非 371 个独立受控因果实验。

### 4.2 经验记忆与 Reflexion 对照

| 方法 | Seen 成功率 | Seen 平均步数 | Unseen 成功率 | Unseen 平均步数 |
|---|---:|---:|---:|---:|
| ReAct | 120/140（85.71%） | 19.97 | 122/134（91.04%） | 18.24 |
| Ordinary Memory | 122/140（87.14%） | 18.59 | 127/134（94.78%） | 15.32 |
| Unverified Memory | 127/140（90.71%） | 17.01 | 129/134（96.27%） | 14.55 |
| **CF-Verified Memory** | **128/140（91.43%）** | **15.54** | **131/134（97.76%）** | **13.22** |
| Reflexion（最多 2 次尝试） | 133/140（95.00%） | 25.19 | 130/134（97.01%） | 21.78 |

CF-Verified Memory 相对 ReAct 在 Seen、Unseen 上分别提高 **5.72、6.72 个百分点**，平均决策步数分别下降约 **22.2%、27.5%**。但它相对 Unverified Memory 的净成功数仅增加 **1、2** 个，不能据此宣称额外收益具有统计显著性。Reflexion 允许最多两次尝试，不属于严格等预算比较。

配对 Episode 分析也显示记忆可能产生负迁移：例如 ReAct → CF-Verified 在 Seen 修复 13 个 Episode，但使原本成功的 5 个回退；Unseen 修复 11 个、回退 2 个。

### 4.3 经验规模消融

| 经验规模 | 数量 | Seen 成功率 | Unseen 成功率 |
|---|---:|---:|---:|
| M0 | 0 | 85.71% | 91.04% |
| M25 | 32 | 90.71% | 93.28% |
| M50 | 65 | **94.29%** | 95.52% |
| M75 | 98 | 92.86% | 94.78% |
| M100 | 130 | 91.43% | **97.76%** |

使用一次固定随机顺序的嵌套经验子集。Seen 在 M50 达到最高成功率，Unseen 在 M100 达到最高成功率；**经验数量增加并未表现出单调提升**。该实验不足以推出普遍的最优经验规模。

### 4.4 Memory × State Tracking：2×2 消融

在固定的执行模型、同一批任务和相同评测协议下，分别控制是否注入 Verified Memory、是否启用 State Tracking。

| 方法 | Memory | State | Seen 成功率 | Seen 平均步数 | Unseen 成功率 | Unseen 平均步数 |
|---|:---:|:---:|---:|---:|---:|---:|
| ReAct | — | — | 120/140（85.71%） | 19.97 | 122/134（91.04%） | 18.24 |
| CF-Verified | ✓ | — | 128/140（91.43%） | 15.54 | 131/134（97.76%） | 13.22 |
| ReAct + State | — | ✓ | 128/140（91.43%） | 17.43 | 126/134（94.03%） | 15.62 |
| **CF-Verified + State** | ✓ | ✓ | **133/140（95.00%）** | **13.79** | **134/134（100.00%）** | **12.11** |

**CF-Verified + State** 相对 ReAct 增加 **13/140** 个 Seen 成功任务和 **12/134** 个 Unseen 成功任务。但状态追踪的额外贡献必须通过配对比较判断，不能只看最佳组合的绝对成功率。

| 配对比较 | Split | Fail→Success | Success→Fail | Exact McNemar 双侧 p |
|---|---|---:|---:|---:|
| CF-Verified → CF-Verified + State | Seen | 8 | 3 | 0.2265625 |
| CF-Verified → CF-Verified + State | Unseen | 3 | 0 | 0.2500000 |
| ReAct + State → CF-Verified + State | Seen | 7 | 2 | 0.1796875 |
| ReAct + State → CF-Verified + State | Unseen | 8 | 0 | 0.0078125 |

前两项状态追踪增益**没有达到 0.05 显著性阈值**。最后一项 p 值属于未经多重比较校正的探索性检验，且 Unseen 已出现明显天花板效应。状态追踪并非消除所有错误：搜索规划、对象类型选择和非状态任务仍可能回退。

开发阶段的状态审计中，Seen 的 Tracked/Untracked 为 **68/72**，Unseen 为 **75/59**。这些是状态追踪覆盖计数，**不是状态预测准确率**。

### 4.5 Claim-Level Evidence Verification：Dev 实验

在训练经验中构造 50 条经验的人工审计基准，当前使用 **Dev10** 调试方法，另 **Test40** 暂未用于 Prompt 调整。Dev10 的 Lesson-Level 标注为：**5 SUPPORTED、1 CONTRADICTED、4 INSUFFICIENT_EVIDENCE**。原子主张提取从 v1 的 27 条调整为 v2 的 **28 条 Claims**。

对同一组 28 条 Claims 及其 Evidence Packets 进行对照：

| 证据匹配模型 | SUPPORTED | CONTRADICTED | INSUFFICIENT | 合计 |
|---|---:|---:|---:|---:|
| Qwen v1 | 10 | 1 | 17 | 28 |
| Qwen v2 | 10 | 0 | 18 | 28 |
| DeepSeek（相同 v2 Prompt 和证据输入） | 2 | 2 | 24 | 28 |

Qwen v2 与 DeepSeek **18/28（64.3%）** 的 Claim 判断一致，另外 10 条出现分歧。**模型间一致率不是准确率**，也不能将 Lesson 的人工标签直接视为所有 Atomic Claims 的真值。对照揭示的主要风险包括：把局部结果泛化成全局规则、忽略对象实例混杂、将 `done=True` 当作成功，以及凭终局失败直接推断唯一原因。

#### 受控因果验证：episode_921（Potato 2×2）

任务为 `put a cool potato in microwave.`。在 zero-based `critical_step=19` 恢复相同环境状态，针对 Potato 2 / 3 分别运行“不冷却”和“冷却”两个条件。

| 物体实例 | 未冷却 | 冷却 | 配对一致性检查 |
|---|---|---|---|
| Potato 2 | 5 动作；目标未达成 | 6 动作；目标达成 | 均通过 |
| Potato 3 | 5 动作；目标未达成 | 6 动作；目标达成 | 均通过 |

四组全部完成；每对试验均通过**相同恢复状态、相同干预前状态、除冷却外动作序列一致**的检查。因而可支持一个**局部因果结论**：在当前任务状态及指定执行路径下，为同一 Potato 增加冷却动作，使目标由未达成变为达成。

该结果**不能证明**所有 Cool 类任务均需相同动作，也不意味着已完成基于 Claim 验证的新记忆库评测。完整动作与反馈保存在本地：`outputs/analysis/controlled_verification/episode_921_potato_2x2_v1.json`。

## 5. 代码结构

下面列出主要实现模块与运行入口，并非仓库的完整文件树；研究过程中部分新模块可能尚未与 GitHub 远程分支同步。

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
│   ├── envs/
│   │   └── environment.py
│   ├── llm/
│   │   └── client.py
│   └── memory/
│       ├── experience.py
│       ├── store.py
│       ├── extractor.py
│       ├── ordinary_extractor.py
│       ├── unverified_extractor.py
│       └── state_tracker.py
├── scripts/
│   ├── analyze_failures.py
│   ├── verify_counterfactuals.py
│   ├── extract_experiences.py
│   ├── extract_ordinary_experiences.py
│   ├── extract_unverified_experiences.py
│   ├── build_experience_store.py
│   ├── compare_methods.py
│   ├── evaluate_reflexion.py
│   ├── evaluate_memory_scaling.py
│   └── analysis/
│       └── match_claim_evidence.py
├── experiments/
│   └── controlled_verification/
│       └── run_potato.py
├── tests/
└── docs/
    └── experiment_results.md
```

## 6. 环境与复现

### 6.1 运行环境

- Python **3.11**，使用 `uv` 管理依赖；
- ALFWorld / TextWorld，以及对应的任务与 PDDL 资源；
- 可用的 OpenAI-compatible LLM API 配置，执行模型为 `qwen3.8-chat`；
- 固定任务 manifest，执行前核对随机种子、路径和参数。

首次安装依赖及准备 ALFWorld 资源可使用：

```bash
uv sync
uv run alfworld-download
```

本项目通过环境变量 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 为 LLM 客户端提供连接配置。不同阶段运行 Qwen 或 DeepSeek 时，应明确记录实际模型名称，且**不要提交 `.env` 或 API Key**。

### 6.2 复现入口

以下命令用于说明主要功能入口。多数脚本在源码中配置输入和输出路径，运行之前请核对本地文件、manifest 和模型设置，以免覆盖已有结果；部分步骤涉及大量 LLM API 调用。

```bash
uv run python -m scripts.evaluation.evaluate_baseline
uv run python -m scripts.analyze_failures
uv run python -m scripts.verify_counterfactuals
uv run python -m scripts.extract_ordinary_experiences
uv run python -m scripts.extract_unverified_experiences
uv run python -m scripts.extract_experiences
uv run python -m scripts.build_ordinary_experience_store
uv run python -m scripts.build_unverified_experience_store
uv run python -m scripts.build_experience_store
uv run python -m scripts.analysis.compare_methods
uv run python -m scripts.evaluation.evaluate_reflexion
uv run python -m scripts.evaluation.evaluate_memory_scaling --split valid_seen --percentages 25 50 75
uv run python -m scripts.evaluation.evaluate_memory_scaling --split valid_unseen --percentages 25 50 75
uv run python -m experiments.controlled_verification.run_potato
```

完整的 Claim Extraction、Evidence Packet 和 State-Aware 批量评测命令依赖本地脚本版本及配置，请按实际源码执行；本 README 不将尚未核实的命令伪装为已经确认可复现的入口。

### 6.3 数据与输出

主要实验数据写入 `outputs/`，例如：

```text
outputs/manifests/
outputs/baseline_qwen3.8-chat_train_seed42_1000/
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000/
outputs/counterfactual_qwen3.8-chat_train_seed42_1000/
outputs/memory/
outputs/memory_scaling/
outputs/analysis/claims_dev_v2.jsonl
outputs/analysis/evidence_packets_dev_v1.jsonl
outputs/analysis/evidence_matches_dev_v2.jsonl
outputs/analysis/evidence_matches_deepseek_dev_v2.jsonl
outputs/analysis/controlled_verification/episode_921_potato_2x2_v1.json
```

`outputs/` 被 `.gitignore` 忽略。README 中记录实验路径和结果**不表示相应完整原始 JSON 均已上传 GitHub**。可溯源复现需要保留任务 manifest、模型配置、原始轨迹、证据 ID 和文件哈希。

## 7. 研究发现与局限

**当前可由实测结果支持的结论：**

- 从失败轨迹提炼并检索长期经验，在两组现有评估任务上均可改善 ReAct 的整体表现；
- 环境反事实验证有额外正向观察，但相比 Unverified 的净增益较小，且存在负迁移；
- State-Aware 执行与经验记忆结合，在当前评测中表现最好，但状态模块的独立增益仍存在统计不确定性；
- 成功的反事实续跑**不能自动证明**对应 Lesson 中所有主张真实、必要或可泛化；
- Potato 2×2 受控实验说明通过固定物体实例与环境状态，可以为单个 Claim 获得比自由续跑更强的**局部因果证据**。

**当前限制：**

- 单一执行模型、固定随机种子，且 seen/unseen 数据已经参与反复分析，仍需独立任务与跨模型验证；
- 多个配对比较未经多重检验校正；Unseen 任务成功率接近或达到上限，存在天花板效应；
- 当前 BM25 检索和经验提炼 Prompt 存在影响性能的其他因素，尚不能把整体收益完全归因于某个单独模块；
- Claim-Level 证据匹配仍在 Dev 阶段，尚无“Claim 过滤后的新经验库”的下游 Agent 成功率结论；
- 历史实验未完整保存 Token Usage 与实际账单，决策步数不是总训练/验证成本。

下一步将完善 Claim 的**案例支持、因果验证和跨任务泛化**三层判据，对 Knife/Butterknife 等具有混杂因素的案例执行受控干预，并在新的经验库构建后评估是否减少回退、提升可靠性。

---

**实验详情、任务类型分解、配对比较、统计限制与完整数据口径：**[docs/experiment_results.md](docs/experiment_results.md)。

