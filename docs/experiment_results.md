# Agent Self-Evolution — Experiment Results

> 更新截至 **2026-10-20**。本文件为**完整实验报告（可直接覆盖 `docs/experiment_results.md`）**：保留历史 ALFWorld 实验，并整合 State Tracking 与 Claim-Level Evidence Verification 的已运行结果。用户本地已有实验程序与原始输出，此次仅更新文档，不新建或修改任何实验代码和数据文件。未记录的 Token Usage、实际费用与独立最终测试集表现不作推测。

## 1. 实验目标

研究在不微调模型参数的情况下，经验记忆及其验证机制能否改善 ReAct Agent 的任务成功率与执行效率，并分析不同机制带来的收益与回退：

1. **Ordinary Memory**：直接从失败轨迹总结经验。
2. **Unverified Memory**：加入失败关键步骤分析及未验证的反事实候选。
3. **CF-Verified Memory**：在真实 ALFWorld 中执行反事实候选、记录续跑结果，再提炼经验。
4. **State Tracking / State-Aware Agent**：显式维护任务所需 clean/cool/heat 状态，提高状态相关决策的一致性。
5. **Claim-Level Evidence Verification（研究中）**：拆分 Lesson 主张、匹配环境证据，并通过定向受控干预解决因果混杂。

前 3 种经验方法在同一执行模型、同一任务 manifest 上评估。后两项为后续研究扩展，不能将其结果混同为此前 Verified Memory 的原有评测结果。

## 2. 模型、数据与协议

| 设置 | 数值或说明 |
|---|---|
| 环境 | ALFWorld 文本任务 |
| 执行 LLM | `qwen3.8-chat` |
| 固定随机种子 | `42` |
| 训练任务 | 1000 |
| `valid_seen` | 140 |
| `valid_unseen` | 134 |
| Agent 决策上限 | 50 |
| 环境交互上限 | 50 |
| 经验检索 | BM25 对 `source_task` 评分 |
| Top-K | 3 |
| 去重 | 完全相同 `lesson` 文本去重 |
| 任务完成信号 | `won=True`；`done=True` 只表示 episode 结束 |

固定 manifest：

```text
outputs/manifests/train_seed42_1000.json
outputs/manifests/valid_seen_seed42_140.json
outputs/manifests/valid_unseen_seed42_134.json
```

`valid_seen` / `valid_unseen` 的任务类型构成不完全相同，不应只从两者原始成功率判断难度。两组评估集已在研究中多次用于调试、方法选择和误差分析，因此不再是完全未接触的最终独立测试集。

## 3. Train ReAct Baseline

### 3.1 整体结果

| Metric | Result |
|---|---:|
| Episodes | 1000 |
| Successes | 867 |
| Success rate | 86.70% |
| Failures | 133 |
| Truncated rate | 1.90% |
| Avg decision steps | 20.26 |
| Avg environment steps | 20.19 |
| Format errors | 67 |
| Inadmissible actions | 670 |
| Format valid rate | 99.67% |
| Admissible action rate | 96.68% |
| Valid action rate | 96.36% |

### 3.2 各任务类型

| Task type | Successes / Episodes | Success rate |
|---|---:|---:|
| `look_at_obj_in_light` | 80/88 | 90.91% |
| `pick_and_place_simple` | 205/208 | 98.56% |
| `pick_clean_then_place_in_recep` | 149/189 | 78.84% |
| `pick_cool_then_place_in_recep` | 105/138 | 76.09% |
| `pick_heat_then_place_in_recep` | 97/130 | 74.62% |
| `pick_two_obj_and_place` | 231/247 | 93.52% |

133 个失败中，clean/cool/heat 三类合计 106 个（约 79.7%）。这为后续状态追踪研究提供了动机，但失败比例本身不能证明状态错误是全部失败的唯一原因。

## 4. 失败分析与初始反事实环境验证

### 4.1 失败分析

输入：133 条 Train 失败轨迹。LLM 提取 `critical_step`、`failure_type`、`failure_reason`、`original_action`、`counterfactual_actions`、`expected_effect`。

```text
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000/
```

### 4.2 替换动作并自主续跑

对候选动作重放到关键决策之前，执行替代动作，再让相同 ReAct Agent 继续运行。

| Metric | Result |
|---|---:|
| Failed episodes analyzed | 133 |
| Counterfactual candidates | 371 |
| Successful candidate continuations | 197 |
| Failed candidate continuations | 174 |
| Candidate continuation success rate | 53.10% |
| Episodes with ≥1 successful candidate | 91/133 (68.42%) |
| Episodes with no successful candidate | 42/133 (31.58%) |
| Average candidates per failed episode | 2.79 |

```text
outputs/counterfactual_qwen3.8-chat_train_seed42_1000/
```

该过程表明“执行某个候选动作后、给定 Agent 后续决策可以完成任务”，但不能证明候选动作是唯一因果原因，更不能直接证明由其提炼的每条自然语言经验都正确。

## 5. 三类经验记忆

| Method | Extraction input | Raw | Unique | Removed duplicates |
|---|---|---:|---:|---:|
| Ordinary | 完整失败轨迹 | 133 | 132 | 1 |
| Unverified | 失败分析、关键步骤、反事实候选与预期效果 | 133 | 132 | 1 |
| Verified | 失败分析、反事实续跑结果和动作轨迹 | 133 | 130 | 3 |

共同经验字段为 `failure_type`、`lesson`、`source_task`；根据 `lesson` 完全相同文本去重，并使用 `BM25(source_task)` 检索 Top-3。三种方法的输入和提炼 Prompt 不同，实际结果因此不能归因于一个孤立因素。

```text
outputs/memory/ordinary_experience_store_qwen3.8-chat_train_seed42_1000.json
outputs/memory/unverified_experience_store_qwen3.8-chat_train_seed42_1000.json
outputs/memory/experience_store_qwen3.8-chat_train_seed42_1000.json
```

另一次后续审计确认，130 条去重 Verified Lesson 中有 **91 条**其来源至少存在一条成功反事实续跑；**39 条**没有这种成功续跑证据，仍被保留在经验库中。这里的 91/39 按去重经验计数，而第 4 节的 91/133、42/133 是失败 episode 层面的统计，**单位不同，不可混淆**。

## 6. Held-out 整体表现

### 6.1 成功率与平均决策步数

| Method | Seen successes | Seen rate | Seen avg steps | Unseen successes | Unseen rate | Unseen avg steps |
|---|---:|---:|---:|---:|---:|---:|
| ReAct | 120/140 | 85.71% | 19.97 | 122/134 | 91.04% | 18.24 |
| Ordinary | 122/140 | 87.14% | 18.59 | 127/134 | 94.78% | 15.32 |
| Unverified | 127/140 | 90.71% | 17.01 | 129/134 | 96.27% | 14.55 |
| **Verified (Ours, original)** | **128/140** | **91.43%** | **15.54** | **131/134** | **97.76%** | **13.22** |
| Reflexion（最多两次尝试） | 133/140 | 95.00% | 25.19 | 130/134 | 97.01% | 21.78 |

### 6.2 成功率增量（百分点）

| Comparison | Seen Δ | Unseen Δ |
|---|---:|---:|
| ReAct → Ordinary | +1.43 | +3.74 |
| Ordinary → Unverified | +3.57 | +1.49 |
| Unverified → Verified | +0.72 | +1.49 |
| ReAct → Verified | +5.72 | +6.72 |

Verified 相对 ReAct 平均决策步数降低约 22.2%（seen）和 27.5%（unseen）；相对 Unverified 进一步降低约 8.6%（seen）和 9.1%（unseen）。这不包括经验库构建或验证开销。

## 7. 分任务类型成功率

### 7.1 valid_seen（140 episodes）

| Task type | ReAct | Ordinary | Unverified | Verified |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 12/13 | 10/13 | 11/13 | 12/13 |
| `pick_and_place_simple` | 35/35 | 35/35 | 35/35 | 35/35 |
| `pick_clean_then_place_in_recep` | 19/27 | 22/27 | 21/27 | 22/27 |
| `pick_cool_then_place_in_recep` | 19/25 | 19/25 | 21/25 | 21/25 |
| `pick_heat_then_place_in_recep` | 12/16 | 13/16 | 15/16 | 15/16 |
| `pick_two_obj_and_place` | 23/24 | 23/24 | 24/24 | 23/24 |

### 7.2 valid_unseen（134 episodes）

| Task type | ReAct | Ordinary | Unverified | Verified |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 18/18 | 17/18 | 18/18 | 18/18 |
| `pick_and_place_simple` | 24/24 | 24/24 | 24/24 | 24/24 |
| `pick_clean_then_place_in_recep` | 24/31 | 27/31 | 29/31 | 29/31 |
| `pick_cool_then_place_in_recep` | 18/21 | 21/21 | 18/21 | 20/21 |
| `pick_heat_then_place_in_recep` | 21/23 | 22/23 | 23/23 | 23/23 |
| `pick_two_obj_and_place` | 17/17 | 16/17 | 17/17 | 17/17 |

部分任务类型并没有持续改善（例如 Unseen 的 Cool：Ordinary 21/21，高于 Verified 20/21），不得将总体增益推广到所有任务。

## 8. Episode-level 配对分析：原始经验方法

按同一 split、同一 episode 文件名对齐，Fail→Success 是新方法修复，Success→Fail 是引入的回退。

### 8.1 valid_seen

| Comparison | Fail→Fail | Fail→Success | Success→Fail | Success→Success | Net gain |
|---|---:|---:|---:|---:|---:|
| ReAct → Ordinary | 10 | 10 | 8 | 112 | +2 |
| Ordinary → Unverified | 9 | 9 | 4 | 118 | +5 |
| Unverified → Verified | 8 | 5 | 4 | 123 | +1 |
| ReAct → Verified | 7 | 13 | 5 | 115 | +8 |

### 8.2 valid_unseen

| Comparison | Fail→Fail | Fail→Success | Success→Fail | Success→Success | Net gain |
|---|---:|---:|---:|---:|---:|
| ReAct → Ordinary | 1 | 11 | 6 | 116 | +5 |
| Ordinary → Unverified | 1 | 6 | 4 | 123 | +2 |
| Unverified → Verified | 1 | 4 | 2 | 127 | +2 |
| ReAct → Verified | 1 | 11 | 2 | 120 | +9 |

Verified 相对 Unverified：Seen 净 +1、Unseen 净 +2。即使成功率有所提高，仍存在回退和统计不确定性。此历史配对表本身未包括显著性检验，不应仅凭计数宣称所有提升显著。

## 9. 格式、合法性及截断摘要

| Split / Method | Format errors | Inadmissible actions | Truncated rate |
|---|---:|---:|---:|
| seen / ReAct | 5 | 100 | 2.86% |
| seen / Ordinary | 0 | 28 | 0.00% |
| seen / Unverified | 0 | 27 | 0.00% |
| seen / Verified | 3 | 73 | 0.71% |
| unseen / ReAct | 0 | 45 | 0.00% |
| unseen / Ordinary | 10 | 81 | 0.75% |
| unseen / Unverified | 0 | 24 | 0.00% |
| unseen / Verified | 0 | 22 | 0.00% |

这些属于历史摘要，并非新版本 State-Aware Agent 的错误计数。

## 10. 实验文件与复现

### 10.1 原有训练集数据

```text
outputs/baseline_qwen3.8-chat_train_seed42_1000/
outputs/failure_analysis_qwen3.8-chat_train_seed42_1000/
outputs/counterfactual_qwen3.8-chat_train_seed42_1000/
outputs/ordinary_experiences_qwen3.8-chat_train_seed42_1000/
outputs/unverified_experiences_qwen3.8-chat_train_seed42_1000/
outputs/experiences_qwen3.8-chat_train_seed42_1000/
```

### 10.2 原有 seen/unseen 评测输出

```text
outputs/baseline_qwen3.8-chat_valid_seen_seed42_140/
outputs/baseline_qwen3.8-chat_valid_unseen_seed42_134/
outputs/ordinary_memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/ordinary_memory_qwen3.8-chat_valid_unseen_seed42_134/
outputs/unverified_memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/unverified_memory_qwen3.8-chat_valid_unseen_seed42_134/
outputs/memory_qwen3.8-chat_valid_seen_seed42_140/
outputs/memory_qwen3.8-chat_valid_unseen_seed42_134/
```

### 10.3 统计脚本

```bash
uv run python -m scripts.analysis.compare_methods
uv run python -m scripts.analysis.analyze_task_types
```

`compare_methods` 逐 episode 按顶层 `success` 统计配对转移；是否包含 Exact McNemar 取决于具体脚本版本，不应把外部计算的 p 值误说成该旧脚本已实现。

## 11. 历史经验方法的结论与证据边界

1. Ordinary Memory 相对 ReAct 在两个 split 上表现更好；经验记忆具有正向总体效果。
2. Unverified Memory 相对 Ordinary 继续提高整体成功率。
3. 原始 Verified Memory 相对 Unverified 的收益存在，但净新增成功数分别只有 +1/+2；不能把效果全部归因于验证器。
4. 引入经验也可能造成负迁移；来源数据、提炼 Prompt、记忆检索结果等都是潜在影响因素。
5. 模型参数未被优化；上述“自进化”是经验层面的训练自由改进。

## 12. 经验库规模消融和推理成本估算

### 12.1 规模消融

使用原有 130 条 Verified Lessons 构建固定随机嵌套子集，M0/M100 复用原评测。

| 经验比例 | 条数 | Seen 成功数 | Seen 成功率 | Seen avg steps | Unseen 成功数 | Unseen 成功率 | Unseen avg steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0 | 120/140 | 85.71% | 19.97 | 122/134 | 91.04% | 18.24 |
| M25 | 32 | 127/140 | 90.71% | 16.15 | 125/134 | 93.28% | 15.80 |
| M50 | 65 | 132/140 | 94.29% | 14.72 | 128/134 | 95.52% | 15.05 |
| M75 | 98 | 130/140 | 92.86% | 16.00 | 127/134 | 94.78% | 13.87 |
| M100 | 130 | 128/140 | 91.43% | 15.54 | 131/134 | 97.76% | 13.22 |

Seen 最优成功率 M50、Unseen 最优 M100。由于只有一次固定随机抽样，不足以证明经验数量和性能存在普遍单调关系。

### 12.2 估算调用与费用边界

| 方法 | Seen avg decisions | Unseen avg decisions |
|---|---:|---:|
| ReAct | 19.97 | 18.24 |
| Ordinary | 18.59 | 15.32 |
| Unverified | 17.01 | 14.55 |
| Verified | 15.54 | 13.22 |
| Reflexion | 25.19 | 21.78 |

在没有额外重试并且每决策仅进行一次 LLM 调用的假设下，平均决策步数可作为 ReAct 类方法的近似调用数。Reflexion 另有 Seen 26 次、Unseen 17 次反思调用，估算平均总调用约 25.38 / 21.91 次/任务；其最多允许两次尝试，不能视作等预算。历史没有保存详细 API token usage/费用，不能精确恢复账单，也不能仅通过验证阶段成本评估全训练流程开销。

## 13. 原实验复现注意事项

- 使用 Python 3.11、`uv`、已准备的 ALFWorld 任务与 PDDL/TextWorld 数据。
- 运行前检查每个脚本顶部的固定路径常量、seed、manifest、模型与输出目录。
- 历史失败分析、环境反事实续跑和经验提炼涉及大量 API 调用；已有输出不必重新运行。
- 对比必须使用同 split 的相同 episode 集合，保存逐 episode 的配对结果。
- 默认 `outputs/` 受 `.gitignore` 忽略。报告中列出路径仅说明本地已产生过这些数据，不等于 GitHub 远程包含所有源 JSON。

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
uv run python -m scripts.build_memory_scaling_stores
uv run python -m scripts.evaluation.evaluate_memory_scaling --split valid_seen --percentages 25 50 75
uv run python -m scripts.evaluation.evaluate_memory_scaling --split valid_unseen --percentages 25 50 75
```

## 14. State Tracking（Memory × State）消融，已完成

### 14.1 2×2 设计

四组固定相同的 qwen3.8-chat 执行模型与任务：ReAct、Verified、ReAct+State、Verified+State。显式状态追踪针对 clean/cool/heat 状态要求，目标是减少未确认或过早失效的状态假设。是否开启 Memory 与是否开启 State 共同构成 2×2 消融。

| Method | Memory | State | Seen Success | Seen avg steps | Unseen Success | Unseen avg steps |
|---|---|---|---:|---:|---:|---:|
| ReAct | 否 | 否 | 120/140 (85.71%) | 19.97 | 122/134 (91.04%) | 18.24 |
| Verified | 是 | 否 | 128/140 (91.43%) | 15.54 | 131/134 (97.76%) | 13.22 |
| ReAct+State | 否 | 是 | 128/140 (91.43%) | 17.43 | 126/134 (94.03%) | 15.62 |
| **Verified+State** | **是** | **是** | **133/140 (95.00%)** | **13.79** | **134/134 (100.00%)** | **12.11** |

### 14.2 配对检验（Exact McNemar，双侧）

| Comparison | Split | Fail→Success | Success→Fail | Net | p-value |
|---|---|---:|---:|---:|---:|
| Verified → Verified+State | Seen | 8 | 3 | +5 | 0.2265625 |
| Verified → Verified+State | Unseen | 3 | 0 | +3 | 0.2500000 |
| ReAct+State → Verified+State | Seen | 7 | 2 | +5 | 0.1796875 |
| ReAct+State → Verified+State | Unseen | 8 | 0 | +8 | 0.0078125 |

Verified → Verified+State 的两个 split 都**未**达到 p<0.05。ReAct+State → Verified+State 的 Unseen 比较 p=0.0078125，但这是多项未经校正的探索性比较之一；结合天花板效应与固定样本，不应夸大泛化结论。

### 14.3 回退和状态审计

Seen 回退包括 `episode_025`（物体类型错误）、`episode_053`（搜索与冷却未及时完成）、`episode_123`（非状态任务的循环）。这提示状态追踪不能解决所有目标类型、搜索和规划问题。

早期 TaskStateTracker 相关审核中，Tracked/Untracked 数在 Seen 为 68/72，Unseen 为 75/59。按轨迹和方法不同，Observed/Confirmed 计数为：Seen Unverified 57/57、Seen Verified 58/58、Unseen Unverified 71/70、Unseen Verified 72/72。这些是规则解析记录，**不是**状态正确率或成功率。

### 14.4 State-Aware 机制与局部验证

`TaskStateTracker` 的设计目标是区分任务所需操作（clean / cool / heat）、具体物体实例、观察到的转换动作、可确认状态以及后续可能的状态失效。它不直接读写 ALFWorld 的隐藏目标状态，也不代表环境级真值；只是根据交互轨迹维护可解释状态证据。典型错误包括：把物体位于冰箱附近误判为已冷却、把“移动到容器”当成状态转换、将 `potato 3` 的状态误套用到 `potato 2`。

本地开发阶段曾记录 `test_state_tracker.py` 8 项测试通过、`test_state_tracking_agent.py` 4 项测试通过，合计 12 项；这些是历史测试记录，**不是**本次文档更新重新执行的 CI 结果。局部 State-Aware 试验中，`episode_021`（clean egg）可在明确确认 clean 状态后完成；`episode_055`（cool potato）则显示经验与状态判断对不同执行路径会产生不同结果。这些是诊断性案例，不作为总体评测的额外成功样本。

### 14.5 State Tracking 与 Claim-Level Verification 的区别

前者改善 **Agent 执行时的状态约束**，后者核查 **长期记忆 Lesson 中的主张是否受到真实证据支持**。`TaskStateTracker.confirmed=True` 不能证明自然语言 Claim 成立。后续如果二者结合，需要分别保留 Memory、State 和 Claim-level Verification 消融，避免将性能增益错误归因于单一组件。

## 15. Claim-Level Evidence Verification（Dev 阶段，研究中）

### 15.1 Dev10 / Test40 标注

在训练经验中抽样 50 条用于证据研究，Dev10 已人工标注：Lesson-Level SUPPORTED 5、CONTRADICTED 1、INSUFFICIENT_EVIDENCE 4。Test40 冻结，尚未用于 Prompt 调优。Claim Extraction v1 提取 27 条，v2 提取 **28 条 Atomic Claims**。

### 15.2 Evidence Matching 模型对照

Dev10 Evidence Packets 关联 **10 个源 episode、28 个原始反事实候选**。分别用相同输入比较 Qwen v2 与 DeepSeek（具体 API alias/模型修订号须在正式复现中记录）。

| Evidence Matcher | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE | Total |
|---|---:|---:|---:|---:|
| Qwen v1 | 10 | 1 | 17 | 28 |
| Qwen v2 | 10 | 0 | 18 | 28 |
| DeepSeek，固定 v2 Prompt 与输入 | 2 | 2 | 24 | 28 |

Qwen v2 与 DeepSeek 的标签在 18/28（64.3%）上一致，10 条分歧。**该数值是模型间一致率，不是标签准确率**；Lesson Gold Label 不能作为 28 条 Claim 的 gold，且更保守不自动意味着更正确。审计发现：object-type substitution、量词范围、因果混杂、`done` / `won` 理解等重要差异。

### 15.3 episode_921 Potato 2×2 Controlled Replay

任务 `put a cool potato in microwave.`。重放至 zero-based critical_step=19，分别对 Potato 2、Potato 3 保持前置状态及非干预动作路径一致，只改变是否增加 `cool potato`。

| 实例 | No cooling | With cooling | Replay / pre-intervention / other actions |
|---|---|---|---|
| Potato 2 | goal_reached=False，5 动作 | goal_reached=True，6 动作 | 配对检查均 True |
| Potato 3 | goal_reached=False，5 动作 | goal_reached=True，6 动作 | 配对检查均 True |

四组状态全部 `completed`；两组配对的 `cooling_effect_supported_in_this_state` 均为 True。**局部因果解释**：在本次已重放状态与预定路径中，对相同 Potato 添加冷却操作，使目标由未达成变为达成。并不证明所有 Cool 任务均需要该操作，也不应把未达成目标等同于 `done=True` 的失败 Episode。

本地完整记录：

```text
outputs/analysis/controlled_verification/episode_921_potato_2x2_v1.json
```

这是一份**仅记录实验结果的报告**；完整动作证据仍在用户本地对应 JSON 中，状态、动作、观察、`admissible`、`done`、`won` 等字段应以该原始记录为准。此处不重新生成脚本或衍生 JSON，也不宣称原始数据已经上传 GitHub。

### 15.4 下一步及证据边界

1. 验证器输出应分开描述 **Case-Level Support**、**Causal Verification** 与 **Generalization Evidence**，避免单个三分类结果含义不清。
2. 对 Knife/Butterknife 等有反证迹象的主张设计更严格的同状态类型替换实验。
3. 保存针对性实验的完整 step-level 证据、SHA-256 和环境版本；将新的受控实验接入 Evidence Packet v2，**不要**修改旧的 v1 模型结果。
4. 只有完成新经验库的构建、过滤与下游执行评测，才能报告新 Claim-Level 方法是否提高成功率；**目前没有这类新下游结果**。
5. 必须在未参与调试的独立数据上验证，当前 Dev10 仅支持方法开发结论。

### 15.5 Dev10 典型差异案例及解释边界

| Episode | 观察与判断 | 限制 |
|---|---|---|
| `episode_397`（Pan / Pot） | 基线用 Pan 未达成目标、反事实用 Pot 达成目标；支持该任务中优先匹配准确物体类型 | 不足以证明任何 ALFWorld 物体类型都不能替换；“必须搜索准确物体”与“必须选择准确物体”不同 |
| `episode_467`（Knife / Butterknife） | 清洁 Knife 并放置 Drawer 后 `won=False`；DeepSeek 将相关 2 条 Claim 判为 `CONTRADICTED` | 尚需对照 Butterknife，排除清洁、容器或其他目标条件的潜在影响 |
| `episode_705`（Clean Mug） | 清洁操作可触发完成，即使后续没有新放置动作 | 不得据此证明“必须先清洁再放置”的操作顺序 |
| `episode_546`（Mug / Cup） | Mug 路径失败、Cup 路径成功 | 支持局部物体类型区别，不支持所有类型均严格区分的普遍断言 |
| `episode_955`（Take Pencil） | 两个 `take pencil` 反事实动作可直接触发 `won=True`，无需后续续跑 | `look/examine` “经常失败”的频率结论仍需多个任务验证；`done=True, won=False` 不是成功 |
| `episode_921`（Potato） | 原始轨迹既改变 Potato 实例又改变是否冷却；后续 2×2 消除了实例混杂 | 新结论限于同一恢复状态与固定动作路径 |
| `episode_190`（Mug / Cup） | 使用 Mug 未获胜，反事实没有成功的 Cup 路径 | 不能只凭失败排除其他原因，尚缺成功对照 |

### 15.6 阶段结论

本次完成了：Dev10 Lesson 级标注、28 Atomic Claims、证据包与双模型匹配、10 条模型分歧审计，以及一次成功的 Potato 2×2 受控验证。证据审计表明“反事实续跑成功”不足以证明原 Lesson 的所有断言；更强的验证机制应分离 **Case-Level Support / Causal Verification / Generalization Evidence**。截至本报告更新，**尚未**在新 Claim-Level 经验库上重新运行 Valid Seen/Unseen 的下游成功率评测，因此不报告此方法的性能收益。

