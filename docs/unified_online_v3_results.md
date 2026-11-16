# Unified Online Evolution：自动反馈调度与恢复验收

更新日期：2026-11-16

## 一、阶段目标

本阶段目标是将已有 Agent Runtime、Feedback Router、
Memory Review 和 Skill Review 整合为可恢复的在线工作流。

本阶段不设计新的记忆检索或技能学习算法，也不以单个
Train 批次的成功率证明自进化性能提升。

## 二、系统实现

主要组件：

| 组件 | 功能 |
|---|---|
| OnlineReviewScheduler | Review 调度、来源校验、结果缓存 |
| OnlineMemoryHandler | 失败分析、有限反事实验证、Lesson 候选提取 |
| OnlineSkillHandler | 对成功但低效的轨迹进行搜索诊断 |
| PersistentLLMBudget | 逐次预留模型调用额度，跨进程恢复 |
| run_unified_v3 | 已冻结批次的统一执行与 Review 入口 |

反馈路由规则：

- MEMORY_REVIEW：Train 学习任务执行失败。
- SKILL_REVIEW：Train 学习任务成功，但动作数明显高于同类参考值。
- ARCHIVE：成功且未触发低效判定。
- Validation / Holdout 不进入在线学习队列。

所有 Review 默认不自动晋升能力。

## 三、历史统一 Pilot

首次冻结 Pilot 使用六个 ALFWorld Train 任务：

- 成功 5/6。
- ARCHIVE：3。
- MEMORY_REVIEW：1。
- SKILL_REVIEW：2。

失败任务经过分析与部分反事实续跑，已测试的两个
反事实候选均未成功，其余候选未完成验证。
Memory Review 最终保持 NO_CHANGE。

两个 Skill Review 未形成经过验证的可迁移技能候选。

最终活跃能力保持 Memory v000、Skill None。

## 四、第二轮在线批次

批次：online_batch_01

数据来源：ALFWorld Train。

通过历史来源收集规则排除 1276 个已知游戏 ID。
选择六种任务类型各一个游戏，协议冻结并保存 SHA256。

| Episode | 任务类型 | 成功 | 环境动作 | 路由 |
|---|---|---|---:|---|
| batch_00 | pick_two_obj_and_place | 是 | 9 | ARCHIVE |
| batch_01 | pick_cool_then_place_in_recep | 是 | 35 | SKILL_REVIEW |
| batch_02 | pick_and_place_simple | 是 | 8 | ARCHIVE |
| batch_03 | look_at_obj_in_light | 是 | 21 | SKILL_REVIEW |
| batch_04 | pick_clean_then_place_in_recep | 是 | 8 | ARCHIVE |
| batch_05 | pick_heat_then_place_in_recep | 是 | 7 | ARCHIVE |

汇总：

- 成功数：6/6。
- 平均环境动作数：14.67。
- ARCHIVE：4。
- SKILL_REVIEW：2。
- MEMORY_REVIEW：0。
- Review 待处理数：0。
- Memory 晋升：0。
- Skill 晋升：0。

### Skill 诊断

batch_01：
- 任务：put a cool pan in stoveburner.
- 实际 35 步，历史同类中位数 11 步。
- 首次拿取目标前 30 步。
- 搜索了 17 个不同位置。

batch_03：
- 任务：examine the pen with the desklamp.
- 实际 21 步，历史同类中位数 8 步。
- 首次拿取目标前 17 步。
- 搜索了 13 个不同位置。

两个任务都成功完成，额外动作主要集中在目标搜索阶段。
这是搜索低效的诊断线索，不是新策略有效的因果证据。

两条 Review 均生成 DIAGNOSTIC_ONLY 记录，
没有生成或晋升新 Skill。

### 最终审计

执行：

    uv run python -m scripts.audit_unified_batch

结果：

- Audit：PASS。
- 6 条 Episode 的成功及路由记录一致。
- 2 条 Skill Review 的结果和缓存通过检查。
- 最终活跃能力保持 Memory v000、Skill None。

原始证据位于本地：

    outputs/unified_evolution_v2/online_batch_01/

其中 batch_audit.json 包含逐任务的轨迹与事件 SHA256。
outputs/ 默认不提交 Git，因此远程仓库不一定包含原始记录。

## 五、V3 统一入口

只读报告：

    uv run python -m scripts.run_unified_v3 --stage report

已有六任务批次的缓存恢复：

    uv run python -m scripts.run_unified_v3 \
      --stage run --limit 6 --max-new-jobs 0

实际检查结果：

- 六个 Episode 全部复用缓存。
- 两个 Skill Review 全部复用缓存。
- 新增 Episode：0。
- 新增 Review：0。
- 新增 Review LLM 调用：0。

V3 增加：

1. MEMORY_REVIEW 的持久化调用预留账本。
2. 每个 Review Job 的 Linux 进程锁。
3. 默认 max-new-jobs=0。
4. Review 结果不可重复改写，已有结果优先复用。

持久化预算采用调用前预留方式。
失败、超时或状态不明的请求不会自动退还额度。
这可能高估实际计费调用，但可避免重试时重新获得额度。

## 六、测试与限制

已验证：

- Scheduler 的缓存与任务来源检查。
- Memory Handler 的模拟分析、恢复及预算边界测试。
- Skill Handler 的诊断和结果校验。
- 持久化预算的重启、失败调用与损坏检查。
- V3 的 Review 锁和预算客户端注入测试。
- 真实六任务批次的执行、反馈路由和缓存恢复。

尚未验证：

1. V3 持久化预算在真实失败任务上的完整端到端执行。
2. 真实 Memory Review 中断后的所有恢复情况。
3. 在线 Memory 候选进入正式版本晋升门控的闭环。
4. 在线 Skill 自动生成、反事实评估和晋升闭环。
5. 新版本能力相对于固定版本的性能提升。
6. V3 对任意新增批次的通用调度支持。

本轮任务全部成功，没有触发新的 MEMORY_REVIEW，
不能将其视为真实 Memory 路径已完成验收。

Skill Final Holdout 未在本阶段使用。

## 七、阶段结论

完成了统一在线任务执行、反馈分流、Skill 低效诊断、
结果缓存和持久化预算组件的工程集成。

第二轮六个 Train 任务成功率为 100%，但所有任务均使用
未变化的 Memory v000 和 Skill None，因此不能归因于自进化。

本阶段最主要的成果是在线系统的可执行性、可审计性、
保守决策与断点恢复能力，而非新的算法性能提升。
