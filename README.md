# Agent Self-Evolution：基于强化反馈的大语言模型智能体自进化系统

基于 ALFWorld 环境的大语言模型智能体自进化研究与工程实践项目。

本项目研究如何在**不更新大语言模型参数**的情况下，利用环境反馈、反事实验证、长期记忆管理和可复用技能学习，使 Agent 从历史执行经验中持续改进决策。

系统以 ReAct Agent 为基础，实现了经验记忆自进化（Memory Evolution）、技能自进化（Skill Evolution）及统一反馈驱动的 Agent 运行框架。

当前项目已完成主要工程模块及在线 Pilot 验证，但尚未通过实验确立持续自进化相对于固定能力基线的稳定性能优势。

## 一、项目架构

```text
                   ALFWorld 任务
                        │
                        ▼
                统一 Agent Runtime
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Active Memory          Active Skills
       BM25 Top-K 检索        条件触发与执行
             └──────────┬──────────┘
                        ▼
              ReAct + State Tracking
                        │
                        ▼
                 ALFWorld 环境交互
                        │
                        ▼
                  执行轨迹与反馈
                        │
                        ▼
                  Feedback Router
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    MEMORY_REVIEW  SKILL_REVIEW      ARCHIVE
          │             │
          ▼             ▼
       失败分析       低效轨迹分析
          │             │
       反事实验证     技能候选提取
          │             │
       Lesson 候选    Skill 候选
          └──────┬──────┘
                 ▼
           版本验证与晋升门控
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
     ACCEPT              REJECT
       │                   │
   更新活跃能力        保留原有能力
```

系统采用保守的能力更新策略：

- LLM 负责提出 Lesson、Skill 及修订建议。
- Python 程序负责条件判断、反馈路由、版本管理和晋升门控。
- ALFWorld 环境负责提供真实动作结果和任务成功信号。
- 候选能力必须经过验证，不能仅凭 LLM 判断直接上线。
- 当证据不足时，系统允许 KEEP、REJECT 或 ABSTAIN。

## 二、核心功能

### 2.1 经验记忆自进化（Memory Evolution）

针对 Agent 执行失败的任务，系统实现以下流程：

1. 保存完整 ReAct 执行轨迹。
2. 使用 LLM 分析失败原因并定位关键决策。
3. 提出候选反事实动作。
4. 在真实 ALFWorld 环境中重放历史状态，替换关键动作并继续执行。
5. 根据真实续跑结果提取候选 Lesson。
6. 使用 BM25 检索相关经验，并生成 ADD、EDIT、REMOVE 或 KEEP 更新建议。
7. 创建候选记忆版本，通过配对验证决定是否晋升。
8. 支持版本恢复、拒绝及回滚。

长期记忆检索使用 BM25 Top-K，支持任务类型相关性处理。

State Tracking 用于记录目标物体及 clean、cool、heat 等状态，辅助 Agent 判断任务执行进度。

### 2.2 技能自进化（Skill Evolution）

针对成功但执行效率较低的任务，系统研究如何提取可复用的条件化技能。

主要流程：

1. 分析高成本成功轨迹，识别潜在低效决策。
2. 结合历史反事实实验构建正反案例。
3. 让 LLM 提出包含触发条件、建议动作和安全约束的 Skill。
4. 在运行时由 Python 检查触发条件。
5. 通过在线对照实验验证技能效果。
6. 根据成功率和执行效率决定接受或拒绝技能。
7. 将验证反馈用于修订判断；证据不足时允许 ABSTAIN。

支持两种技能干预模式：

| 模式 | 执行方式 |
|---|---|
| Hint | 程序向 LLM 提供技能建议，由 LLM 决定是否采纳 |
| Execute | 条件满足时程序直接执行一次技能动作，之后恢复 ReAct |

系统保留技能触发事件、实际采纳情况、环境动作数及 Agent 级别的 LLM 调用次数，避免将未触发技能时的随机轨迹变化错误归因于技能。

### 2.3 统一 Agent 自进化框架

在 Memory Evolution 和 Skill Evolution 基础上，实现统一运行与反馈管理模块。

主要组件：

| 组件 | 功能 |
|---|---|
| Active Capabilities | 加载已接受的 Memory 与 Skill，校验版本及证据 |
| Unified Agent Runtime | 统一配置 ReAct、Memory Retrieval、Skill 和 State Tracking |
| Feedback Router | 根据环境反馈将任务分流 |
| Unified Controller | 保存任务来源、轨迹哈希、能力版本和路由事件 |
| Review Queue Consumer | 管理 Memory / Skill 待处理任务及数据使用范围 |
| Online Review | 调用现有分析与反事实验证模块处理在线反馈 |

Feedback Router 使用三种路由：

- `MEMORY_REVIEW`：失败任务，进入失败分析流程。
- `SKILL_REVIEW`：成功但动作成本较高的任务，进入效率分析流程。
- `ARCHIVE`：普通成功任务，保存轨迹但暂不学习。

Validation 和 Holdout 数据不允许进入训练反馈队列。

## 三、实验配置

| 配置项 | 设置 |
|---|---|
| 环境 | ALFWorld / TextWorld |
| 执行模型 | qwen3.8-chat |
| Python | 3.11 |
| Agent | ReAct |
| 经验检索 | BM25 Top-K=3 |
| State Tracking | 开启 |
| 最大决策步数 | 50 |
| 模型参数更新 | 无 |
| 实验管理 | 冻结任务清单、轨迹缓存、SHA256 校验 |
| 能力更新 | 候选版本 + 验证门控 |

不同阶段使用不同的任务集合与实验协议，结果不能直接视为同一组严格等预算对照实验。

## 四、主要实验结果

### 4.1 长期记忆连续自进化

使用 32 条初始经验和 130 条原始经验进行对照。

连续学习实验包含 3 轮、每轮 20 个学习任务，共 60 个任务。

| 方法 | 成功任务 | 成功率 | 平均决策步数 | 最终版本 |
|---|---:|---:|---:|---|
| Fixed-32 | 57/60 | 95.00% | 15.15 | v000 |
| Append-32 | 57/60 | 95.00% | 14.77 | v000 |
| Dynamic-32 | 57/60 | 95.00% | 14.73 | v000 |
| Fixed-130 | 58/60 | 96.67% | 15.02 | v000 |

Append-32 曾生成候选记忆版本，但验证收益未达到预设的晋升门槛。

Dynamic-32 未产生被接受的新版本。

因此，三种以 32 条经验开始的方法最终都保留原始 v000 记忆。

#### 独立留出任务评估

| 指标 | Fixed-32 | Fixed-130 |
|---|---:|---:|
| 成功任务 | 30/30 | 29/30 |
| 成功率 | 100.00% | 96.67% |
| 平均决策步数 | 14.73 | 17.13 |

由于三种 32 条记忆的方法最终使用相同的活跃记忆，Fixed-32 的留出任务结果代表其共同的最终能力配置，并非三次独立评估。

实验表明，在该任务集合上，更多经验不一定带来更好的表现，但不能据此认定 32 条经验普遍优于 130 条经验。

### 4.2 反馈驱动技能学习

从成功但低效的历史轨迹中提出条件化搜索技能：

`stalled_search_countertop_redirect`

候选技能在连续检查多个位置仍未找到目标物体时，建议搜索尚未访问的 countertop。

在线开发实验包含 12 个此前未使用的 ALFWorld Train 任务。

| 指标 | Baseline | Hint | Execute |
|---|---:|---:|---:|
| 成功任务 | 12/12 | 12/12 | 12/12 |
| 平均环境动作数 | 13.92 | 13.17 | 14.83 |
| 技能触发次数 | 0 | 5 | 3 |

Hint 相对 Baseline 平均节省 0.75 步，但没有达到预先设定的平均至少节省 1 步的晋升标准。

Execute 平均增加约 0.92 步，也没有达到晋升标准。

最终结果：

- 候选技能 c001：`REJECTED_ON_DEV`
- 反馈驱动修订：`ABSTAIN`
- 新技能 c002：未创建
- Active Skill：None

这证明系统可以完成技能提出、在线验证、拒绝及修订决策，但没有证明当前搜索技能具有稳定的性能收益。

该技能实验冻结的 30 个 Final Holdout 任务未执行。

详细结果：[技能学习实验报告](docs/skill_learning_results.md)。

### 4.3 统一 Agent 在线 Pilot

基于统一 Runtime 与 Feedback Router，选择六种任务类型各一个新的 ALFWorld Train 游戏进行在线 Pilot。

| 指标 | 结果 |
|---|---:|
| 在线任务数 | 6 |
| 成功任务数 | 5 |
| 成功率 | 83.33% |
| ARCHIVE | 3 |
| MEMORY_REVIEW | 1 |
| SKILL_REVIEW | 2 |
| Memory 晋升次数 | 0 |
| Skill 晋升次数 | 0 |

#### Memory Review

失败任务：

`look at bowl under the desklamp`

Agent 在 50 步限制内未能完成任务。

FailureAnalyzer 提出三个候选反事实动作，其中两个经过真实 ALFWorld 重放验证：

| 替代动作 | 验证结果 |
|---|---|
| examine desk 1 | 失败 |
| take bowl 1 from desk 1 | 失败 |
| inventory | 未验证 |

前两个反事实分支均失败，因此没有提取经过成功验证的 Lesson，也没有更新活跃记忆。

最终状态为 `STOPPED_WITH_UNTESTED_CANDIDATE`。

#### Skill Review

两条成功任务被识别为搜索效率偏低：

- fresh_01：27 步完成加热 cup 并放入 cabinet 的任务。
- fresh_05：23 步完成将 spraybottle 放到 countertop 的任务。

两条任务的主要动作成本集中在目标搜索阶段。

_CANDIDATE_IN_PILOT`。
但�

#### Pilot 结论

统一系统成功完成：

真实任务执行 → 反馈路由 → Review Job 生成 → 轨迹及证据保存 → Memory/Skill 分析 → 无充分证据时保持原有能力。

最终保持：

- Active Memory：v000（32 条）
- Active Skill：None

本次 Pilot 验证的是系统集成与保守处理逻辑，不代表已经证明统一自进化系统能够持续提升任务表现。

详细结果：[统一自进化实验报告](docs/unified_evolution_results.md)。

## 五、环境安装

项目使用 Python 3.11 和 uv 管理依赖。

```bash
uv sync
```

需要提前准备 ALFWorld 数据及实验所需的初始经验文件。

配置兼容 OpenAI API 的模型服务：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-endpoint"
export OPENAI_MODEL="qwen3.8-chat"
```

不要将实际 API Key 提交到 Git 仓库。

## 六、运行方式

### 6.1 运行单元测试

```bash
uv run python -m unittest discover -s tests -v
```

当前已通过 66 项单元测试，覆盖状态追踪、记忆更新、版本恢复、技能执行、反馈路由及统一 Controller 等功能。

### 6.2 运行连续记忆自进化

初始化实验：

```bash
uv run python -m scripts.run_continual_evolution \
  --stage plan \
  --protocol-root outputs/continual_32 \
  --run-name dynamic_32 \
  --mode dynamic
```

运行实验：

```bash
uv run python -m scripts.run_continual_evolution \
  --stage run \
  --protocol-root outputs/continual_32 \
  --run-name dynamic_32 \
  --mode dynamic
```

支持 `fixed`、`append` 和 `dynamic` 三种模式。

使用 `--reference-memory` 可以在 fixed 模式下加载 130 条参考经验。

### 6.3 统一 Agent 在线实验

以下命令适用于已完成初始化、并保留本地冻结协议及缓存数据的实验工作目录。

查看已冻结的在线实验：

```bash
uv run python -m scripts.run_unified_pilot --stage report
```

从缓存恢复任务执行：

```bash
uv run python -m scripts.run_unified_pilot \
  --stage run \
  --limit 6
```

检查在线反馈：

```bash
uv run python -m scripts.review_unified_online --stage report
```

检查反事实验证记录：

```bash
uv run python -m scripts.verify_unified_memory --stage report
```

执行 `run` 前应确认所使用的任务及协议，缺少缓存时可能产生新的模型调用。

## 七、实验数据与复现说明

实验采用固定任务清单、版本快照、SHA256 校验和执行轨迹缓存，尽量减少不同运行之间的配置偏差。

主要实验数据保存在：

```text
outputs/continual_32/
outputs/skill_learning_v1/
outputs/unified_evolution_v1/
```

`outputs/` 默认由 Git 忽略，不会随代码仓库一起上传。

因此，仅克隆 GitHub 仓库不能完整复现本文列出的历史实验结果；还需要对应的 ALFWorld 数据、冻结协议及原始运行记录。

本项目部分历史评估集已经用于多轮方法探索，相关结果应结合其数据使用范围理解，不应全部视为完全独立的最终泛化测试。

## 八、研究结论

本项目实现了一套不依赖大模型参数更新的 Agent 反馈学习与能力管理框架，主要包含：

1. **经验记忆学习**：从失败轨迹及反事实验证中提取 Lesson，并管理候选记忆版本。
2. **条件化技能学习**：从成功但低效的轨迹中提取候选 Skill，通过真实环境对照实验筛选。
3. **统一反馈处理**：自动区分失败任务、低效成功任务及普通成功任务，并保存可追溯的 Review 记录。
4. **保守能力更新**：候选能力必须经过验证，支持拒绝、保持不变及版本恢复，避免将未经验证的经验直接加入活跃系统。

现有实验验证了主要工程流程的可行性，也揭示了反事实成功、局部效率提升与可泛化能力之间的区别。

**目前尚未证明连续 Memory/Skill 自进化能够稳定优于固定能力基线。** 后续研究应在独立任务集合及严格对照协议下验证实际收益。

## 九、研究文档

- [完整实验结果](docs/experiment_results.md)
- [技能学习实验](docs/skill_learning_results.md)
- [统一自进化在线 Pilot](docs/unified_evolution_results.md)

项目持续以真实环境反馈、可验证证据和可复现实验为基础推进。
