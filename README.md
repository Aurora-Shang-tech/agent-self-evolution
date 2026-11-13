# Agent Self-Evolution

**Training-Free LLM Agent Self-Evolution via Counterfactual Feedback and Memory Management**

基于 ALFWorld 的大语言模型智能体自进化研究项目。在不更新模型参数的情况下，利用环境反馈、反事实验证与长期记忆管理，探索 Agent 能否通过持续交互改善决策能力。

## 1. 系统架构

ALFWorld Task
→ ReAct Agent
→ Memory Retrieval + State Tracking
→ Trajectory & Feedback
→ Failure Analysis
→ Counterfactual Verification
→ Lesson Extraction
→ Memory Update (ADD / EDIT / REMOVE / KEEP)
→ Candidate Version
→ Paired Validation
→ Accept / Reject / Rollback

系统依据真实环境的任务完成信号验证反事实结果，而不是仅依靠 LLM 判断成功与否。

## 2. 核心功能

- **ReAct Agent**：执行 ALFWorld 任务，记录完整决策轨迹。
- **Failure Analysis**：定位失败的关键决策，生成替代动作。
- **Counterfactual Verification**：恢复关键环境状态，验证替代动作并保存续跑轨迹。
- **Experience Learning**：从失败与成功反事实中提取候选经验，支持多个成功分支。
- **Memory Management**：支持 ADD、EDIT、REMOVE、KEEP。
- **Memory Retrieval**：BM25 检索与同类型新增经验优先检索。
- **State Tracking**：跟踪 clean、cool、heat 操作及物体实例状态。
- **Version Control**：支持候选验证、版本接受或拒绝、异常恢复和回滚。

## 3. 实验设置

| Configuration | Value |
|---|---|
| Environment | ALFWorld / TextWorld |
| Model | qwen3.8-chat |
| Python | 3.11 |
| Initial Memory | 130 CF-Verified Lessons |
| Learning Tasks | 3 rounds × 24 = 72 |
| Validation Tasks | 12 |
| Final Holdout | 30 |
| Max Steps | 50 |
| Retrieval Top-K | 3 |
| State Tracking | Enabled |
| Parameter Training | None |

对照方法：

- **Fixed**：固定初始记忆，不进行更新。
- **Append-only**：只允许添加新经验。
- **Dynamic**：允许 ADD、EDIT、REMOVE、KEEP，通过验证门控控制更新。

## 4. 实验结果

### Continuous Evolution

| Method | Success | Success Rate | Avg. Steps | Accepted Updates |
|---|---:|---:|---:|---:|
| Fixed | 69/72 | 95.83% | 14.28 | 0 |
| Append-only | 70/72 | 97.22% | 13.71 | 0 |
| Dynamic | 69/72 | 95.83% | 14.69 | 0 |

三组最终均保留原始 130 条经验，活跃版本为 v000，记忆哈希完全一致。

### Final Holdout

| Metric | Result |
|---|---:|
| Successful Tasks | 29/30 |
| Success Rate | 96.67% |
| Average Decision Steps | 12.87 |
| Memory Size | 130 |
| Active Version | v000 |

由于三组最终记忆完全相同，Holdout 仅执行一次共同固定记忆策略的评估，不代表三组独立的测试成绩。

### 结果分析

系统已完成连续任务执行、失败反馈、反事实验证、经验提取和版本控制的端到端工程闭环。

但本次实验没有产生被正式接受的记忆更新，因此尚不能证明 Dynamic 优于 Fixed 或 Append-only。

后续新增的多反事实分支提取逻辑已通过代码回归测试，但不属于上述冻结实验的性能结果。

## 5. 安装与运行

使用 Python 3.11：

    uv sync

配置 OpenAI-compatible API：

    export OPENAI_API_KEY="your-api-key"
    export OPENAI_BASE_URL="your-api-endpoint"
    export OPENAI_MODEL="qwen3.8-chat"

运行测试：

    uv run python -m unittest discover -s tests -v

创建实验：

    uv run python -m scripts.run_continual_evolution --stage plan --run-name experiment --mode dynamic

执行实验：

    uv run python -m scripts.run_continual_evolution --stage run --run-name experiment --mode dynamic

支持 fixed、append、dynamic 三种模式。

完整运行需要提前准备 ALFWorld 数据、初始经验库及冻结实验 Manifest。默认 outputs/ 不提交 GitHub，因此仅克隆源代码不能直接复现上述完整实验。

## 6. 项目结构

- src/agents/：ReAct 执行
- src/analysis/：失败分析
- src/counterfactual/：环境重放与反事实验证
- src/memory/：经验检索、更新与状态追踪
- src/evolution/：连续自进化与版本管理
- src/envs/：ALFWorld 环境封装
- scripts/run_continual_evolution.py：统一实验入口
- tests/：核心功能与版本安全测试
- docs/experiment_results.md：历史研究实验记录

## 7. 研究结论

本项目实现了基于强化反馈的 Training-Free Agent Self-Evolution 工程系统，具备反事实学习、长期记忆更新和受控版本管理能力。

当前实验验证了工程可行性，但尚未获得动态记忆更新优于固定记忆的实证结果。

历史实验详情见 docs/experiment_results.md。部分历史运行脚本已在代码精简过程中移除。
