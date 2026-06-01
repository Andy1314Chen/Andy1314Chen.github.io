---
title: "250 行配置赢得 MLSys 竞赛：Harness 工程实践"
date: 2026-06-01
tags:
  - "Harness"
  - "MLSys"
  - "triton"
  - "GPU-Kernel"
  - "Claude-Code"
description: "Dogacel 用 250 行配置文件在 MLSys 2026 FlashInfer 竞赛拿到 DSA 赛道冠军，34.93x 加速比。这篇文章拆解他的 harness 工程设计——约束层、执行层、停滞处置和记忆系统如何配合，让 Agent 自主优化到超越专家水平。"
toc: true
draft: false
---

# 250 行配置赢得 MLSys 竞赛：Harness 工程实践

MLSys 2026 FlashInfer AI Kernel Generation Contest，DSA 赛道。

Dogacel（Doğaç Eldenk）拿到双料第一：Full-Agent 模式 34.93x over baseline，Agent-Assisted 同样第一。34.93x 是所有参赛方案中单 kernel 最高加速比，最终延迟 0.010ms，跑在 NVIDIA Blackwell B200 上。

单人团队。一个人、一个 Claude Code、一个 Modal 账号、不到 250 行配置文件。

这篇文章不讲 kernel 怎么写。要拆的是 harness——不是让 Agent 变聪明，而是设计一套机制管住它。

---

## 一、问题：Agent 会反复踩同一个坑

让 Agent 自主优化 kernel，流程很自然：给 baseline、给 benchmark、让它自己迭代。实际跑起来会卡在一个地方——它在同一个方向上反复尝试。`NUM_WARPS=8` 不行试 `4`，不行试 `16`；`.cg` on K loads 不行试 on Q loads，不行试 on partial stores。每一步改动单独看都合理，但整个方向已经到头了。

原因不少。对话上下文积累了错误的推理路径和过时的直觉；它不会主动从历史记录里系统性提取教训；它倾向于提前宣布"这个方向已经到头了"——然后又换一个方向重复同样的模式。更根本的是，缺一个机制让它在瓶颈时停下来换视角。

Dogacel 的 harness 对这些问题逐个设计了约束。

---

## 二、约束层：CLAUDE.md

CLAUDE.md 放在工作目录下，Claude Code 启动时自动加载。不是使用指南，是宪法——只划边界。

```markdown
# CLAUDE.md

## Non-negotiable rules

- Stay in Triton. No CUDA, no language switching.
- Absolute latencies only. Speedup ratios lie: reference latency swings 20-30%
  across Modal VMs.
- One optimization per iteration. Coupled changes misattribute wins.
- No GPU locally. All compile + benchmark through Modal.
- Log every experiment via /log-experiment, including failures.
- Never stop the loop. Only the user ends optimization.
- No benchmark gaming. No memoizing outputs, no CUDA graph stuff.
- No web search. You are in an isolated environment.
- Don't ask anything to the user. You are designed to work autonomously.
```

每条规则对准一个具体问题：

| 规则 | 解决什么 |
|------|---------|
| Stay in Triton | 避免在语言间无意义切换消耗迭代 |
| Absolute latencies only | Modal VM 间波动 20-30%，加速比可以造假 |
| One optimization per iteration | 一次改两个东西，快了不知道谁贡献的 |
| No web search | 强制从 benchmark 反馈学习，不走外部知识捷径 |
| Never stop the loop | Agent 天然倾向提前宣布完成 |
| No benchmark gaming | CUDA graph / memoize 让 benchmark 显示假加速 |
| Don't ask anything | Agent 可能用"问用户"来拖延或转移决策 |

防作弊有两层。"No web search" 是行为约束，Agent 被要求不搜索。`settings.json` 里还有物理层闸门：网络白名单只开放 Modal API 域名，想搜也做不到。

---

## 三、执行层：三个 command

### /optimize — 主循环

`optimize.md` 定义了 8 步迭代，每步是自然语言指令：

```
1. Assess   → 读 sparse_fused.py + baseline + summary.md + LESSONS.md
2. Plan     → 一个改动。先结构优化，再微调。扫 summary.md 避免重复
3. Implement→ 编辑 solution/triton/sparse_fused.py
4. Validate → /benchmark quick（2 workloads：最小+最大）
5. Measure  → /benchmark stride 2（~10 workloads，2-3 分钟）
6. Log      → /log-experiment（含失败）
7. Decide   → clear win(≥5%) / marginal / regression → keep or revert
8. Budget   → stride 2 为默认，full 仅在确认 new best 时
```

15 分钟调度一次（Claude Code 内置 `/loop` command）。Session 单线程，上一次跑完才开始计时下一个 15 分钟。Stride 2 约 2-3 分钟，远在窗口内；full benchmark（10-15 分钟）大约每 5 轮跑一次。

评测分三级，不是每次跑全量：

| 级别 | Workloads | 耗时 | 用途 |
|------|-----------|------|------|
| quick | 2 | ~30s | correctness 验证 |
| stride 2 | ~10 | 2-3min | 日常 ranking 信号 |
| full | 128 | 10-15min | 确认 new best / drift check |

这里需要解释 Modal：serverless GPU 云平台，参赛者用它跑 B200（数据中心 GPU，不零售）。每次 `modal run` 是全新 VM，按秒计费、用完销毁。两次 benchmark 可能跑在不同物理机上，reference latency 波动 20-30%。这就是 "Absolute latencies only" 的根本原因。

Decide 步骤用精确阈值：clear win ≥5%（stride 2 上跨 VM 噪声约 2-3%，5% 以上才可靠），marginal 时调 A/B 对比脚本消除跨 VM 噪声，regression 直接 revert。

### /benchmark 和 /log-experiment

`benchmark.md` 定义三级评测的用途和参数。`log-experiment.md` 定义强制记录格式：

```
experiments/exp_N/
  ├── sparse_fused.py    ← 该 iter 的 kernel 副本
  ├── bench.log           ← Modal 原始输出
  └── result.md           ← 结构化：Description / Pass / Latency / Learnings
experiments/summary.md    ← 一行追加：Exp / Date / Description / Latency / Pass / Notes
experiments/LESSONS.md    ← 跨 iter 持久化教训
```

核心约束：never skip, even on failures or ablations。失败的 experiment 往往比成功的更有价值——它们标记了"不要重复这条路"。

---

## 四、停滞处置：plateau 检测与 research sub-agent

### 什么是 plateau

kernel 优化的性能曲线通常这样：前几个 iter 下降明显（结构优化在起作用），然后斜率趋近于零。Agent 反复尝试同一方向的变体，每个改动单独看合理，但方向已经穷尽了。

### 触发条件

触发逻辑写在 `optimize.md` 的 step 7（Decide）中。Agent 读完 benchmark 结果后查 `summary.md`，自己判断：

```
Research-agent triggers when any:
- True plateau: 5+ experiments within 5% of each other
- Correctness wall: 3 consecutive correctness failures
- About to repeat failure: summary.md shows this attempt already died
- Out of ideas on the current axis

但 NOT a trigger: three honest micro-wins in a row on the same axis
（tile tuning 确实能做到，同一方向上连续小胜是合法进度）
```

5 这个阈值是综合三个约束的经验值：VM 噪声 × tile tuning 空间 × 时间成本。

### 为什么用 sub-agent 而不是让主 Agent 继续

因为上下文污染。主 Agent 在连续 plateau 时，对话上下文里已经积累了错误推理路径、过时的直觉、以及对当前方向的沉没成本偏见。

Claude Code 的 `Task` 工具调用 sub-agent 时，创建全新的独立 session，不继承任何对话记忆。research agent 的 prompt 第一句：

> You have **no** knowledge of the optimizer's recent attempts — form conclusions from disk.

它只读磁盘文件，不读主 Agent 的对话历史。和 skill（自动追加到当前上下文）是本质区别：skill 提供"知识"，sub-agent 提供"干净的视角"。

### 调用过程

```
exp_38~42 连续 5 个 iter 全部 revert，delay 在 0.016ms 附近
      ↓
主 Agent step 7: 读 summary.md → 判断 plateau 触发
      ↓
调用 Task(subagent_type="research",
         prompt="Read experiments/ and diagnose. Write plan.")
      ↓
[Claude Code 创建独立 session，注入 research.md]
      ↓
[research agent 运行]
  → 读 CLAUDE.md, summary.md, LESSONS.md, workload_profile.md, profile.md
  → 执行 9 项故障诊断
  → 写 experiments/exp_43/plan.md
  → 不编辑 kernel 代码
      ↓
[返回 plan.md 路径给主 Agent]
      ↓
主 Agent 下一轮 /optimize step 1: 读到 exp_43/plan.md
  → 有 plan.md 无 result.md → reserved folder → 按 plan 执行
  → 执行完后 /log-experiment 写 result.md
```

两个 agent 之间的协调只有文件存在性：`plan.md` + 无 `result.md` = 待执行，有 `result.md` = 已完成。不需要消息队列、不需要状态机。

### 故障诊断清单

research agent 逐项排查 9 个问题，每项 cite 具体的 experiment number：

```
1. Repetition loop   — 变着花样重复同一个 idea
2. Local minimum     — 5+ experiments, <5% gain each, same design
3. Correctness wall  — 连续 correctness 失败，数值/算法问题
4. Wrong bottleneck  — compute-bound 上做 memory 优化（或反之）
5. Missing fundamental — online softmax、flash-decoding 等标准技术还没试
6. Over-engineering  — 复杂度高到阻塞进一步优化
7. Ignored prior plan— 之前 plan 建议没被执行
8. Buffer persistence— 漏掉 trivial 的 buffer 复用
9. Overlooked shortcuts — workload 分布让 kernel 有 trivial 路径
```

---

## 五、记忆层：LESSONS.md

LESSONS.md 是整个方案的知识系统。54 个 experiment 产出 50+ 条 lesson，分三类：

反面教材型——标记失败路径：
```
- Never pre-scale bf16 Q by a scalar via cast→mul→cast round-trip.
  The bf16→fp32→bf16 truncates 7-bit mantissa → abs_err ~3e-2.
- NaN landmine in online softmax: exp2(-inf - -inf) = NaN.
  Guard with m_new_safe before exp2.
```

机制解释型——理解为什么，从 experiment 中推理出来：
```
- Halving HBM byte volume doesn't help when intermediate is L2-resident.
  B200 has 126 MB L2. Short-lived back-to-back producer/consumer
  keeps intermediates in L2.
- Host-level input-characteristic dispatch is free.
  A single `if num_tokens <= 2:` branch adds zero kernel overhead.
```

精确数值型——当前最优值，可能因结构改动失效：
```
- NUM_SPLITS sweet spot is 8 for this problem.
  Going 8→16 regresses +33-50%.
- num_warps=8 is strict optimum for H=16.
  num_warps=4: cross-head serialisation. num_warps=16: wasted MMA tile rows.
```

每条 lesson 的形成过程：假设 → experiment → 观察到意外 → 推理机制 → 形成 lesson → 后续 experiment 检验。科学方法，不是外部知识注入。

---

## 六、优化轨迹

从 0.100ms 到 0.012ms，54 个 experiment：

```
exp_1   0.100 ms  →  纯 Triton fused, baseline
exp_2   0.024 ms  →  split-K (4.2×!)   ← 最大单次收益
exp_6   0.022 ms  →  dynamic loop bound, 88% padding exploitation
exp_7   0.021 ms  →  D-parallel combine
exp_11  0.016 ms  →  hybrid dispatch: fused(T≤2) + split(T≥3)
exp_15  0.016 ms  →  atomic-barrier fused launch (-6.7%)
exp_18  0.016 ms  →  volatile load 替代 atomic poll
exp_26  0.016 ms  →  stride-partition (outlier -26%)
exp_37  0.016 ms  →  monotonic counter (-2.9% on all T≥3)
exp_48  0.016 ms  →  evict_last on Q (multi-reader L2 sharing)
exp_51  0.012 ms  →  NUM_SPLITS 8→16 (-16~-19.5% on T=7/8)
```

结构优化优先于微调（exp_2 的 split-K 是 4.2×，exp_51 的 NUM_SPLITS 调整是 16-19%）。hybrid dispatch 对 Agent 来说几乎是免费的——T≤2 和 T≥3 走不同 kernel，人写两套维护成本高，Agent 多写几行几乎零成本。收益递减但非零：exp_37 在第 36 个 iter 后依然找到 2.9% 的提升。

---

## 七、为什么 harness 在 kernel 优化中有效

harness 没有让 Agent 变聪明。它做的事是放大任务本身的优势，弥补 Agent 的天然短板。

**任务侧**——kernel 优化天然适合这种模式。单一客观指标（延迟），秒级反馈（stride-2 只需 2-3 分钟），改动边界清晰（一个 kernel 函数），correctness 可自动化（和 PyTorch reference 数值对比），profiling 给 actionable 反馈（NCU 会告诉你具体瓶颈在哪，不是说"代码太慢"）。Agent 在预训练时已经内化了 tiling、shared memory、tensor core、warp specialization 这些概念，harness 提供正确的反馈信号让它逐一尝试找组合。

**harness 侧**——每个设计对准一个短板：

| harness 设计 | 解决什么 |
|-------------|---------|
| stride-2 快速评测 | Agent 需要快 feedback 才能持续探索 |
| absolute latency 指标 | VM 噪声会让 speedup 造假 |
| LESSONS.md + summary.md | Agent 会遗忘之前试过什么 |
| stall detection → research agent | Agent 会在 plateau 反复微调 |
| never stop the loop | Agent 倾向提前宣布完成 |
| no web search + 沙箱 | Agent 可能用外部知识走捷径 |
| one optimization per iteration | 耦合改动无法归因 |

不是所有任务都适合 harness。通用软件工程里，需求模糊、反馈慢（一次 CI 跑 20 分钟）、信号噪、改动波及面大。kernel 优化成功的前提是任务本身已经提供了正确激励——快 feedback、清指标、可验证。harness 不需要凭空造出这些，只需要不破坏它们。

---

## 参考

- 仓库：https://github.com/Dogacel/auto-gpu-kernel
- 技术报告：https://github.com/Dogacel/auto-gpu-kernel/blob/main/report.pdf
- 竞赛官网：https://mlsys26.flashinfer.ai