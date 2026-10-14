# ③ Agnes 模型参与核心任务的证据

**模型：`agnes-3.0-flash`**

本文件回答一个问题：**Agnes 模型在本项目中承担了什么，证据是什么。**

---

## 一、角色定位

`agnes-3.0-flash` 在本项目中是 **teacher（教师）**，其产出是**全部监督信号的来源**。

CHTD 方法的定义是：教师与学生跑在同一条字节冻结的轨道上（AGH preset `minimal-rl`），
唯一变量是背后的模型；然后让**学生先跑**，教师在**学生的真实岔口**上给出决策作为训练标签。
因此"教师模型"不是辅助工具，而是**训练数据的生产者**——没有它就没有训练集。

| 项 | 内容 |
| --- | --- |
| 模型名称 | `agnes-3.0-flash` |
| 在 AGH 中的 route | `teacher` |
| 调用协议 | OpenAI 兼容（`openai-completions`） |
| 接入地址 | `https://api.agnes-ai.cn/v1` |
| 凭据注入 | `secret://teacher/default`（AGH 密钥引用，不落盘、不入库） |
| 参与环节 | ① 教师自身轨迹采集（baseline）② 学生岔口处的决策标注（fork 评审） |
| 对应 AGH 事件 | `session/start` 的 `modelSettings` 声明 `teacher/agnes-3.0-flash`；`x/core/model-switch` 记录切换到该 route |

---

## 二、证据

### 证据 1：执行记录（账本中的模型声明）

普查全部 1,123 个账本，按 route/model 统计会话启动次数：

| route / model | 会话启动次数 |
| --- | ---: |
| **`teacher/agnes-3.0-flash`** | **1,714** |
| `teacher/deepseek-v4-pro` | 31 |
| `teacher/qwen3.5:4b` | 5 |
| `teacher/agnes-2.5-pro` | 4 |

来源：[`1-agh-execution-census.json`](1-agh-execution-census.json) → `model_invocations_by_route`

> `agnes-3.0-flash` 占教师侧调用的 **97.7%**（1714/1754）。其余模型为早期探索或其他用途。

### 证据 2：实际执行链

[`2-tool-call-chains/chain-b.json`](2-tool-call-chains/chain-b.json) 是 `agnes-3.0-flash` 的
**真实执行链**：列目录 → 并行读两个文件 → 给出诊断。逐事件记录在案，含工具参数与返回内容。

```
 3  model-switch  → teacher/agnes-3.0-flash
17  tool/call      ls   {"path": "."}
21  tool/result    ls   [成功]  average.py \n test_average.py
35  tool/call      read {"path": "average.py"}
36  tool/call      read {"path": "test_average.py"}
41  tool/result    read [成功]  '1  """A small statistics helper.""" … def mean(values): …'
44  tool/result    read [成功]  '1  from average import mean … def test_mean_of_three(): …'
57  assistant      "The defect is in the test file, not the implementation: …"
```

### 证据 3：它产出的训练数据集

| 数据集 | 行数 | 内容 | 来源 |
| --- | ---: | --- | --- |
| `teacher-demo` v1 | 33 | 每任务 1 条完整演示 | 教师自身轨迹 |
| `teacher-demo` v2 | 63 | 34 任务多条 | 教师自身轨迹 |
| `teacher-demo` v3 | 83 | 34 任务多条 | 教师自身轨迹 |
| `teacher-demo` **v5** | **93** | 34 任务，含 1/2/3 次尝试 | 教师自身轨迹 |
| `teacher-state` | 9 | 教师决策落在**教师自己的状态**上（隔离对照臂） | fork 评审 |
| `chtd`（学生状态） | 9 | 教师决策落在**学生的岔口**上（CHTD 主臂） | fork 评审 |

每行样本的 `messages` 即为 `agnes-3.0-flash` 的真实轨迹，`teacher_turn_indices` 标出其决策位置。

来源：`/mnt/proj/chtd-experiments/teacher-demo-20261012/samples-v5.jsonl`（私有存档），
普查见 [`4-config-and-verification.md`](4-config-and-verification.md)。

### 证据 4：因果证据 —— 它的数据训练出的模型性能

这是最强的证据形式：**用 Agnes 模型产出的数据训练学生模型，学生在留出集上的表现显著提升。**

| 对照 | 训练数据 | 失败数（15 题，三级量表判读） |
| --- | ---: | ---: |
| **未微调基线** | 0 行 | **5/15**（33%） |
| v1 臂 | 33 行（Agnes 产出） | 2/15 |
| v2 臂 | 63 行（Agnes 产出） | 0/15 |
| **v5 臂** | **93 行（Agnes 产出）** | **1/15** |
| CHTD 臂（学生状态） | 9 行（Agnes 决策） | 2/15 |

- 失败的定义：答案判定"代码没问题"、归咎于测试、或编造出题目中不存在的机制——即**完全错误的结论**。
- 六个独立训练配置全部落在 0–2/15 区间；在观测基线失败率下，
  该结果的联合概率 **≤ 3.4 × 10⁻⁹**（最保守口径 2.6 × 10⁻⁵）。

来源：[`4-config-and-verification.md`](4-config-and-verification.md) 与
`runs/holdout-4-20261014/dose-ladder-20261014.md`。

### 证据 5：数据完整性门控

Agnes 产出的演示并非照单全收，而是通过三条同时成立才进入训练集：

| 门控 | 判据 | 实测淘汰 |
| --- | --- | --- |
| 诊断正确 | `check.py` 断言通过 | — |
| **无工具调用被拒或失败** | 账本中该会话无 `isError` 的工具结果 | 首轮 35 条中淘汰 12 条 |
| 至少一次读取成功 | 至少一个 `read` 返回真实文件内容 | — |

> 链 C（[`2-tool-call-chains/chain-c.json`](2-tool-call-chains/chain-c.json)）正是被该门控淘汰的
> 典型样本：模型先尝试工作区外路径、被拒后自我纠正。若放任这类轨迹入训，会教出错误习惯。

---

## 三、口径说明（重要）

**`agnes-3.0-flash` 是本项目的教师侧数据生产者，学生侧是可替换的小模型
（`qwen3-4b-instruct-2507`）。**

这个分工是 CHTD 方法本身的设计要求，而非限制：方法的命题是
「学生在自己到达的状态上接受教师指导，比学完整教师轨迹更省数据」，
因此两侧**必须**是不同模型，否则无法测量"迁移"。

**模型的替换方式**：本项目已把模型进入流水线的每个位置参数化
（环境变量 + CLI 参数，默认值为记录中实际使用的值），
详见 [`docs/running-with-your-own-models.md`](../running-with-your-own-models.md)。
他人可用自己的教师/学生模型复现整条流程。

---

## 四、核验路径

```bash
# 1. 查看模型调用普查
cat docs/evidence/1-agh-execution-census.json | python -m json.tool | head -40

# 2. 查看 Agnes 模型的真实执行链
cat docs/evidence/2-tool-call-chains/chain-b.json | python -m json.tool

# 3. 重算证据（需完整工作副本）
python eval/build_evidence.py
```
