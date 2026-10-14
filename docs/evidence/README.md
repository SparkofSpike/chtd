# 运行证据包

**本项目：CHTD — Co-Harness Targeted Distillation（共轨定向蒸馏）**

本目录是参赛要求的「运行证据」四项的**自包含**交付物。全部数据从 AGH 账本（`sessions.db`）
逐事件导出，未经改写；第三方机器路径已脱敏。

| 参赛要求 | 对应文件 |
| --- | --- |
| ① AGH 执行记录 | [`1-agh-execution-census.json`](1-agh-execution-census.json) |
| ② ≥1 条工具/设备调用链 | [`2-tool-call-chains/`](2-tool-call-chains/)（3 条，见下） |
| ③ Agnes 模型参与核心任务证据 | [`3-agnes-participation.md`](3-agnes-participation.md) |
| ④ 关键配置与专业验证结果 | [`4-config-and-verification.md`](4-config-and-verification.md) |

校验和：[`SHA256SUMS.txt`](SHA256SUMS.txt)（`sha256sum -c SHA256SUMS.txt` 可验）

---

## ① AGH 执行记录

`1-agh-execution-census.json` —— 对本仓库 `data/` 下全部 AGH 账本的普查：

| 指标 | 数值 |
| --- | --- |
| 账本数（`sessions.db`） | **1,123** |
| 会话数 | **1,649** |
| 事件数 | **120,066** |

**模型调用分布**（按 route / model）：

| route / model | 会话启动次数 |
| --- | ---: |
| `teacher/agnes-3.0-flash` | 1,714 |
| `teacher/deepseek-v4-pro` | 31 |
| `teacher/qwen3.5:4b` | 5 |
| `teacher/agnes-2.5-pro` | 4 |

按 run 的细分见同文件 `by_run` 字段。

> **为什么是汇总而非账本本身**：`data/` 目录在 `.gitignore` 下（体积约 745 MB，且含大量中间产物），
> 不适合入库。此汇总的每个字段都可由 `eval/build_evidence.py` 重新生成——
> 在完整工作副本上运行该脚本即可复算。

**核验方法**：

```bash
python eval/build_evidence.py      # 重新生成本目录全部文件
```

---

## ② 工具 / 设备调用链

三条链，均从账本逐事件导出，每条含：事件序号、类型、一行摘要、脱敏后的原始载荷。

### 链 A — 学生臂执行链

| 字段 | 值 |
| --- | --- |
| 文件 | [`2-tool-call-chains/chain-a.json`](2-tool-call-chains/chain-a.json) |
| 模型 | **`student/qwen3-4b-instruct-2507`**（CHTD 学生臂，LoRA seed 0） |
| 任务 | `fix-string-case` |
| 判读结论 | **correct** |
| 预设 | `minimal-rl`（字节冻结） |
| 轨道哈希 | `42432bbdbbf0016821…` |
| 事件数 | 61（导出 11 条关键事件） |
| 工具序列 | `ls` → `read` |

**执行过程**：

```
 3  model-switch  → student/qwen3-4b-instruct-2507
 6  user/prompt    "Something in this directory is broken. Read the files and tell me what
                    the defect is. Do not run any commands and do not change any files…"
17  assistant      "I'll help you identify what's wrong with the files in the directory…"
18  tool/call      ls   {"path": "."}
22  tool/result    ls   [成功]  answers.py
35  assistant      "Great, there's only one file in the directory: answers.py. Let's read…"
36  tool/call      read {"path": "answers.py"}
40  tool/result    read [成功]  '1  def is_yes(answer): …'
53  assistant      "The is_yes function in answers.py … returns True if the input is exactly
                    the string 'yes' … it's case-sensitive …"
58  turn/end
```

> 这条链展示了一个**完整、健康**的学生任务回合：列目录 → 读文件 → 定位缺陷。
> 「设备调用」在本项目中即 AGH 的工具调用（`ls` / `read` / `find` / `grep` / `shell`），
> 每次调用都经 AGH 策略层判定只读/破坏性/replay 语义后才执行。

### 链 B — Agnes 教师臂执行链

| 字段 | 值 |
| --- | --- |
| 文件 | [`2-tool-call-chains/chain-b.json`](2-tool-call-chains/chain-b.json) |
| 模型 | **`teacher/agnes-3.0-flash`** |
| 事件数 | 65（导出 11 条） |
| 工具序列 | `ls` → `read` → `read`（**一轮内并行读两个文件**） |

**执行过程**：

```
 3  model-switch  → teacher/agnes-3.0-flash
17  tool/call      ls   {"path": "."}
21  tool/result    ls   [成功]  average.py \n test_average.py
35  tool/call      read {"path": "average.py"}
36  tool/call      read {"path": "test_average.py"}      ← 并行调用
41  tool/result    read [成功]  '1  """A small statistics helper.""" … def mean(values): …'
44  tool/result    read [成功]  '1  from average import mean … def test_mean_of_three(): …'
57  assistant      "The defect is in the test file, not the implementation:
                    - average.py is correct. mean properly guards against an empty sequence…"
```

> 这条链是**全部监督信号的来源**：agnes-3.0-flash 真实执行工具、真实阅读代码、真实给出诊断。

### 链 C — 门控生效与模型自我纠正

| 字段 | 值 |
| --- | --- |
| 文件 | [`2-tool-call-chains/chain-c.json`](2-tool-call-chains/chain-c.json) |
| 模型 | `teacher/agnes-3.0-flash` |
| 事件数 | 210（导出 32 条） |
| 工具序列 | `ls` ×3 → `read` ×2 → `shell` → `ls` → `read` ×2 → … |

**执行过程（节选）**：

```
17  tool/call      ls   {"path": "C:\Users\<redacted>\.claude\skills\test-docs-workflow"}
21  tool/result    ls   [拒绝]  "ls refused: … is outside the workspace. Switch to Full
                                permissions (完全权限) or select the target directory…"
34  tool/call      ls   {"path": "C:\Users\<redacted>"}
38  tool/result    ls   [拒绝]  "ls refused: … is outside the workspace…"
51  tool/call      ls   {}                       ← 模型自我纠正，回到工作区
55  tool/result    ls   [成功]  average.py \n test_average.py
69  tool/call      read {"path": "C:\Users\<redacted>\average.py"}
75  tool/result    read [拒绝]  "read failed: E_FS_DENIED: … is outside every allow rule"
91  assistant      "The listed paths are outside the workspace. Let me find where they
                    actually live."
92  tool/call      shell {"command": "pwd"}
97  tool/result    shell [拒绝]  "the user rejected this action; do not retry the same call"
108 tool/call      ls   {"path": "."}            ← 回到正确的工作区根目录
112 tool/result    ls   [成功]  average.py \n test_average.py
```

> **这条链是 CHTD 门控设计的直接动机。** 它展示了两种行为同时存在：
> ① 模型会尝试访问工作区外的路径——若这类轨迹进入训练集，就会教出"先试外部路径"的错误习惯；
> ② AGH 的 `WorkspaceRelativePaths` 门控**拒绝**了这些调用，模型随后自我纠正。
>
> 在本项目的采集流程中，这类轨迹被**明确排除**：实测首轮 35 条演示中有 **12 条**因
> "首次调用越界被拒"而落选（见 `docs/teacher-demo` 相关记录中的采集报告）。

---

## 关于脱敏

| 项 | 处理 |
| --- | --- |
| 第三方 Windows 用户路径 | → `C:\Users\<redacted>` |
| 第三方 Linux 家目录 | → `/home/<redacted>` |
| 团队自有机器 Tailscale 地址 | 保留（均为团队成员自己的机器；`100.x` 属 CGNAT 网段，无 tailnet 授权不可路由） |
| 凭据 / API key | 全部账本中不存在（已做全历史扫描，见 `4-config-and-verification.md` 的安全审计一节） |

脱敏脚本对每条链的**全部字段**（含工具参数与返回文本）执行替换，并在生成后自检
（`eval/build_evidence.py` 第 3 步），确认无残留。

---

## 复现

```bash
# 重新生成整个证据包（需要完整工作副本，含 data/ 与训练盒导出的链缓存）
python eval/build_evidence.py

# 校验已提交的证据包未被篡改
cd docs/evidence && sha256sum -c SHA256SUMS.txt
```
