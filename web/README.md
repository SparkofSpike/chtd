# CHTD Web Console（Web 演示端）

为 **CHTD（Co-Harness Targeted Distillation，同轨定向蒸馏）** 提供可运行、可现场演示、可复现的
Web 端。它真实呈现「**任务规划 → 工具调用 → 执行反馈 → 结果验证 → 异常处理**」完整闭环，并用
`runs/`、`data/` 中的**真实数据**可视化已完成的对照实验。

> 本目录是给比赛评委与开源读者的作品本体。所有实验数字均可溯源到具体文件；找不到的标「暂无数据」，
> 不编造、不 mock 假轨迹。实施契约见 [PLAN.md](PLAN.md)；5 分钟逐页讲稿见 [DEMO_SCRIPT.md](DEMO_SCRIPT.md)。

---

## 1. 两种运行模式

| 模式 | 何时用 | 数据来源 | 界面标注 |
| --- | --- | --- | --- |
| **Replay（默认）** | 无模型 / 现场离线 | 读取 `runs/`、`data/` 真实 ledger（records.jsonl + sessions.db） | 顶栏 Replay 徽标 |
| **Live** | 有 agnes CLI，且至少一条路由就绪 | 后端真实起一个 AGH 进程，SSE 推送运行中的轨迹 | 顶栏 Live 徽标 |

- **Live 是真的执行**（见 §8）：模型自己读文件、调工具、给结论；轨迹随运行实时出现，跑完由项目
  自己的 `check_diagnosis` 判分。**不需要 GPU** —— teacher 路由是一条托管 API。
- **Live 不可用时自动降级 Replay**：事件流会发 `mode_fallback` 说明原因，界面徽标保持 Replay。
- 顶栏右侧的**路由健康灯**每 5 秒探测一次：哪条路由就绪、用的哪个模型。

---

## 2. 目录结构

```
web/
├── PLAN.md              # 冻结契约（信息架构 / 数据来源 / API / 轨迹重建 / Live）
├── README.md            # 本文件
├── DEMO_SCRIPT.md       # 5 分钟逐页讲稿
├── start.ps1 / stop.ps1 # 一键启动 / 停止
├── demo-data/           # 随仓库附带的 Replay 数据（真实 ledger 快照，见 §7）
├── backend/             # FastAPI（独立 venv，只读解析 + HTTP 转发，不加载模型）
│   ├── app/             # 主程序、loaders、routers、live
│   ├── data/            # export_data.py 导出的只读 JSON
│   └── tests/           # pytest
└── frontend/            # Vite + React + TS + Tailwind + Recharts（构建产物 dist/）
```

页面（6 个）：**Overview**（问题→三承诺自绘 SVG→数字卡）、**Arena / Replay**（任意两臂同题并排、
逐步轨迹、错误恢复标记、track hash）、**Live**（真实运行：选任务→跑→实时轨迹→判分）、
**Experiments**（真实数据图表，每图来源脚注 + 支持/不支持）、
**Reproduce**（端点/rollout/eval 复现命令）。

### Arena 的臂选择

左、右两臂可任选，默认 `base` ↔ `CHTD · seed1`：

| 臂（`arm_group`） | 通过率 | 说明 |
| --- | ---: | --- |
| `base` | 34/48 | 未微调基线 |
| **`chtd-seed1`** | **38/48** | 学生状态臂（seed1 的两个分片已合并） |
| `chtd-seed0` | 7/48 | **该种子在本任务集出现策略退化**，见下 |
| `control-seed0` / `control-seed1` | 34/48 | 教师自轨迹对照臂 |

`chtd-seed0` 的退化是本项目记录在案的发现（`runs/holdout-4-20261014/README.md`）：它始终以
`find test*.py` 开局而不先列目录，假设了不存在的 `src/`，48 题中 29 题因此回答"目录不存在"。
选到该臂时界面会给出提示条 —— 它保留在可选列表里正是因为它是一个真实结果，不是因为它是代表结果。

---

## 3. 环境要求

- Windows + PowerShell。
- **Python 3.14.7**（后端建独立 venv；仅装轻量 Web 包，不装 torch/transformers）。
- **Node v22.23.2 + npm 10.9.8**（前端依赖本地打包）。
- 后端服务端口 **8088**；模型端点默认 base **127.0.0.1:8000**、adapter **127.0.0.1:8001**。

---

## 4. 一键启动（推荐）

在 PowerShell 中：

```powershell
cd <repo>\web
.\start.ps1
```

`start.ps1` 会：① 建后端 venv 并安装轻量依赖；② 若 `backend/data/*.json` 缺失则运行
`export_data.py` 导出真实数据；③ 若前端 `dist/` 缺失则 `npm install` + `npm run build`；
④ 启动 uvicorn（托管前端 dist）。

启动后浏览器打开： **http://127.0.0.1:8088**

停止：

```powershell
.\stop.ps1
```

---

## 5. 手动启动（分步）

### 5.1 后端

```powershell
cd <repo>\web\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python export_data.py            # 生成 data/*.json，终端逐项打印来源
uvicorn app.main:app --host 127.0.0.1 --port 8088
```

### 5.2 前端

```powershell
cd <repo>\web\frontend
npm install
npm run build                    # 产物 dist/，由后端静态托管
# 开发模式（可选，热更新；/api 代理到 127.0.0.1:8088）：
npm run dev
```

> 前端 dist 由后端在根路由 `/` 托管（含 SPA fallback）。若 dist 尚未构建，后端仍正常启动并给占位页，
> 不影响 API 与 Replay 数据接口。

---

## 6. 镜像 / 离线排错

- **pip 慢或失败**：用清华镜像
  ```powershell
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- **npm 慢或失败**：`frontend/.npmrc` 已设 `registry=https://registry.npmmirror.com`；也可手动：
  ```powershell
  npm config set registry https://registry.npmmirror.com
  ```
- **PATH 里的 npm 损坏/缺失**（如报找不到 `cache/policy.js`）：给 start.ps1 指定可用的 npm 绝对路径：
  ```powershell
  $env:CHTD_NPM = "D:\keyapp\npm.cmd"   # 换成你机器上的 npm.cmd 路径
  .\start.ps1
  ```
- **离线演示**：前端所有依赖（含 Recharts、字体）均本地打包，**无外网 CDN/字体外链**；
  后端只读本地文件。首次完成 `export_data.py` 与 `npm run build` 后，断网可完整演示。
- **执行策略阻止脚本**（如 `无法加载 start.ps1，因为在此系统上禁止运行脚本`）：
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\start.ps1
  ```
- **端口被占用**：确认没有其它进程占用 8088（或先 `.\stop.ps1`）。

---

## 7. 数据来源（可溯源）

**Replay 所需的数据随仓库附带**在 `web/demo-data/holdout-4-20261014-v3/`（约 3.4 MB 压缩 / 19 MB
展开），因此克隆仓库即可离线演示，无需先准备完整工作副本。后端路径解析优先用这份快照，
找不到时回落 `data/holdout-4-20261014-v3/`。

后端在启动时只读解析（不重跑、不修改）：

| 内容 | 位置 |
| --- | --- |
| 逐步任务汇总 records.jsonl（11 个臂目录） | `web/demo-data/holdout-4-20261014-v3/rollouts/<arm>/` |
| 精细事件流（规划/工具/结果/恢复） | `web/demo-data/holdout-4-20261014-v3/agh_home/data/sessions.db`（表 events） |
| 实验汇总（剂量/失败率/池化/确定性） | `runs/holdout-4-20261014/` |
| 评分/校准/功效 | `runs/holdout-3-20261014/` |
| 工具健康（教师态） | `runs/teacher-state-20261014/tool-call-census.json` |
| 预置任务定义 | `harness/tasks/*.jsonl` |

导出的每个 `backend/data/*.json` 顶层含 `_meta`（生成时间、CHTD_ROOT、来源文件 `sources`、加载器版本），
每个数字因此可在其所在模块的 `_meta.sources` 与来源脚注中溯源到来源文件 + 复现脚本。

### 7.1 交付前的三处修正（2026-10-15）

本次接入仓库时对回放做了三项修正，均可在界面上验证：

1. **模型名按实际执行显示。** 账本的 `session/start` 声明的是 profile 默认 route（`teacher/agnes-3.0-flash`），
   但 harness 随即发出 `x/core/model-switch` 切到 `student/qwen3-4b-instruct-2507` —— 真正作答的是后者。
   现在回放会跟踪该事件并显示实际模型，同时保留初始声明供对照。
2. **第三方机器路径脱敏。** 工具结果里的 ENOENT 会带上完整路径，现在统一折叠为 `<workspace>` / `<home>`，
   保留有意义的尾部（仍能看到是哪个相对路径失败）。
3. **分片合并。** seed1 臂由两个互不重叠的分片组成（33 + 15 题），现在按 `arm_group` 合并为一个臂，
   同一任务在任一臂缺失时仍能找到配对的另一臂。

---

## 8. Live：真的跑起来（不是骨架）

Live 不是“到不了就降级”的占位符：`POST /api/live/run` 会在 `web/backend/.live/runs/<id>/` 里
建一个临时 AGH home，真实起一个 agnes CLI 子进程，并在它写 ledger 的同时轮询那张表 —— 所以页面上
的轨迹是**边跑边长出来的**，不是跑完才一次性回放。

一次运行需要三样：**agnes CLI**、**node**、**至少一条就绪的路由**。`GET /api/live/capabilities`
会告诉你缺哪一样，Live 页顶部也直接列出来。

### 两条路由

| 路由 | 是什么 | 就绪条件 |
| --- | --- | --- |
| `teacher` | 托管 API（默认 `https://api.agnes-ai.cn/v1`） | 有凭据 —— **不需要 GPU** |
| `student` | 你自己服务的 OpenAI 兼容端点 | 该端点 `/v1/models` 答 200 |

`student` 路由指向哪由你定：页面上展开「模型配置」就能改 **路由 / 模型 id / baseUrl / 密钥 / 超时**，
不必改代码。任何能答 `/v1/models` 与 `/v1/chat/completions` 的端点都行（本项目用
`serve_openai.py` 服务 Qwen3-4B，但页面不假定这一点）。

### 凭据从哪里来

按优先级：请求体里的 `api_key` → 环境变量 `AGNES_SECRET_<ROUTE>_DEFAULT` → 密钥文件
（`CHTD_LIVE_KEY_FILE`，默认 `data/chtd-round-20261011/teacher.key`）。

从页面填的密钥只会进入那一次运行的子进程环境：**不写进 run 目录、不回显、不记日志**。

### 判分

跑完由本项目自己的 `harness/tasks/check_diagnosis.py` 打分 —— 就是产出历史 holdout-3 分数的
那个脚本（短语匹配 + 反例拒绝 + 虚构文件检查），所以 Live 的成绩与记录里的成绩是同一把尺。
注意 holdout-3 任务里那个 `check.py` 是 **fixture 的一部分**（给模型看的证据），不是判分命令；
Live 不用它。

### 环境变量

| 变量 | 缺省 | 作用 |
| --- | --- | --- |
| `CHTD_AGNES_CLI` | `E:\Coding\agnes-harness\...\agnes.mjs` | agnes CLI 入口 |
| `CHTD_NODE` | `node` | node 可执行文件 |
| `CHTD_LIVE_ROUTE` / `CHTD_LIVE_MODEL` | `teacher` / `agnes-3.0-flash` | 默认路由与模型 |
| `CHTD_LIVE_BASE_URL` | `https://api.agnes-ai.cn/v1` | 默认端点 |
| `CHTD_LIVE_PRESET` | `minimal-rl` | 冻结轨道 |
| `CHTD_LIVE_TIMEOUT` | `240` | 单次运行上限（秒） |
| `CHTD_LIVE_KEY_FILE` | `data/chtd-round-20261011/teacher.key` | 服务端凭据文件 |

### 可跑哪些任务

`GET /api/live/tasks` 列的是入库任务集（`harness/tasks/holdout-3.jsonl` 等）里**本机有 fixture**
的那些：带 fixture 才能跑（少 fixture 的在页面上是禁选项，不会被静静跑成一个模棱两可的失败）。
也可以选「自定义 prompt」跑一个任意任务 —— 那种运行不给分，只给答案与完整轨迹。

### 克隆下来就能跑（实测）

`data/` 不进仓库（它是生成物），但 fixture 会回落到入库的 `harness/tasks/<set>/<task_id>/`。
2026-10-15 在一个只含入库文件的导出树里实测：45 个任务列出、30 个带 fixture；选一个用自备凭据
跑通（4 次工具调用，exit 0，判分通过）。

**凭据是唯一本机没有的东西**，界面会直说缺它，并在「模型配置」里让你填自己的。

### 红线不变

Live 的临时产物只写到 `web/backend/.live/`；fixture 是**拷进** run 目录再跑的，不写回
`data/`、`runs/`、`harness/`。

---

## 9. 测试

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest -q
```

覆盖：加载器（存在文件解析、缺失文件标「暂无数据」、provenance 非空）、轨迹重建（工具调用/结果配对、
错误恢复正例/反例）、API（TestClient 校验 health/experiments/sessions/tasks 与 Arena SSE 帧）。

---

## 10. 合规与红线

- 只新增 `web/`；**不修改** `distill/eval/harness/packages` 与 `runs/`、`data/`；不重跑训练评测。
- 不接入 OpenAI/Claude/Gemini 等第三方模型，前端无任何 key。
- 页面并列「证据支持 / 证据不支持」，不夸大（例如不写「全面优于其他方法」；p=0.077 标注为
  「强提示而非证实」；v5 为完整教师演示，不作为 CHTD 形态证据）。
- 改动走 fork-branch → PR → review，不直接 push main。
