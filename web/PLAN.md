# CHTD Web Console — MVP 计划（一页要点）

> 目标：**30 分钟内跑通“无 GPU 即可完整演示”的可运行版本**；先跑通后打磨。Live、剩余图表与文档在 MVP 后再补。

## 红线
- 只读：不重跑，不改 `distill/eval/harness/packages`、`runs/`、`data/`；代码只新增到 `web/`。
- 只调 Agnes 模型（serve_openai 的 Qwen），无第三方模型、无 key；Replay 用真实 ledger，不 mock；数字缺失标“暂无数据”。
- 并列“支持/不支持”，不夸大（p=0.077=强提示而非证实；v5 为完整教师演示，非 CHTD 证据）。

## 范围（MVP，三条线并行）
- **后端** `web/backend`（FastAPI，独立 venv；不装 torch/transformers）：
  - 加载/导出（`export_data.py` → `backend/data/*.json`，打印来源）。
  - 接口：`GET /api/health`、`GET /api/experiments`、`GET /api/sessions`、`GET /api/sessions/{key}`、
    `POST /api/arena/run`（SSE）；根路由静态托管 `frontend/dist`（dist 缺失不阻塞启动）。
- **前端** `web/frontend`（Vite+React+TS+Tailwind+Recharts，暗色，一次成型，不精修）：
  - ① Overview：方法三承诺 + 核心结论数字卡。
  - ② Experiments：4 张关键图（Recharts 默认样式），每图来源脚注。
  - ③ Arena/Replay：base vs chtd 并排回放真实轨迹，含成功/失败，显示 track hash。
  - Reproduce：仅静态命令文本。
- **Live**：只保留 `/api/health` 探测 + arena 骨架；请求 live 直接发 mode_fallback（“Live 暂不可用，已切换 Replay”）走真实 Replay；不做子进程编排。

## 4 张图与所需文件（后端只解析这些 + 回放）
| 图 | 数据 | 来源文件 |
| --- | --- | --- |
| 失败率对比 | 15 题 4–5 vs 0–2；57 池化 31.6→17.5 | runs/holdout-4/dose-ladder-*.md、pooled-analysis-*.md；scores-*.json |
| 数据效率 | 9 vs 33 | runs/holdout-4/dose-cells.json、dose-ladder-*.md |
| 剂量曲线 | 33–93 rows 的 ordinal/failures | runs/holdout-4/dose-cells.json、dose-ladder-*.md |
| 确定性 | base 6 次一致 vs adapter 9/15 不同 | runs/holdout-4/dose-ladder-*.md |
| 回放 | 逐步轨迹 | data/holdout-4-v3/rollouts/<arm>/records.jsonl + agh_home/data/sessions.db（表 events）；tool-call-census.json |

回放重建：events 按 seq，`step/start` 分组；`assistant/message`=规划/最终答案；`tool/call`(name,args) ↔ `tool/result`(content,isError)；
isError 后下一步标错误恢复；verifier/signal 给判定。base↔chtd 按 task_id 配对，同步动作/结果不同→高亮分歧。

## 依赖 / 镜像（一次装好）
- 后端：`python -m venv .venv`；`pip install fastapi "uvicorn[standard]" pydantic httpx pytest -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 前端：`.npmrc` 设 `registry=https://registry.npmmirror.com`；`npm install`；`npm run build`。
- 离线：前端本地打包、无外网 CDN/字体外链。

## 启动 / 验证（从简但必须有）
- `web/start.ps1`（建 venv→装包→导出→build→起 uvicorn 8088）；打开 http://127.0.0.1:8088。
- 抽查：后端起；curl `/api/health`、`/api/experiments`、`/api/sessions/{key}`（2–3 个，含成功/失败）200；
  `npm run build` 过；后端托管 dist 能开首页；pytest 仅覆盖数据加载器；截图 1–2 张。

## MVP 后（延后）
Live 子进程真实驱动（经 AGH）、其余图表（工具健康/评分校准/功效）、Replay 案例库搜索、文档完善与打包。
