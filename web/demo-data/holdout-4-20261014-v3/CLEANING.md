# demo-data 路径规范化记录

本目录的数据是真实运行账本的快照。为随公开仓库交付，**机器绝对路径前缀**已折叠为
`<repo>` / `<workspace>`，其余字段（答案、判定、耗时、事件、哈希）**未做任何改动**。

| 文件 | 替换数 | sha256（清洗后） |
| --- | ---: | --- |
| `rollouts\holdout4-base-nocheck\records.jsonl` | 48 | `69760a6cab9e1c18…` |
| `rollouts\holdout4-chtd-seed0\records.jsonl` | 1 | `b474f71e0167ca91…` |
| `rollouts\holdout4-chtd-seed0-nocheck\records.jsonl` | 48 | `e1d8fdd409e0ac7e…` |
| `rollouts\holdout4-chtd-seed1-nocheck\records.jsonl` | 33 | `138f2b67b1aa2c2a…` |
| `rollouts\holdout4-chtd-seed1-resume\records.jsonl` | 15 | `814699ab89e43db8…` |
| `rollouts\holdout4-control-seed0\records.jsonl` | 48 | `37485a9eaa6b31ad…` |
| `rollouts\holdout4-control-seed0-nocheck\records.jsonl` | 48 | `5f45b819d6692f93…` |
| `rollouts\holdout4-control-seed1-nocheck\records.jsonl` | 48 | `1de1c5d18b056b2a…` |
| `rollouts\holdout4-smoke-basecontrol\records.jsonl` | 2 | `de516e6e581598f0…` |
| `rollouts\holdout4-smoke-basecontrol-node24\records.jsonl` | 1 | `0ce860f177d95211…` |
| `rollouts\holdout4-smoke-chtd-clean\records.jsonl` | 1 | `917dbe31d2c210dc…` |
| `agh_home/data/sessions.db` | 134 | `41be6a028d8640a7…` |"

- 折叠的前缀：`D:\Users\Mojito\PycharmProjects\ML_Project\chtd`
- 折叠为：`<repo>`
- 总计：**427** 处

## 为什么

回放里工具返回的 `ENOENT` 消息会带上被访问文件的完整路径。保留它有意义（能看到失败的是
哪个相对路径），但把运行者的家目录一并公开没有必要。折叠后仍可读：

```
read failed: ENOENT: no such file or directory, open '<repo>\data\holdout-4-20261014-v3\...'
```

## 复现清洗

后端在渲染时另有一层脱敏（`app/loaders/trajectory.py::sanitize_text`），
所以即使拿到未清洗的原始账本，界面也不会显示机器路径。

## 未改动的内容

- 判定结果（`passed` / `check_output`）
- 模型输出（`answer`）与事件流
- 时间、退出码、轨道与 profile 哈希
- 全部工具调用与返回（除路径前缀外逐字保留）
