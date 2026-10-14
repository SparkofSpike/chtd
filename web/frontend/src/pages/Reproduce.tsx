const blocks: { title: string; lines: string[]; note?: string }[] = [
  {
    title: '1.（可选）起一个模型端点',
    lines: [
      '# 只有要跑 student 路由时才需要这一步。teacher 路由是托管 API，',
      '# 不需要 GPU、不需要权重 —— 这也是页面能在一台笔记本上跑起来的原因。',
      'python serve_openai.py --model-path <Qwen 权重路径> --port 8000',
      '# 带 LoRA adapter 的臂（换成你要评测的那份）：',
      'python serve_openai.py --model-path <Qwen 权重路径> --adapter-path <adapter 路径> --port 8001',
    ],
    note: '端点起来后 Live 页的路由灯会自己变绿 —— 不可达的探测结果缓存 45 秒，稍等一次轮询即可。',
  },
  {
    title: '2. 跑一次真实 rollout',
    lines: [
      '# 最省事：打开 http://127.0.0.1:8088/#/live ，选一个任务，点「运行」。',
      '# 页面走的就是下面这条命令，判分与轨迹都从同一次运行的账本里读。',
      '',
      '# 命令行等价形式（AGH_HOME 指向一个临时 home）：',
      '$env:AGNES_PROFILE = "local-dev"',
      'node <agnes.mjs> -p --standalone --preset minimal-rl --cwd <工作区> --model primary=teacher/agnes-3.0-flash "<任务的 prompt>"',
    ],
    note: '模型与端点不必是记录里的那对：Live 页的「模型配置」可以改路由、模型 id、baseUrl 与密钥，不用改代码。',
  },
  {
    title: '3. 复现本项目的数字（eval 脚本）',
    lines: [
      'python eval/dose_response.py                 # 剂量-响应曲线',
      'python eval/pooled_effect.py                 # 57 题池化：31.6% → 17.5%',
      'python eval/failure_rate.py                  # 失败率端点',
      'python eval/scorer_benchmark_holdout3.py     # 判分器校准：75.7% / 75.0%',
      'python eval/power_holdout3.py                # 统计功效：MDE@30题 = 26.6pp',
      'python eval/replication_floor.py             # 确定性测量地板',
    ],
    note: '这些脚本只读 runs/ 与 data/ 下的记录，不需要模型。',
  },
  {
    title: '4. 起这个演示端自己',
    lines: [
      'cd web; .\\start.ps1                          # 建 venv → 导出数据 → 托管前端，一键',
      '',
      '# 或者分步：',
      'cd web/backend; python export_data.py        # 导出 backend/data/*.json',
      'uvicorn app.main:app --host 127.0.0.1 --port 8088',
      'cd web/frontend; npm run build               # 产物 dist/，由后端静态托管',
    ],
  },
]

export default function Reproduce() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">Reproduce</h1>
      <p className="text-sm leading-relaxed text-ink-dim">
        原始会话、权重与云端数据在 <span className="mono">/mnt/proj/chtd-experiments/</span>
        （见 <span className="mono">docs/cloud-storage.md</span>）。Replay 形态
        <span className="text-ink">不需要 GPU、不需要模型、断网也能跑</span>；Live 形态需要可执行的
        agnes CLI 与至少一条就绪的路由。
      </p>
      <p className="text-sm leading-relaxed text-ink-dim">
        要换成你自己的模型，见{' '}
        <span className="mono">docs/running-with-your-own-models.md</span> —— 那页列出模型进入流水线的
        每一个位置、全部环境变量，以及两个会拒绝配置不一致的门控。
      </p>
      {blocks.map((b) => (
        <section key={b.title} className="card p-4">
          <h2 className="mb-2 text-sm font-semibold text-ink">{b.title}</h2>
          <pre className="mono overflow-x-auto rounded bg-bg-base p-3 text-xs leading-relaxed text-accent-soft">
            {b.lines.join('\n')}
          </pre>
          {b.note && <p className="mt-2 text-xs leading-relaxed text-ink-dim">{b.note}</p>}
        </section>
      ))}
    </div>
  )
}
