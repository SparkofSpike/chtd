const blocks: { title: string; lines: string[]; note?: string }[] = [
  {
    title: '1. 启动 base 与 adapter 两个模型端点',
    lines: [
      '# base（无 --adapter-path），端口 8000',
      'python serve_openai.py --port 8000',
      '# adapter（带 LoRA adapter），端口 8001',
      'python serve_openai.py --port 8001 --adapter-path runs/chtd-1/adapter',
    ],
  },
  {
    title: '2. 经 harness 跑 rollout（Live 模式）',
    lines: [
      '# 需装有 agnes-harness 构建产物；本机无 agnes.mjs 时自动降级 Replay',
      '$env:CHTD_AGNES_CLI = "E:\\Coding\\agnes-harness\\packages\\cli\\dist\\local\\agnes.mjs"',
      'node $env:CHTD_AGNES_CLI -p --standalone --preset minimal-rl --model primary=student/qwen3-4b-instruct-20261014',
    ],
    note: '本机现状：无 agnes CLI 构建，Live 稳定走降级 Replay；全站 Replay 形态无 GPU/无模型可完整演示。',
  },
  {
    title: '3. 复现数字（eval 脚本）',
    lines: [
      'python eval/dose_response.py',
      'python eval/pooled_effect.py',
      'python eval/failure_rate.py',
      'python eval/scorer_benchmark_holdout3.py',
      'python eval/power_holdout3.py',
    ],
  },
  {
    title: '4. 后端与前端',
    lines: [
      'cd web/backend; python export_data.py        # 导出 data/*.json',
      'uvicorn app.main:app --host 127.0.0.1 --port 8088',
      'cd web/frontend; npm run build               # 产物 dist/，由后端静态托管',
    ],
  },
]

export default function Reproduce() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">Reproduce</h1>
      <p className="text-sm text-ink-dim">
        原始会话、权重与云端数据在 <span className="mono">/mnt/proj/chtd-experiments/</span>（见 docs/cloud-storage.md）。
        下列命令在装有 agnes-harness 构建并启动两端点的机器上可复现 Live；否则使用 Replay。
      </p>
      {blocks.map((b) => (
        <section key={b.title} className="card p-4">
          <h2 className="mb-2 text-sm font-semibold text-ink">{b.title}</h2>
          <pre className="mono overflow-x-auto rounded bg-bg-base p-3 text-xs leading-relaxed text-accent-soft">
            {b.lines.join('\n')}
          </pre>
          {b.note && <p className="mt-2 text-xs text-warn">{b.note}</p>}
        </section>
      ))}
    </div>
  )
}
