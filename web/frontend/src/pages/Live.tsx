// Live：在浏览器里跑一次真实 rollout。
//
// 与 Arena 的区别是这个页面**真的执行**：后端起一个 AGH 进程，页面通过 SSE 看着轨迹长出来。
// 所以页面上有两件 Arena 没有的事 —— 运行前的能力探测（这台机器能不能跑、用哪条路由、
// 凭据从哪来），以及运行后的判分（由项目自己的 check_diagnosis 打分）。
//
// 配置项没有写死的值：模型、端点、密钥都可以在页面上改，缺省值来自后端的能力探测。

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { liveApi, runLive } from '../api/live'
import type { LiveCapabilities, LiveEvent, LiveTask } from '../api/live'

type Frame =
  | { kind: 'step'; index: number; plan: string; turn?: number }
  | {
      kind: 'call'
      index: number
      name: string
      args: Record<string, unknown> | undefined
      result?: { content: string; isError: boolean; duration_ms?: number }
    }
  | { kind: 'note'; text: string; tone: 'ok' | 'bad' | 'warn' | 'dim' }

interface Outcome {
  answer: string
  passed: boolean | null
  seconds: number | null
  exit: number | null
  timedOut: boolean
  check: { ran: boolean; ok?: boolean; output?: string; scorer?: string; detail?: string } | null
  sessionKey: string | null
  runId: string | null
}

function truncate(s: string, n: number): string {
  if (!s) return ''
  return s.length > n ? s.slice(0, n) + ' …' : s
}

function Dot({ ok }: { ok: boolean | null | undefined }) {
  const color = ok === null || ok === undefined ? 'bg-ink-faint' : ok ? 'bg-ok' : 'bg-bad'
  return <span className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${color}`} />
}

export default function Live() {
  const [caps, setCaps] = useState<LiveCapabilities | null>(null)
  const [tasks, setTasks] = useState<LiveTask[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)

  const [source, setSource] = useState<'task' | 'custom'>('task')
  const [taskId, setTaskId] = useState('')
  const [prompt, setPrompt] = useState('')
  const [route, setRoute] = useState('')
  const [model, setModel] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [timeoutSecs, setTimeoutSecs] = useState(240)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [running, setRunning] = useState(false)
  const [frames, setFrames] = useState<Frame[]>([])
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let stop = false
    ;(async () => {
      try {
        const [c, t] = await Promise.all([liveApi.capabilities(), liveApi.tasks()])
        if (stop) return
        setCaps(c)
        setTasks(t.tasks)
        setRoute(c.defaults.route)
        setModel(c.defaults.model)
        setBaseUrl(c.routes.find((r) => r.route === c.defaults.route)?.base_url ?? '')
        setTimeoutSecs(c.defaults.timeout)
        const first = t.tasks.find((x) => x.fixture_ready) ?? t.tasks[0]
        if (first) {
          setTaskId(first.task_id)
          setPrompt(first.prompt)
        }
      } catch (e) {
        if (!stop) setLoadError((e as Error).message)
      }
    })()
    return () => {
      stop = true
    }
  }, [])

  const selected = useMemo(
    () => tasks.find((t) => t.task_id === taskId) ?? null,
    [tasks, taskId],
  )

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [frames])

  const pickTask = useCallback(
    (id: string) => {
      setTaskId(id)
      const t = tasks.find((x) => x.task_id === id)
      if (t) setPrompt(t.prompt)
    },
    [tasks],
  )

  const onEvent = useCallback((ev: LiveEvent) => {
    switch (ev.event) {
      case 'begin': {
        const files = (ev.files as string[]) ?? []
        setFrames([
          {
            kind: 'note',
            tone: 'dim',
            text: `开始运行：${ev.task_id} · 路由 ${ev.route}/${ev.model} · preset ${ev.preset}${
              files.length ? ` · 工作区文件 ${files.join(', ')}` : ''
            }`,
          },
        ])
        break
      }
      case 'step': {
        const index = Number(ev.step_index ?? 0)
        setFrames((prev) => {
          const at = prev.findIndex((f) => f.kind === 'step' && f.index === index)
          const next: Frame = { kind: 'step', index, plan: String(ev.plan ?? ''), turn: ev.turn as number }
          if (at >= 0) {
            const copy = [...prev]
            copy[at] = next
            return copy
          }
          return [...prev, next]
        })
        break
      }
      case 'tool_call': {
        setFrames((prev) => [
          ...prev,
          {
            kind: 'call',
            index: Number(ev.step_index ?? 0),
            name: String(ev.name ?? '?'),
            args: (ev.args as Record<string, unknown> | undefined) ?? undefined,
          },
        ])
        break
      }
      case 'tool_result': {
        setFrames((prev) => {
          const copy = [...prev]
          for (let i = copy.length - 1; i >= 0; i--) {
            const f = copy[i]
            if (
              f.kind === 'call' &&
              f.index === Number(ev.step_index ?? 0) &&
              f.name === String(ev.name ?? '?') &&
              !f.result
            ) {
              copy[i] = {
                ...f,
                result: {
                  content: String(ev.content ?? ''),
                  isError: Boolean(ev.isError),
                  duration_ms: ev.duration_ms as number | undefined,
                },
              }
              break
            }
          }
          return copy
        })
        break
      }
      case 'final': {
        setOutcome((prev) => ({
          answer: String(ev.answer ?? ''),
          passed: (ev.verdict as { passed?: boolean } | undefined)?.passed ?? null,
          seconds: prev?.seconds ?? null,
          exit: prev?.exit ?? null,
          timedOut: prev?.timedOut ?? false,
          check: prev?.check ?? null,
          sessionKey: prev?.sessionKey ?? null,
          runId: prev?.runId ?? null,
        }))
        break
      }
      case 'end': {
        const s = (ev.session ?? {}) as Record<string, unknown>
        setOutcome((prev) => ({
          answer: prev?.answer ?? '',
          passed: (s.passed as boolean | null) ?? prev?.passed ?? null,
          seconds: (s.seconds as number) ?? null,
          exit: (s.exit as number) ?? null,
          timedOut: Boolean(s.timed_out),
          check: (s.check as Outcome['check']) ?? null,
          sessionKey: (s.session_key as string) ?? null,
          runId: (s.run_id as string) ?? null,
        }))
        break
      }
      case 'live_error': {
        const reasons = (ev.reasons as string[]) ?? []
        setError([String(ev.message ?? '运行失败'), ...reasons].join(' · '))
        break
      }
      default:
        break
    }
  }, [])

  const start = useCallback(async () => {
    if (running) return
    setError(null)
    setOutcome(null)
    setFrames([])
    setRunning(true)
    const controller = new AbortController()
    abortRef.current = controller
    const opts =
      source === 'task'
        ? { task_id: taskId, route, model, base_url: baseUrl, api_key: apiKey || undefined, timeout: timeoutSecs }
        : { prompt, route, model, base_url: baseUrl, api_key: apiKey || undefined, timeout: timeoutSecs }
    try {
      await runLive(opts, onEvent, (msg) => setError(msg), controller.signal)
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }, [running, source, taskId, prompt, route, model, baseUrl, apiKey, timeoutSecs, onEvent])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setRunning(false)
  }, [])

  const liveReady = caps?.mode === 'live'
  const runnable = tasks.filter((t) => t.fixture_ready)
  const groupedTasks = useMemo(() => {
    const bySet = new Map<string, LiveTask[]>()
    for (const t of tasks) {
      const list = bySet.get(t.set) ?? []
      list.push(t)
      bySet.set(t.set, list)
    }
    return [...bySet.entries()]
  }, [tasks])

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink">在线运行：真的在跑，不是录像</h1>
          <p className="mt-1 text-sm text-ink-dim">
            在这里跑的每一次都由 AGH 真实执行 —— 模型自己读文件、调工具、给结论，
            再由项目自己的判分器打分。轨迹随运行实时出现。
          </p>
        </div>
        <span className={`chip ${liveReady ? 'border-accent/60 text-accent' : 'border-warn/60 text-warn'}`}>
          <Dot ok={liveReady} />
          {liveReady ? 'Live 就绪' : 'Live 不可用'}
        </span>
      </header>

      {loadError && (
        <div className="card border-bad/50 p-3 text-sm text-bad">后端不可达：{loadError}</div>
      )}

      {caps && !liveReady && (
        <div className="card border-warn/50 p-3 text-sm">
          <div className="font-medium text-warn">这台机器上 Live 跑不起来，缺的是：</div>
          <ul className="mt-1 list-inside list-disc text-ink-dim">
            {caps.reasons.map((r) => (
              <li key={r} className="mono text-xs">
                {r}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-ink-faint">
            Arena 页的回放不受影响 —— 它读的是已记录的真实轨迹，不需要模型端点。
          </p>
        </div>
      )}

      {caps && (
        <div className="card p-3">
          <div className="grid gap-3 text-xs sm:grid-cols-3">
            <div className="flex items-center gap-2">
              <Dot ok={caps.agnes_cli.exists} />
              <span className="shrink-0 text-ink-dim">agnes CLI</span>
              <span className="mono break-all text-ink-faint">{caps.agnes_cli.path}</span>
            </div>
            <div className="flex items-center gap-2">
              <Dot ok={caps.node.ok} />
              <span className="text-ink-dim">node</span>
              <span className="mono text-ink-faint">{caps.node.version || caps.node.bin}</span>
            </div>
            <div className="flex items-center gap-2">
              <Dot ok={runnable.length > 0} />
              <span className="text-ink-dim">可跑任务</span>
              <span className="mono text-ink-faint">
                {runnable.length} / {tasks.length}
              </span>
            </div>
          </div>
          <div className="mt-3 space-y-1.5 border-t border-line pt-3">
            {caps.routes.map((r) => (
              <div key={r.route} className="flex flex-wrap items-center gap-2 text-xs">
                <Dot ok={r.ready} />
                <span className="mono text-ink">{r.route}</span>
                <span className="chip border-line text-ink-faint">
                  {r.kind === 'hosted' ? '托管 API' : '本地端点'}
                </span>
                <span className="mono text-ink-faint">{r.model}</span>
                <span className="mono text-ink-faint">{r.base_url}</span>
                <span className={`chip ${r.credential.ok ? 'border-ok/50 text-ok' : 'border-warn/50 text-warn'}`}>
                  凭据：{r.credential.source === 'env' ? '环境变量' : r.credential.source === 'file' ? '密钥文件' : '无'}
                </span>
                {!r.ready && <span className="text-ink-faint">{r.note}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
        {/* 配置 */}
        <section className="card h-fit p-4">
          <div className="flex items-center gap-2">
            <button
              className={`chip ${source === 'task' ? 'border-accent/60 text-accent' : 'border-line text-ink-dim'}`}
              onClick={() => setSource('task')}
            >
              任务集
            </button>
            <button
              className={`chip ${source === 'custom' ? 'border-accent/60 text-accent' : 'border-line text-ink-dim'}`}
              onClick={() => setSource('custom')}
            >
              自定义 prompt
            </button>
          </div>

          {source === 'task' ? (
            <>
              <label className="mt-3 block text-xs text-ink-dim">任务</label>
              <select
                className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                value={taskId}
                onChange={(e) => pickTask(e.target.value)}
              >
                {groupedTasks.map(([set, list]) => (
                  <optgroup key={set} label={set}>
                    {list.map((t) => (
                      <option key={t.task_id} value={t.task_id} disabled={!t.fixture_ready}>
                        {t.task_id}
                        {t.fixture_ready ? '' : '（缺 fixture）'}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>

              {selected && (
                <div className="mt-3 space-y-2 text-xs">
                  <div className="rounded border border-line bg-bg-base p-2">
                    <div className="text-ink-faint">提示词</div>
                    <p className="mt-1 whitespace-pre-wrap leading-relaxed text-ink-dim">
                      {selected.prompt}
                    </p>
                  </div>
                  {selected.files.length > 0 && (
                    <div className="text-ink-faint">
                      fixture：<span className="mono text-ink-dim">{selected.files.join(', ')}</span>
                    </div>
                  )}
                  {selected.expect.length > 0 && (
                    <details className="rounded border border-line bg-bg-base p-2">
                      <summary className="cursor-pointer text-ink-faint">
                        判分标准（答案须命中，且不得为「没缺陷」）
                      </summary>
                      <ul className="mt-1.5 space-y-1">
                        {selected.expect.map((p) => (
                          <li key={p} className="mono text-[11px] text-ok/90">
                            ✓ {p}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <label className="mt-3 block text-xs text-ink-dim">要模型做什么</label>
              <textarea
                className="mono mt-1 h-40 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="例如：读出这个目录里的文件，说明其中哪个是错的。"
              />
              <p className="mt-1 text-[11px] text-ink-faint">
                自定义 prompt 不带判分，只给出答案与完整轨迹。
              </p>
            </>
          )}

          <div className="mt-4 border-t border-line pt-3">
            <button
              className="text-xs text-ink-dim hover:text-ink"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? '收起' : '展开'}模型配置
            </button>
            {showAdvanced && (
              <div className="mt-2 space-y-2">
                <div>
                  <label className="block text-xs text-ink-dim">路由</label>
                  <select
                    className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                    value={route}
                    onChange={(e) => {
                      setRoute(e.target.value)
                      const r = caps?.routes.find((x) => x.route === e.target.value)
                      if (r) {
                        setModel(r.model)
                        setBaseUrl(r.base_url)
                      }
                    }}
                  >
                    {(caps?.routes ?? []).map((r) => (
                      <option key={r.route} value={r.route}>
                        {r.route}（{r.ready ? '就绪' : '未就绪'}）
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-ink-dim">模型 id</label>
                  <input
                    className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-xs text-ink-dim">端点 baseUrl</label>
                  <input
                    className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-xs text-ink-dim">
                    密钥（留空则用服务端凭据；只在本次运行的子进程环境里）
                  </label>
                  <input
                    className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-…"
                  />
                </div>
                <div>
                  <label className="block text-xs text-ink-dim">超时（秒）</label>
                  <input
                    className="mono mt-1 w-24 rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                    type="number"
                    min={10}
                    max={1800}
                    value={timeoutSecs}
                    onChange={(e) => setTimeoutSecs(Number(e.target.value) || 240)}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button
              className="rounded bg-accent/20 px-3 py-1.5 text-sm font-medium text-accent hover:bg-accent/30 disabled:opacity-40"
              disabled={running || !liveReady || (source === 'task' ? !taskId : !prompt.trim())}
              onClick={start}
            >
              {running ? '运行中…' : '运行'}
            </button>
            {running && (
              <button
                className="rounded border border-bad/50 px-3 py-1.5 text-sm text-bad hover:bg-bad/10"
                onClick={stop}
              >
                停止
              </button>
            )}
            {running && <span className="text-xs text-ink-faint">真实执行，通常 10–60 秒</span>}
          </div>
        </section>

        {/* 运行日志 */}
        <section className="card flex min-h-[420px] flex-col p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-ink">运行轨迹</h2>
            {outcome?.runId && <span className="mono text-[11px] text-ink-faint">{outcome.runId}</span>}
          </div>

          <div ref={logRef} className="mt-3 flex-1 overflow-y-auto pr-1" style={{ maxHeight: 560 }}>
            {frames.length === 0 && !error && (
              <p className="text-sm text-ink-faint">
                {running ? '正在启动 AGH…' : '还没有运行。左侧选一个任务，点「运行」。'}
              </p>
            )}
            <div className="space-y-2">
              {frames.map((f, i) =>
                f.kind === 'note' ? (
                  <p key={i} className="text-xs text-ink-faint">
                    {f.text}
                  </p>
                ) : f.kind === 'step' ? (
                  <div key={i} className="flex items-center gap-2 pt-1">
                    <span className="mono text-[11px] text-ink-faint">#{f.index + 1}</span>
                    <span className="h-px flex-1 bg-line" />
                  </div>
                ) : (
                  <div key={i} className="rounded border border-line bg-bg-base p-2">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className="mono rounded bg-accent/15 px-1.5 py-0.5 text-accent">
                        {f.name}
                      </span>
                      {f.result?.duration_ms != null && (
                        <span className="mono text-ink-faint">
                          {(f.result.duration_ms / 1000).toFixed(1)}s
                        </span>
                      )}
                      {f.result?.isError && <span className="chip border-bad/50 text-bad">错误</span>}
                    </div>
                    {f.args && Object.keys(f.args).length > 0 && (
                      <pre className="mono mt-1 overflow-x-auto text-[11px] text-ink-dim">
                        {JSON.stringify(f.args)}
                      </pre>
                    )}
                    {f.result && (
                      <pre
                        className={`mt-1.5 whitespace-pre-wrap break-words border-l-2 pl-2 text-[11px] leading-relaxed ${
                          f.result.isError ? 'border-bad text-bad/90' : 'border-ok/60 text-ink-dim'
                        }`}
                      >
                        {truncate(f.result.content, 600)}
                      </pre>
                    )}
                  </div>
                ),
              )}
              {running && <p className="animate-pulse text-xs text-accent">● 运行中</p>}
              {error && <p className="text-sm text-bad">{error}</p>}
            </div>
          </div>

          {outcome && (
            <div className="mt-3 border-t border-line pt-3">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`chip ${
                    outcome.passed === null
                      ? 'border-line text-ink-dim'
                      : outcome.passed
                        ? 'border-ok/60 text-ok'
                        : 'border-bad/60 text-bad'
                  }`}
                >
                  {outcome.passed === null ? '未判分' : outcome.passed ? '通过' : '未通过'}
                </span>
                {outcome.seconds != null && (
                  <span className="mono text-xs text-ink-faint">{outcome.seconds}s</span>
                )}
                {outcome.exit != null && (
                  <span className="mono text-xs text-ink-faint">exit {outcome.exit}</span>
                )}
                {outcome.timedOut && <span className="chip border-warn/50 text-warn">超时</span>}
                {outcome.sessionKey && (
                  <span className="mono text-[11px] text-ink-faint">{outcome.sessionKey}</span>
                )}
              </div>
              {outcome.check?.output && (
                <pre className="mono mt-2 whitespace-pre-wrap break-words rounded border border-line bg-bg-base p-2 text-[11px] text-ink-dim">
                  {outcome.check.output}
                </pre>
              )}
              {outcome.answer && (
                <details className="mt-2" open>
                  <summary className="cursor-pointer text-xs text-ink-dim">模型答案</summary>
                  <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-ink">
                    {outcome.answer}
                  </p>
                </details>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
