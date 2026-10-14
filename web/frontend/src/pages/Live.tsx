// Live：在浏览器里跑一次真实 rollout。
//
// 与 Arena 的区别是这个页面**真的执行**：后端起一个 AGH 进程，页面通过 SSE 看着轨迹长出来。
// 所以页面上有两件 Arena 没有的事 —— 运行前的能力探测（这台机器能不能跑、用哪条路由、
// 凭据从哪来），以及运行后的判分（由项目自己的 check_diagnosis 打分）。
//
// 配置项没有写死的值：模型、端点、密钥都可以在页面上改，缺省值来自后端的能力探测。

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { liveApi, runLive } from '../api/live'
import type { LiveCapabilities, LiveEvent, LiveTask, RouteStatus } from '../api/live'

type Frame =
  | { kind: 'step'; index: number; plan: string; turn?: number; recoveryKind?: string | null }
  | {
      kind: 'call'
      index: number
      name: string
      args: Record<string, unknown> | undefined
      result?: {
        content: string
        isError: boolean
        duration_ms?: number
        resultKind?: string
      }
    }
  | { kind: 'note'; text: string; tone: 'ok' | 'bad' | 'warn' | 'dim' }

// 一次工具调用的四种结局。前两种是**护栏在工作**，不是故障，却很容易被当成故障：
//   guardrail —— 路径在沙箱外，工作区边界拦住
//   approval  —— 轨道把 edit/write/shell 放在审批门后，无人应答时拒绝
// 实测 92 条错误里 64 条属于这两类，而且模型每一次都自己改了路子。用同一个红色
// 描述它们，读者只会得出“系统坏了”的结论。
const OUTCOME: Record<string, { label: string; chip: string; box: string }> = {
  ok: { label: '返回', chip: 'border-ok/50 text-ok', box: 'border-ok/60 text-ink-dim' },
  guardrail: {
    label: '边界拦下',
    chip: 'border-warn/50 text-warn',
    box: 'border-warn/60 text-warn/90',
  },
  approval: {
    label: '审批未应答',
    chip: 'border-warn/50 text-warn',
    box: 'border-warn/60 text-warn/90',
  },
  model: { label: '错误', chip: 'border-bad/50 text-bad', box: 'border-bad text-bad/90' },
}

const outcomeOf = (kind: string | undefined, isError: boolean) =>
  OUTCOME[kind ?? (isError ? 'model' : 'ok')] ?? OUTCOME.ok

interface Outcome {
  answer: string
  passed: boolean | null
  seconds: number | null
  exit: number | null
  timedOut: boolean
  check: { ran: boolean; ok?: boolean; output?: string; scorer?: string; detail?: string } | null
  sessionKey: string | null
  runId: string | null
  /** 本次运行的调用按结局分类计数，用来区分“护栏在工作”和“真的出错”。 */
  resultCounts: { ok?: number; guardrail?: number; approval?: number; model?: number } | null
  /** harness 进程自己的输出尾部：退出码不是 0 时，这是唯一能说明发生了什么的东西。 */
  stdoutTail: string
}

function truncate(s: string, n: number): string {
  if (!s) return ''
  return s.length > n ? s.slice(0, n) + ' …' : s
}

/** 与后端 `normalize_base_url` 同规则：harness 在 baseUrl 后拼 `/chat/completions`，
 *  所以它必须带版本段。这里只用于展示“实际会用哪个地址”，让填了 `http://host:8000`
 *  的人看得到它会变成 `http://host:8000/v1`，而不是等运行失败才知道。 */
function effectiveBaseUrl(url: string): string {
  const cleaned = url.trim().replace(/\/+$/, '')
  if (!cleaned) return cleaned
  return /\/v\d+(\/|$)/.test(cleaned) ? cleaned : `${cleaned}/v1`
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
  const [checking, setChecking] = useState(false)
  const [frames, setFrames] = useState<Frame[]>([])
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)
  const configRef = useRef<HTMLDivElement | null>(null)

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

  const refreshCaps = useCallback(async () => {
    // 只刷新能力，不碰读者已经填进去的值 —— 点「重新检测」的人刚把端点起起来，
    // 把他填的地址重置成默认值正是最坏的反应。
    setChecking(true)
    try {
      setCaps(await liveApi.capabilities(true))
      setLoadError(null)
    } catch (e) {
      setLoadError((e as Error).message)
    } finally {
      setChecking(false)
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
          const next: Frame = {
            kind: 'step',
            index,
            plan: String(ev.plan ?? ''),
            turn: ev.turn as number,
            recoveryKind: (ev.recovery_kind as string | null) ?? null,
          }
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
                  resultKind: (ev.result_kind as string | undefined) ?? undefined,
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
          resultCounts: prev?.resultCounts ?? null,
          stdoutTail: prev?.stdoutTail ?? '',
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
          resultCounts: (s.result_counts as Outcome['resultCounts']) ?? null,
          stdoutTail: String(s.stdout_tail ?? ''),
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

  // 路由行上的「配置」：展开配置区、选中该路由、把它的默认值填上。
  // 直接把请求方送到能改的地方，而不是让他去找。
  const openConfig = useCallback((target?: RouteStatus) => {
    if (target) {
      setRoute(target.route)
      setModel(target.model)
      setBaseUrl(target.base_url)
    }
    setShowAdvanced(true)
    configRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
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
                {/* 改模型与端点的地方就在下面，但它是可折叠的 —— 与其让读者去找到
                    那个开关，不如让每一行自己带路。 */}
                <button
                  className="ml-auto rounded border border-line px-2 py-0.5 text-ink-dim hover:border-accent/50 hover:text-accent"
                  onClick={() => openConfig(r)}
                >
                  配置
                </button>
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

          {/* 主要动作放在任务选定之后、模型配置之前。它原本在卡片最底部，在 1280×720
              的视口里落在 y≈676 —— 浏览器占掉地址栏就看不全了，而录屏与投影多半如此。
              换自己的模型是次要设置（默认折叠），不该把主要按钮挤到它后面。 */}
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

          <div className="mt-4 border-t border-line pt-3" ref={configRef}>
            {/* 这里曾经是一行 72×16 的暗色小字，藏在任务选择器下面 —— 读者被指到「模型配置」
                却找不到它。改成整宽的正式按钮，并把当前选中的路由写在右侧，
                既看得见、也知道里面现在是什么。 */}
            <button
              className={`flex w-full items-center justify-between rounded border px-2.5 py-2 text-xs transition-colors ${
                showAdvanced
                  ? 'border-accent/50 bg-accent/5 text-accent'
                  : 'border-line text-ink-dim hover:border-accent/40 hover:text-ink'
              }`}
              onClick={() => setShowAdvanced((v) => !v)}
              aria-expanded={showAdvanced}
            >
              <span className="flex items-center gap-1.5">
                <span className="text-[10px]">{showAdvanced ? '▾' : '▸'}</span>
                模型与端点配置
              </span>
              <span className="mono text-ink-faint">
                {route}/{model}
              </span>
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
                        // 选了一条显示未就绪的路由就当场重测一次。能力结果是缓存过的
                        // （失败 45 秒 / 成功 10 秒），所以“未就绪”常常只是上次探测的旧值 ——
                        // 让读者自己去找重测按钮，然后看到“未就绪却跑通了”那种矛盾，不如
                        // 在这里自动消除。
                        if (!r.ready) refreshCaps()
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
                  {effectiveBaseUrl(baseUrl) !== baseUrl.trim() && (
                    <div className="mt-1 text-[11px] text-ink-faint">
                      实际使用：<span className="mono text-ink-dim">{effectiveBaseUrl(baseUrl)}</span>
                      （harness 会在它后面拼 <span className="mono">/chat/completions</span>，所以要带版本段）
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-xs text-ink-dim">
                    密钥（留空则用服务端凭据；只在本次运行的子进程环境里）
                  </label>
                  {/* 包在 form 里：浏览器对“不在表单里的密码框”会在控制台发一条提示，
                      而且回车不应该把整个页面重载。这个 form 不提交任何东西。 */}
                  <form onSubmit={(e) => e.preventDefault()}>
                    <input
                      className="mono mt-1 w-full rounded border border-line bg-bg-base px-2 py-1.5 text-xs text-ink"
                      type="password"
                      autoComplete="off"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="sk-…"
                    />
                  </form>
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

                {/* 常驻的重检按钮，而不是只在报错时出现：改了地址的人需要验证新地址，
                    而这正是路由状态还停在旧结果的时候。 */}
                <div className="flex items-center gap-2 border-t border-line pt-2">
                  <button
                    className="whitespace-nowrap rounded border border-line px-2.5 py-1 text-xs text-ink-dim hover:border-accent/50 hover:text-accent disabled:opacity-50"
                    disabled={checking}
                    onClick={refreshCaps}
                  >
                    {checking ? '检测中…' : '重新检测路由'}
                  </button>
                  <span className="text-[11px] leading-snug text-ink-faint">
                    端点未就绪的结果缓存 45 秒，改了配置点这里。
                  </span>
                </div>

                {/* 选中一条还没就绪的路由时，把原因和下一步直接写在这里。
                    否则读者填完地址点「运行」只会收到一句路由未就绪。 */}
                {(() => {
                  const r = caps?.routes.find((x) => x.route === route)
                  if (!r || r.ready) return null
                  const port = (r.base_url.match(/:\d+\/?$/) || [])[0]?.replace(/\D/g, '') || '8000'
                  return (
                    <div className="rounded border border-warn/40 bg-warn/5 p-2 text-[11px] leading-relaxed text-warn">
                      {r.kind === 'local' ? (
                        <>
                          这条路由指向 <span className="mono">{r.base_url}</span>
                          ，它现在没有响应。先把这个端点起起来 —— 任何能答{' '}
                          <span className="mono">/v1/models</span> 的 OpenAI 兼容服务都行：
                          <div className="mono mt-1 whitespace-pre-wrap rounded bg-bg-base p-1.5 text-ink-dim">
                            python serve_openai.py --model-path &lt;权重路径&gt; \
                            {'\n  '}--model-id {model} --port {port}
                          </div>
                          {/* --model-id 要与上面填的模型 id 一致：profile 里声明的模型和端点
                              报回来的名字不一致时，路由能通但跑出来的记录对不上。 */}
                          <div className="text-ink-dim">
                            <span className="mono">--model-id</span> 要和上面填的模型 id 一致。
                          </div>
                          端点起来后点下面「重新检测路由」就能看到它变绿。
                        </>
                      ) : (
                        <>这条路由需要凭据；把它填在下面的「密钥」里，或设置环境变量{' '}
                          <span className="mono">{r.credential.env}</span>。
                        </>
                      )}
                    </div>
                  )
                })()}
              </div>
            )}
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
                  <div key={i} className="pt-1">
                    <div className="flex items-center gap-2">
                      <span className="mono text-[11px] text-ink-faint">#{f.index + 1}</span>
                      {/* 上一个动作被拦下之后，模型调整了做法，新一步标出来。
                          措辞按被拦下的原因分开：“边界拦下”与“审批未应答”不是故障，
                          写成“错误恢复”会让读者以为运行出了问题。 */}
                      {f.recoveryKind && (
                        <span className="chip border-warn/50 text-warn">
                          {f.recoveryKind === 'guardrail'
                            ? '边界拦下后调整'
                            : f.recoveryKind === 'approval'
                              ? '审批未应答后调整'
                              : '错误后恢复'}
                        </span>
                      )}
                      <span className="h-px flex-1 bg-line" />
                    </div>
                    {/* 这一步的规划文本。之前只画了序号和分隔线，模型自己说的话
                        全部丢掉 —— 而“任务规划”正是这个页面要展示的环节之一。
                        后端还会在 plan 迟到时重发这一帧（见 _Cursor），这里得接住。 */}
                    {f.plan && (
                      <p className="mt-1.5 whitespace-pre-wrap text-sm leading-relaxed text-ink">
                        {f.plan.length > 900 ? f.plan.slice(0, 900) + ' …' : f.plan}
                      </p>
                    )}
                  </div>
                ) : (
                  <div
                    key={i}
                    className={`rounded border bg-bg-base p-2 ${
                      f.result && !f.result.isError ? 'border-line' : 'border-line'
                    }`}
                  >
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className="mono rounded bg-accent/15 px-1.5 py-0.5 text-accent">
                        {f.name}
                      </span>
                      {f.result?.duration_ms != null && (
                        <span className="mono text-ink-faint">
                          {(f.result.duration_ms / 1000).toFixed(1)}s
                        </span>
                      )}
                      {f.result?.isError && (
                        <span className={`chip ${outcomeOf(f.result.resultKind, true).chip}`}>
                          {outcomeOf(f.result.resultKind, true).label}
                        </span>
                      )}
                    </div>
                    {f.args && Object.keys(f.args).length > 0 && (
                      <pre className="mono mt-1 overflow-x-auto text-[11px] text-ink-dim">
                        {JSON.stringify(f.args)}
                      </pre>
                    )}
                    {f.result && (
                      <pre
                        className={`mt-1.5 whitespace-pre-wrap break-words border-l-2 pl-2 text-[11px] leading-relaxed ${
                          outcomeOf(f.result.resultKind, f.result.isError).box
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
              {/* 调用结局分布。这一行是给读者降噪用的：一次真实运行里常有被拦下的调用，
                  逐个看红黄标签很容易得出“系统坏了”；集中写出来就能看出
                  “被边界拦下 N 次、模型 N 次都自己改了路子”。 */}
              {outcome.resultCounts && (() => {
                const c = outcome.resultCounts!
                const total = (c.ok ?? 0) + (c.guardrail ?? 0) + (c.approval ?? 0) + (c.model ?? 0)
                if (!total) return null
                const parts: string[] = [`成功 ${c.ok ?? 0}`]
                if (c.guardrail) parts.push(`边界拦下 ${c.guardrail}`)
                if (c.approval) parts.push(`审批未应答 ${c.approval}`)
                if (c.model) parts.push(`出错 ${c.model}`)
                const boundary = (c.guardrail ?? 0) + (c.approval ?? 0)
                return (
                  <div className="mt-2 text-[11px] leading-relaxed text-ink-faint">
                    调用 {total} 次：{parts.join(' · ')}
                    {boundary > 0 && (
                      <span className="text-ink-dim">
                        {' '}
                        —— 被拦下的都是设计中的边界（工作区范围 / 审批门），不是故障。
                      </span>
                    )}
                  </div>
                )
              })()}
              {outcome.check?.output && (
                <pre className="mono mt-2 whitespace-pre-wrap break-words rounded border border-line bg-bg-base p-2 text-[11px] text-ink-dim">
                  {outcome.check.output}
                </pre>
              )}
              {/* 退出码不是 0，或者根本没跑到判分 —— 这两种情况下读者手上只有一个数字，
                  而 harness 已经把原因写在自己的输出里了。失败时默认展开。 */}
              {outcome.stdoutTail && (outcome.exit !== 0 || !outcome.check?.ran) && (
                <details className="mt-2" open>
                  <summary className="cursor-pointer text-xs text-warn">
                    运行未正常结束（exit {outcome.exit ?? '?'}）—— harness 输出尾部
                  </summary>
                  <pre className="mono mt-1 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded border border-warn/30 bg-bg-base p-2 text-[11px] text-ink-dim">
                    {outcome.stdoutTail}
                  </pre>
                </details>
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
