import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { Session, SessionIndexItem, SessionsResponse } from '../api/types'
import Timeline from '../components/Timeline/Timeline'

function truncateHash(h?: string) {
  return h ? h.slice(0, 12) + '…' : ''
}

// The arms this task set contains, in the order the UI offers them. `chtd-seed1` is the
// student-state arm whose behaviour matches the training material; `chtd-seed0` regressed to
// guessing paths without listing the workspace on this task set, which is itself a recorded
// finding (runs/holdout-4-20261014/README.md) and stays selectable for that reason.
const ARM_LABELS: Record<string, { label: string; hint?: string }> = {
  'base': { label: 'base', hint: '未微调基线' },
  'chtd-seed1': { label: 'CHTD · seed1', hint: '学生状态臂（推荐对照）' },
  'chtd-seed0': { label: 'CHTD · seed0', hint: '该种子在本任务集出现策略退化' },
  'control-seed0': { label: 'control · seed0', hint: '教师自轨迹对照臂' },
  'control-seed1': { label: 'control · seed1', hint: '教师自轨迹对照臂' },
}

const DEFAULT_LEFT = 'base'
const DEFAULT_RIGHT = 'chtd-seed1'

const armLabel = (g: string) => ARM_LABELS[g]?.label ?? g
const armHint = (g: string) => ARM_LABELS[g]?.hint ?? ''

export default function Arena() {
  const [idx, setIdx] = useState<SessionsResponse | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [taskId, setTaskId] = useState<string>('')
  const [leftArm, setLeftArm] = useState<string>(DEFAULT_LEFT)
  const [rightArm, setRightArm] = useState<string>(DEFAULT_RIGHT)
  const [left, setLeft] = useState<Session | null>(null)
  const [right, setRight] = useState<Session | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.sessions()
      .then((d) => setIdx(d))
      .catch((e) => setErr(e.message))
  }, [])

  // group -> task_id -> session index entry (smoke runs excluded: they are pipeline checks)
  const byGroup = useMemo(() => {
    const m = new Map<string, Map<string, SessionIndexItem>>()
    if (!idx) return m
    for (const s of idx.sessions) {
      const g = s.arm_group ?? s.arm
      if (!g || s.arm.toLowerCase().includes('smoke')) continue
      const inner = m.get(g) ?? new Map<string, SessionIndexItem>()
      inner.set(s.task_id, s)
      m.set(g, inner)
    }
    return m
  }, [idx])

  // Groups offered in the pickers, richest first, unknown groups appended.
  const groups = useMemo(() => {
    const known = Object.keys(ARM_LABELS).filter((g) => byGroup.has(g))
    const extra = [...byGroup.keys()].filter((g) => !(g in ARM_LABELS)).sort()
    return [...known, ...extra]
  }, [byGroup])

  // Tasks where both selected arms have a run, so a side-by-side always has two sides.
  const tasks = useMemo(() => {
    const l = byGroup.get(leftArm)
    const r = byGroup.get(rightArm)
    if (!l || !r) return []
    const back = byGroup.get('base')
    const all = new Set([...l.keys(), ...r.keys()])
    return [...all]
      .filter((t) => l.has(t) && r.has(t))
      .sort((a, b) => {
        const la = l.get(a)!.passed
        const lb = l.get(b)!.passed
        const ra = r.get(a)!.passed
        const rb = r.get(b)!.passed
        // disagreements first: they are the informative rows
        const da = la === ra ? 1 : 0
        const db = lb === rb ? 1 : 0
        if (da !== db) return da - db
        void back
        return a.localeCompare(b)
      })
  }, [byGroup, leftArm, rightArm])

  // Keep the selection valid when the arm pair changes.
  useEffect(() => {
    if (tasks.length && !tasks.includes(taskId)) setTaskId(tasks[0])
    if (!tasks.length) setTaskId('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks])

  useEffect(() => {
    if (!taskId) {
      setLeft(null)
      setRight(null)
      return
    }
    const lk = byGroup.get(leftArm)?.get(taskId)?.session_key
    const rk = byGroup.get(rightArm)?.get(taskId)?.session_key
    if (!lk && !rk) return
    let on = true
    setLoading(true)
    Promise.all([
      lk ? api.session(lk).catch(() => null) : Promise.resolve(null),
      rk ? api.session(rk).catch(() => null) : Promise.resolve(null),
    ])
      .then(([l, r]) => {
        if (!on) return
        setLeft(l)
        setRight(r)
      })
      .finally(() => on && setLoading(false))
    return () => {
      on = false
    }
  }, [taskId, leftArm, rightArm, byGroup])

  const rowOf = (arm: string, t: string) => byGroup.get(arm)?.get(t)
  const bothRun = tasks.length > 0

  // 两臂结果不同就是这张表里真正要看的那几行。分组与选项文本共用同一个判定，
  // 免得“排在前面的”与“标了星的”将来各自漂移。
  const isDivergent = (t: string) =>
    rowOf(leftArm, t)?.passed !== rowOf(rightArm, t)?.passed

  const taskOption = (t: string) => {
    const l = rowOf(leftArm, t)
    const r = rowOf(rightArm, t)
    return (
      <option key={t} value={t}>
        {t}（{l?.passed ? '通过' : '失败'} / {r?.passed ? '通过' : '失败'}）
      </option>
    )
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">并排回放：同一道题，两条轨迹</h1>
      <p className="text-sm text-ink-dim">
        两侧是<strong className="text-ink">同一条冻结轨道</strong>上的两次真实运行，唯一变量是模型 ——
        轨迹、工具调用与判定均逐事件取自 AGH 账本，未改写。
      </p>

      {err && (
        <div className="card border-bad/50 p-4 text-sm text-bad">
          暂无数据（后端 /api/sessions 不可达：{err}）。
        </div>
      )}

      {idx && groups.length > 0 && (
        <div className="card p-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs text-ink-dim">左臂</label>
              <select
                className="w-full rounded border border-line bg-bg-base px-2 py-1.5 text-sm text-ink"
                value={leftArm}
                onChange={(e) => setLeftArm(e.target.value)}
              >
                {groups.map((g) => (
                  <option key={g} value={g}>
                    {armLabel(g)}
                  </option>
                ))}
              </select>
              {armHint(leftArm) && (
                <div className="mt-1 text-xs text-ink-faint">{armHint(leftArm)}</div>
              )}
            </div>
            <div>
              <label className="mb-1 block text-xs text-ink-dim">右臂</label>
              <select
                className="w-full rounded border border-line bg-bg-base px-2 py-1.5 text-sm text-ink"
                value={rightArm}
                onChange={(e) => setRightArm(e.target.value)}
              >
                {groups.map((g) => (
                  <option key={g} value={g}>
                    {armLabel(g)}
                  </option>
                ))}
              </select>
              {armHint(rightArm) && (
                <div className="mt-1 text-xs text-ink-faint">{armHint(rightArm)}</div>
              )}
            </div>
            <div>
              <label className="mb-1 block text-xs text-ink-dim">
                任务（两臂均有记录；结果不同的排在前）
              </label>
              <select
                className="w-full rounded border border-line bg-bg-base px-2 py-1.5 text-sm text-ink"
                value={taskId}
                onChange={(e) => setTaskId(e.target.value)}
                disabled={!bothRun}
              >
                {!bothRun && <option value="">— 无共同任务 —</option>}
                {/* 分组而不只是排序：48 项平铺时，演示真正要看的那几行（两臂结果不同）
                    排在前面但仍然混在一列里，分组之后一眼能看出有几个、在哪。 */}
                {tasks.some((t) => isDivergent(t)) && (
                  <optgroup label={`结果分歧 —— 演示重点（${tasks.filter(isDivergent).length} 题）`}>
                    {tasks.filter(isDivergent).map((t) => taskOption(t))}
                  </optgroup>
                )}
                {tasks.some((t) => !isDivergent(t)) && (
                  <optgroup label={`结果一致 —— 背景对照（${tasks.filter((t) => !isDivergent(t)).length} 题）`}>
                    {tasks.filter((t) => !isDivergent(t)).map((t) => taskOption(t))}
                  </optgroup>
                )}
              </select>
            </div>
          </div>
          {leftArm === rightArm && (
            <div className="mt-2 text-xs text-warn">左右臂相同，请选择不同的两臂对比。</div>
          )}
          {leftArm === 'chtd-seed0' || rightArm === 'chtd-seed0' ? (
            <div className="mt-2 rounded border border-warn/40 bg-warn/5 p-2 text-xs text-warn">
              提示：CHTD seed0 在本任务集（holdout-4）上出现已知的策略退化 —— 48 题中 29 题报告
              “目录不存在”，因为它始终以 <span className="mono">find test*.py</span> 开局、假设了
              不存在的 <span className="mono">src/</span> 目录，而未先列出工作区（通过 7/48）。
              该现象与 seed1 的 38/48 形成对照，是我们记录在案的稳定性发现，不是本方法的代表结果。
            </div>
          ) : null}
        </div>
      )}

      {loading && <div className="text-sm text-ink-dim">加载轨迹…</div>}

      {(left || right) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {[
            { arm: leftArm, s: left },
            { arm: rightArm, s: right },
          ].map(({ arm, s }) => (
            <div key={arm} className="card p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                <span className="font-semibold text-ink">{armLabel(arm)}</span>
                {s?.passed != null && (
                  <span className={`chip ${s.passed ? 'border-ok/50 text-ok' : 'border-bad/50 text-bad'}`}>
                    {s.passed ? '通过' : '失败'}
                  </span>
                )}
                {s?.seconds != null && <span className="mono text-xs text-ink-faint">{s.seconds}s</span>}
                {s?.track_sha256 && (
                  <span className="mono text-xs text-ink-faint" title={s.track_sha256}>
                    轨道 {truncateHash(s.track_sha256)}
                  </span>
                )}
                {s?.model_settings?.[0] && (
                  <span className="mono text-xs text-ink-faint" title="实际执行模型（含运行时切换）">
                    {s.model_settings[0].route}/{s.model_settings[0].model}
                  </span>
                )}
              </div>
              {s ? (
                <Timeline
                  steps={s.steps ?? []}
                  finalAnswer={s.final_answer ?? s.answer}
                  verdict={s.verdict}
                  passed={s.passed}
                />
              ) : (
                <div className="text-sm text-ink-faint">暂无数据</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
