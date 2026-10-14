import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { api } from '../api/client'
import type { Health } from '../api/types'

const nav = [
  { to: '/', label: 'Overview', end: true },
  { to: '/experiments', label: 'Experiments' },
  { to: '/arena', label: 'Arena / Replay' },
  { to: '/reproduce', label: 'Reproduce' },
]

function Dot({ ok }: { ok: boolean | undefined }) {
  const color =
    ok === undefined ? 'bg-ink-faint' : ok ? 'bg-ok' : 'bg-bad'
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} />
}

// 顶栏：产品名 + Live/Replay 徽标 + base/adapter 健康灯（轮询 /api/health 5s）。
export default function TopBar() {
  const [health, setHealth] = useState<Health | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const h = await api.health()
        if (!stop) {
          setHealth(h)
          setErr(null)
        }
      } catch (e) {
        if (!stop) setErr((e as Error).message)
      }
    }
    tick()
    const id = setInterval(tick, 5000)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [])

  const mode = health?.mode ?? 'replay'

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg-base/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
        <div className="mr-2">
          <div className="text-sm font-semibold text-ink">
            CHTD
            <span className="ml-2 text-xs font-normal text-ink-dim">
              Co-Harness Targeted Distillation
            </span>
          </div>
        </div>

        {/* 模式徽标 */}
        <span
          className={`chip ${
            mode === 'live'
              ? 'border-accent/60 text-accent'
              : 'border-ink-faint text-ink-dim'
          }`}
        >
          <Dot ok={mode === 'live' ? true : undefined} />
          {mode === 'live' ? 'Live' : 'Replay'}
        </span>

        {/* 模型健康灯 */}
        <div className="flex items-center gap-3 text-xs text-ink-dim">
          <span className="flex items-center gap-1.5">
            <Dot ok={health?.endpoints.base.ok} />
            base
            <span className="mono text-ink-faint">:8000</span>
          </span>
          <span className="flex items-center gap-1.5">
            <Dot ok={health?.endpoints.adapter.ok} />
            adapter
            <span className="mono text-ink-faint">:8001</span>
          </span>
          {err && <span className="text-bad">后端未连接</span>}
        </div>

        <nav className="ml-auto flex flex-wrap items-center gap-1">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `rounded px-2.5 py-1 text-sm ${
                  isActive
                    ? 'bg-accent/15 text-accent'
                    : 'text-ink-dim hover:text-ink'
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
