import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { api } from '../api/client'
import type { Health } from '../api/types'

const nav = [
  { to: '/', label: 'Overview', end: true },
  { to: '/experiments', label: 'Experiments' },
  { to: '/arena', label: 'Arena / Replay' },
  { to: '/live', label: 'Live' },
  { to: '/reproduce', label: 'Reproduce' },
]

function Dot({ ok }: { ok: boolean | undefined }) {
  const color =
    ok === undefined ? 'bg-ink-faint' : ok ? 'bg-ok' : 'bg-bad'
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} />
}

// 顶栏：产品名 + Live/Replay 徽标 + 路由健康灯（轮询 /api/health 5s）。
//
// 灯看的是**路由**而不是固定端口：Live 能不能跑取决于至少一条路由就绪，而路由是什么、
// 用哪个模型、托管还是本地，都是运行时可换的（见 Live 页的「模型配置」）。
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

        {/* 路由健康灯 */}
        <div className="flex flex-wrap items-center gap-3 text-xs text-ink-dim">
          {(health?.routes ?? []).map((r) => (
            <span key={r.route} className="flex items-center gap-1.5">
              <Dot ok={r.ready} />
              {r.route}
              <span className="mono text-ink-faint">{r.model}</span>
            </span>
          ))}
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
