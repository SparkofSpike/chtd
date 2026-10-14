import { useState } from 'react'

interface Props {
  label: string
  value: string
  sub?: string
  source?: string
  accent?: boolean
}

// 核心数字卡：数字 + 来源 tooltip（PLAN §2 概览数字卡）。
export default function StatCard({ label, value, sub, source, accent }: Props) {
  const [show, setShow] = useState(false)
  return (
    <div
      className="card relative p-4"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <div className="text-xs text-ink-dim">{label}</div>
      <div
        className={`mt-1 text-2xl font-semibold ${
          accent ? 'text-accent' : 'text-ink'
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-ink-faint">{sub}</div>}
      {source && show && (
        <div className="absolute bottom-full left-0 z-20 mb-2 w-72 rounded border border-line bg-bg-raised p-2 text-xs text-ink-dim shadow-lg">
          <span className="text-ink-dim">来源：</span>
          <span className="mono">{source}</span>
        </div>
      )}
    </div>
  )
}
