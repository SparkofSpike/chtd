interface Props {
  support: string
  notSupport: string
}

// 证据支持 / 证据不支持 并列卡（PLAN §8 措辞红线，不得夸大）。
export default function EvidencePanel({ support, notSupport }: Props) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div className="card border-l-2 border-l-ok p-4">
        <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-ok">
          <span className="inline-block h-2 w-2 rounded-full bg-ok" />
          证据支持
        </div>
        <p className="text-sm leading-relaxed text-ink-dim">{support}</p>
      </div>
      <div className="card border-l-2 border-l-bad p-4">
        <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-bad">
          <span className="inline-block h-2 w-2 rounded-full bg-bad" />
          证据不支持
        </div>
        <p className="text-sm leading-relaxed text-ink-dim">{notSupport}</p>
      </div>
    </div>
  )
}
