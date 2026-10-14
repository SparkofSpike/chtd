interface Props {
  source?: string
  reproduce?: string
}

// 来源脚注：来源文件 + 复现脚本（PLAN §4 每图必带）。
export default function SourceFootnote({ source, reproduce }: Props) {
  if (!source && !reproduce) return null
  return (
    <div className="mt-2 border-t border-line pt-2 text-xs text-ink-faint">
      {source && (
        <div>
          <span className="text-ink-dim">来源：</span>
          <span className="mono">{source}</span>
        </div>
      )}
      {reproduce && (
        <div>
          <span className="text-ink-dim">复现：</span>
          <span className="mono">{reproduce}</span>
        </div>
      )}
    </div>
  )
}
