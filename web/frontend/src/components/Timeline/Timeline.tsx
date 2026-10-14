import type { TimedStep } from './types'
import StepNode from './StepNode'

interface Props {
  steps: TimedStep[]
  finalAnswer?: string
  verdict?: unknown
  passed?: boolean
}

// verdict 后端可能是 string 或 {verdict, reasons} 对象，统一成短串。
function verdictText(v: unknown): string {
  if (!v) return ''
  if (typeof v === 'string') return v
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>
    return String(o.verdict ?? o.scope ?? '')
  }
  return String(v)
}

// 自绘轨迹时间线：步骤序列 + 最终答案 + 评分判定（PLAN §2 Arena/Replay）。
export default function Timeline({ steps, finalAnswer, verdict, passed }: Props) {
  const vtext = verdictText(verdict)
  if (!steps || steps.length === 0) {
    return <div className="card p-4 text-sm text-ink-faint">暂无轨迹步骤数据</div>
  }
  return (
    <div>
      {steps.map((s, i) => (
        <StepNode key={i} step={s} isLast={i === steps.length - 1} />
      ))}

      {/* 最终答案 */}
      {finalAnswer && (
        <div className="mb-3 pl-8">
          <div className="card border-accent/40 p-3">
            <div className="text-xs font-semibold text-accent">最终答案 final answer</div>
            <pre className="mt-1 whitespace-pre-wrap break-words text-sm text-ink">
              {finalAnswer}
            </pre>
          </div>
        </div>
      )}

      {/* 评分判定 */}
      {(passed != null || vtext) && (
        <div className="pl-8">
          <div
            className={`card inline-block px-3 py-2 text-sm font-semibold ${
              passed ? 'border-ok/50 text-ok' : 'border-bad/50 text-bad'
            }`}
          >
            判定：{passed ? '通过 passed' : '失败 failed'}
            {vtext ? ` · ${vtext}` : ''}
          </div>
        </div>
      )}
    </div>
  )
}
