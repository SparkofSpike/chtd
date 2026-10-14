import { useState } from 'react'
import type { TimedStep } from './types'

function truncate(s: string, n = 280): string {
  if (!s) return ''
  return s.length > n ? s.slice(0, n) + ' …' : s
}

interface Props {
  step: TimedStep
  isLast?: boolean
}

// 单步节点：规划 → 工具调用(名+参数) → 工具返回(content/error) → 恢复标记。
export default function StepNode({ step, isLast }: Props) {
  const [expanded, setExpanded] = useState(false)
  const border = step.divergent
    ? 'border-warn'
    : step.calls.some((c) => c.result?.isError)
      ? 'border-bad/60'
      : 'border-line'

  return (
    <div className="relative pl-8">
      {/* 竖线 */}
      {!isLast && (
        <span className="absolute left-[9px] top-6 h-full w-px bg-line" />
      )}
      {/* 节点圆点 */}
      <span
        className={`absolute left-0 top-1.5 h-[19px] w-[19px] rounded-full border-2 ${
          step.recovery
            ? 'border-warn bg-warn/20'
            : step.divergent
              ? 'border-warn bg-warn/20'
              : step.calls.some((c) => c.result?.isError)
                ? 'border-bad bg-bad/20'
                : 'border-accent bg-bg-panel'
        }`}
      />
      <div className={`card mb-3 border ${border} p-3`}>
        <div className="flex items-center gap-2 text-xs">
          <span className="mono text-ink-dim">#{step.index + 1}</span>
          {step.recovery && (
            <span className="chip border-warn/50 text-warn">错误恢复</span>
          )}
          {step.divergent && (
            <span className="chip border-warn/50 text-warn">
              分歧{step.divergenceNote ? `：${step.divergenceNote}` : ''}
            </span>
          )}
        </div>

        {step.plan && (
          <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-ink">
            {expanded ? step.plan : truncate(step.plan)}
            {step.plan.length > 280 && (
              <button
                className="ml-1 text-accent underline"
                onClick={() => setExpanded((v) => !v)}
              >
                {expanded ? '收起' : '展开'}
              </button>
            )}
          </p>
        )}

        {/* 工具调用 */}
        <div className="mt-2 space-y-2">
          {step.calls.map((c, i) => (
            <div key={i} className="rounded border border-line bg-bg-base p-2">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="mono rounded bg-accent/15 px-1.5 py-0.5 text-accent">
                  {c.name}
                </span>
                {c.duration_ms != null && (
                  <span className="mono text-ink-faint">
                    {(c.duration_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              {c.args && Object.keys(c.args).length > 0 && (
                <pre className="mono mt-1 overflow-x-auto text-[11px] text-ink-dim">
                  {JSON.stringify(c.args, null, 0)}
                </pre>
              )}
              {c.result && (
                <div
                  className={`mt-1.5 rounded border-l-2 pl-2 text-[11px] leading-relaxed ${
                    c.result.isError
                      ? 'border-bad text-bad/90'
                      : 'border-ok/60 text-ink-dim'
                  }`}
                >
                  <div className="mono text-ink-faint">
                    {c.result.isError ? 'ERROR' : 'RESULT'}
                  </div>
                  <pre className="whitespace-pre-wrap break-words">
                    {expanded ? c.result.content : truncate(c.result.content, 320)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
