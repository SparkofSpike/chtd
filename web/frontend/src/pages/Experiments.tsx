import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, CartesianGrid,
} from 'recharts'
import SourceFootnote from '../components/SourceFootnote'
import { api } from '../api/client'

function Empty({ what }: { what: string }) {
  return (
    <div className="flex h-40 items-center justify-center rounded border border-dashed border-line text-sm text-ink-faint">
      暂无数据（{what}）
    </div>
  )
}

interface DoseRow {
  arm: string
  training_rows: number
  ordinal_mean: number | string
  failures: string
  gate?: string
  note?: string
}

// 解析 "2/15" → 2；"4-5/15" → 4（取下界，柱状图可比）
function failNum(f: string): number {
  const m = String(f).match(/(\d+)(?:-(\d+))?\s*\/\s*15/)
  return m ? parseInt(m[1], 10) : NaN
}
function ordNum(v: number | string): number | null {
  const n = typeof v === 'number' ? v : parseFloat(v)
  return isNaN(n) ? null : n
}

function useExp(name: string) {
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  const [fail, setFail] = useState<string | null>(null)
  useEffect(() => {
    let on = true
    api.experiments(name)
      .then((d) => on && setData(d))
      .catch((e) => on && setFail(e.message))
    return () => {
      on = false
    }
  }, [name])
  return { data, fail }
}

const axisStyle = { fontSize: 11, fill: '#9aa7b8' }
const tooltipStyle = { background: '#141a26', border: '1px solid #1f2735' }

export default function Experiments() {
  const dose = useExp('dose_ladder')
  const pooled = useExp('pooled')
  const dh = useExp('determinism')

  const rows = (dose.data?.rows as DoseRow[] | undefined) ?? []

  // ① 15 题失败数 bar
  const fifteenBar = rows
    .map((r) => ({ arm: r.arm, failures: failNum(r.failures) }))
    .filter((r) => !isNaN(r.failures))
  // ③ 剂量曲线（跳过 base 的区间字符串）
  const doseCurve = rows
    .map((r) => ({ arm: r.arm, rows: r.training_rows, ordinal: ordNum(r.ordinal_mean) }))
    .filter((r) => r.ordinal != null)
  // ② 数据效率：chtd-student-state(9) vs v1(33)
  const eff = rows
    .filter((r) => r.arm === 'chtd-student-state' || r.arm === 'v1')
    .map((r) => ({ arm: r.arm, 训练行数: r.training_rows, 失败数: failNum(r.failures) }))

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-ink">Experiments Dashboard（全真实数据）</h1>
      <p className="text-sm text-ink-dim">
        图表数据来自后端导出的真实 ledger（<span className="mono">/api/experiments/&lt;name&gt;</span>）。
      </p>

      {/* ① 失败率对比 */}
      <section className="card p-4">
        <h2 className="text-sm font-semibold text-ink">① 失败数对比：15 题 base vs 训练配置 + 57 池化失败率</h2>
        <div className="mt-3 grid gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-1 text-xs text-ink-dim">15 题失败数（base 4–5 → 训练 0–2）</div>
            {fifteenBar.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={fifteenBar}>
                  <CartesianGrid stroke="#1f2735" />
                  <XAxis dataKey="arm" tick={axisStyle} angle={-20} textAnchor="end" height={60} />
                  <YAxis tick={axisStyle} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="failures" fill="#38bdf8" name="失败数(/15)" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty what={dose.fail ?? '等待数据'} />
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-dim">57 池化失败率 %（base/chtd/ctrl/v5）</div>
            {pooled.data?.pooled_bar ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={pooled.data.pooled_bar as Array<Record<string, unknown>>}>
                  <CartesianGrid stroke="#1f2735" />
                  <XAxis dataKey="family" tick={axisStyle} />
                  <YAxis tick={axisStyle} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="fail_rate" fill="#22d3ee" name="失败率 %" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty what={pooled.fail ?? '等待数据'} />
            )}
          </div>
        </div>
        <SourceFootnote
          source="runs/holdout-4-20261014/{dose-ladder,pooled-analysis}-20261014.md"
          reproduce="eval/failure_rate.py · eval/pooled_effect.py"
        />
      </section>

      {/* ② 数据效率 9 vs 33 */}
      <section className="card p-4">
        <h2 className="text-sm font-semibold text-ink">② 数据效率：9 条 student-state ≈ 33 条 teacher-demo（同为 2/15 失败）</h2>
        <div className="mt-3">
          {eff.length ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={eff}>
                <CartesianGrid stroke="#1f2735" />
                <XAxis dataKey="arm" tick={axisStyle} />
                <YAxis tick={axisStyle} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
                <Bar dataKey="训练行数" fill="#fbbf24" isAnimationActive={false} />
                <Bar dataKey="失败数" fill="#34d399" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty what={dose.fail ?? '等待数据'} />
          )}
        </div>
        <SourceFootnote
          source="runs/holdout-4-20261014/dose-ladder-20261014.md"
          reproduce="eval/dose_statistics.py"
        />
      </section>

      {/* ③ 剂量曲线 33–93 */}
      <section className="card p-4">
        <h2 className="text-sm font-semibold text-ink">③ 剂量响应：training rows vs ordinal mean（33–93 平坦）</h2>
        <div className="mt-3">
          {doseCurve.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={doseCurve}>
                <CartesianGrid stroke="#1f2735" />
                <XAxis dataKey="rows" tick={axisStyle} type="number" />
                <YAxis tick={axisStyle} domain={[1.4, 1.9]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="ordinal" stroke="#38bdf8" name="ordinal mean"
                  dot={{ r: 4 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Empty what={dose.fail ?? '等待数据'} />
          )}
        </div>
        <SourceFootnote
          source="runs/holdout-4-20261014/dose-ladder-20261014.md"
          reproduce="eval/dose_response.py"
        />
      </section>

      {/* ④ 确定性 */}
      <section className="card p-4">
        <h2 className="text-sm font-semibold text-ink">④ 确定性：base 6 次逐字一致 vs adapter 同权重 9/15 不同</h2>
        <div className="mt-3">
          {dh.data?.determinism ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={dh.data.determinism as Array<Record<string, unknown>>}>
                <CartesianGrid stroke="#1f2735" />
                <XAxis dataKey="arm" tick={axisStyle} />
                <YAxis tick={axisStyle} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="differing" fill="#f87171" name="两次评估不同答案数(/15)" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Empty what={dh.fail ?? '等待数据'} />
          )}
        </div>
        <SourceFootnote
          source="runs/holdout-4-20261014/dose-ladder-20261014.md"
          reproduce="eval/replication_floor.py"
        />
      </section>
    </div>
  )
}
