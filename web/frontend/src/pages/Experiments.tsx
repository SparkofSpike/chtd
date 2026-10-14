import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, LabelList, Cell,
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

// dose-ladder 的九个臂分三族，分类来自 runs/holdout-4-20261014/dose-ladder-20261014.md 的表：
// base 是未训练的底线，`chtd (student-state)` 是 CHTD 臂，v1–v5b 全是 teacher-demo 剂量臂。
// 上色按族，因为图上真正要说的事是「一组是基线、一组是 CHTD、其余是对照剂量」，
// 而九个臂同色时这三层看不出区别。
const FAMILY = {
  base: { label: 'base（未训练）', color: '#94a3b8' },
  chtd: { label: 'CHTD（student-state）', color: '#22d3ee' },
  dose: { label: 'teacher-demo 剂量臂', color: '#3b82f6' },
} as const

type Family = keyof typeof FAMILY

function familyOf(arm: string): Family {
  if (arm === 'base') return 'base'
  if (arm.startsWith('chtd')) return 'chtd'
  return 'dose'
}

/** 轴标签用的短名；全名在 tooltip 里还给读者。 */
function shortArm(arm: string): string {
  return arm === 'chtd-student-state' ? 'chtd' : arm
}

// 57 题池化的四族。这里的划分与 dose 梯不同（这四个是池化时的分组），
// 但同一个颜色在两图里必须指同一件事：灰 = 未训练、青 = CHTD。
const POOLED_COLOR: Record<string, string> = {
  base: '#94a3b8',
  chtd: '#22d3ee',
  ctrl: '#fbbf24',
  v5: '#34d399',
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
    .map((r) => ({
      arm: r.arm,
      short: shortArm(r.arm),
      failures: failNum(r.failures),
      family: familyOf(r.arm),
    }))
    .filter((r) => !isNaN(r.failures))
  // ③ 剂量曲线（跳过 base 的区间字符串）
  const doseCurve = rows
    .map((r) => ({ arm: r.arm, rows: r.training_rows, ordinal: ordNum(r.ordinal_mean) }))
    .filter((r) => r.ordinal != null)
  // ② 数据效率：chtd-student-state(9 行) vs v1(33 行)。两个量纲放两张图，
  // 因为同图时 33 会把 2 压成一条线，「数据少 3.7 倍、成绩一样」就看不出来。
  const eff = rows
    .filter((r) => r.arm === 'chtd-student-state' || r.arm === 'v1')
    .map((r) => ({
      arm: r.arm,
      short: r.arm === 'chtd-student-state' ? 'CHTD' : 'teacher-demo v1',
      rows: r.training_rows,
      fails: failNum(r.failures),
    }))

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-ink">实验数据（全部来自真实记录）</h1>
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
                  <XAxis dataKey="short" tick={axisStyle} interval={0} height={40} />
                  <YAxis tick={axisStyle} allowDecimals={false} domain={[0, 5]} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelFormatter={(_, payload) =>
                      (payload?.[0]?.payload as { arm?: string } | undefined)?.arm ?? ''
                    }
                  />
                  <Bar dataKey="failures" name="失败数(/15)" isAnimationActive={false}>
                    {fifteenBar.map((r) => (
                      <Cell key={r.arm} fill={FAMILY[r.family].color} />
                    ))}
                    <LabelList dataKey="failures" position="top" fill="#9aa7b8" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty what={dose.fail ?? '等待数据'} />
            )}
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-ink-faint">
              {(Object.keys(FAMILY) as Family[]).map((f) => (
                <span key={f} className="flex items-center gap-1">
                  <span
                    className="inline-block h-2 w-2 rounded-sm"
                    style={{ background: FAMILY[f].color }}
                  />
                  {FAMILY[f].label}
                </span>
              ))}
            </div>
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-dim">57 池化失败率 %（base/chtd/ctrl/v5）</div>
            {pooled.data?.pooled_bar ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={pooled.data.pooled_bar as Array<Record<string, unknown>>}
                  margin={{ top: 18, right: 8, left: 0, bottom: 0 }}
                >
                  <CartesianGrid stroke="#1f2735" />
                  <XAxis dataKey="family" tick={axisStyle} interval={0} />
                  <YAxis tick={axisStyle} unit="%" />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v: number) => [`${v}%`, '失败率']}
                  />
                  <Bar dataKey="fail_rate" name="失败率 %" isAnimationActive={false}>
                    {(pooled.data.pooled_bar as Array<Record<string, unknown>>).map((r) => (
                      <Cell
                        key={String(r.family)}
                        fill={POOLED_COLOR[String(r.family)] ?? '#38bdf8'}
                      />
                    ))}
                    <LabelList dataKey="fail_rate" position="top" fill="#9aa7b8" fontSize={11}
                      formatter={(v: number) => `${v}%`} />
                  </Bar>
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
        <h2 className="text-sm font-semibold text-ink">
          ② 数据效率：9 条 student-state ≈ 33 条 teacher-demo（同为 2/15 失败）
        </h2>
        <p className="mt-1 text-xs text-ink-dim">
          两个量纲分开画：数据量差约 3.7 倍，成绩完全相同。
        </p>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <div>
            <div className="mb-1 text-xs text-ink-dim">训练行数（少者胜）</div>
            {eff.length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={eff} margin={{ top: 18, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#1f2735" />
                  <XAxis dataKey="short" tick={axisStyle} interval={0} />
                  <YAxis tick={axisStyle} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [`${v} 行`, '训练行数']} />
                  <Bar dataKey="rows" fill="#fbbf24" name="训练行数" isAnimationActive={false}>
                    <LabelList dataKey="rows" position="top" fill="#fbbf24" fontSize={12} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty what={dose.fail ?? '等待数据'} />
            )}
          </div>
          <div>
            <div className="mb-1 text-xs text-ink-dim">诊断失败数（满分 15）</div>
            {eff.length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={eff} margin={{ top: 18, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#1f2735" />
                  <XAxis dataKey="short" tick={axisStyle} interval={0} />
                  <YAxis tick={axisStyle} domain={[0, 15]} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [`${v} / 15`, '失败数']} />
                  <Bar dataKey="fails" fill="#34d399" name="失败数" isAnimationActive={false}>
                    <LabelList dataKey="fails" position="top" fill="#34d399" fontSize={12} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty what={dose.fail ?? '等待数据'} />
            )}
          </div>
        </div>
        <p className="mt-3 rounded border border-line bg-bg-base px-3 py-2 text-xs text-ink-dim">
          读完这两张图只需要一个动作：看左图差多少，再看右图差多少 ——{' '}
          <span className="text-ink">左图差 24 行，右图差 0</span>。这就是「9 行 ≈ 33 行」的全部含义。
        </p>
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
              <BarChart data={dh.data.determinism as Array<Record<string, unknown>>}
                margin={{ top: 18, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#1f2735" />
                <XAxis dataKey="arm" tick={axisStyle} interval={0} />
                <YAxis tick={axisStyle} allowDecimals={false} domain={[0, 12]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="differing" fill="#f87171" name="两次评估不同答案数(/15)" isAnimationActive={false}>
                  <LabelList dataKey="differing" position="top" fill="#f87171" fontSize={12} />
                </Bar>
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
