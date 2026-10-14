import SvgDiagram from '../components/SvgDiagram'
import StatCard from '../components/StatCard'
import EvidencePanel from '../components/EvidencePanel'

// 核心数字来自 PLAN.md §2/§3 与 docs/findings-20261014.md（已冻结，逐字引用）。
const cards = [
  {
    label: '15 题诊断失败（base → 训练配置）',
    value: '4–5 → 0–2',
    sub: '联合概率 ≤3.4×10⁻⁹（保守 2.6×10⁻⁵）',
    source: 'runs/holdout-4-20261014/dose-ladder-20261014.md',
    accent: true,
  },
  {
    label: '57 题池化失败率（base → CHTD 族）',
    value: '31.6% → 17.5%',
    sub: '18/57 → 10/57；McNemar p=0.077',
    source: 'runs/holdout-4-20261014/pooled-analysis-20261014.md',
  },
  {
    label: '数据效率（student-state vs teacher-demo）',
    value: '9 行 ≈ 33 行',
    sub: '同为 2/15 失败；约 1/4 数据（单格边际）',
    source: 'runs/holdout-4-20261014/dose-ladder-20261014.md',
  },
  {
    label: '确定性测量地板',
    value: 'base 6 次逐字一致',
    sub: 'adapter 同权重 9/15 答案不同（0.33 等级点）',
    source: 'eval/replication_floor.py · eval/base_spread.py',
  },
  {
    label: '评分器校准 / 功效',
    value: 'gate↔人 75%',
    sub: '人↔人 84%；30 题 MDE=26.6pp',
    source: 'runs/holdout-3-20261014/handread-20261014.json',
  },
]

export default function Overview() {
  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-xl font-semibold text-ink">
          跨 harness 蒸馏的两大失败
        </h1>
        <p className="mt-2 max-w-4xl text-sm leading-relaxed text-ink-dim">
          学生在教师自有脚手架上学习轨迹，导致：① 学生学到脚手架而非能力——换 harness 后迁移失效；
          ② 评估奖励错误指标——长度/延迟/token 花费与目标能力同动，改进被归因错误。
          CHTD（Co-Harness Targeted Distillation）让师生共用同一冻结轨道，在学生到达的状态上由教师决策标注，
          并剥离与目标能力相关但非因果的指标。
        </p>
      </section>

      <section className="card p-4">
        <h2 className="mb-2 text-sm font-semibold text-ink">方法三承诺</h2>
        <SvgDiagram />
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold text-ink">核心结论（数字可溯源，悬停查看来源）</h2>
        {/* 五张卡要铺满两行：前三张各占三列的 2/6，后两张各占一半。用三列网格的话
            第二行会空出三分之一，看着像没画完。 */}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {cards.map((c, i) => (
            <StatCard
              key={c.label}
              {...c}
              className={
                i < 3 ? 'lg:col-span-2' : i === cards.length - 1 ? 'sm:col-span-2 lg:col-span-3' : 'lg:col-span-3'
              }
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold text-ink">证据支持 / 证据不支持</h2>
        <div className="space-y-3">
          <EvidencePanel
            support="微调在设计指定端点（诊断失败率）显著降失败：base 4–5/15，全部训练配置 0–2/15；57 池化 31.6%→17.5%，三集同向（p=0.077，强提示而非证实）；9 行≈33 行（提示性）。"
            notSupport="不主张「CHTD 全面优于其他方法」（ordinal 未超噪、未胜 teacher-own）；v5 是完整教师演示（10× 数据），非 CHTD 形态证据；33–93 行剂量平坦，不宣称 seed 稳定/数据越多越好；85%/89% judge 一致来自双重选择面板，不能替代人工。"
          />
        </div>
      </section>
    </div>
  )
}
