// 方法三承诺自绘 SVG（PLAN §2：Co-harness / Targeted Distillation / Distractor Stripping）。
// 纯 SVG，不使用任何图片库；轨道/分叉/剥离三态。
export default function SvgDiagram() {
  return (
    <svg
      viewBox="0 0 960 300"
      className="h-auto w-full"
      role="img"
      aria-label="CHTD 三承诺示意图"
    >
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9aa7b8" />
        </marker>
      </defs>

      {/* 面板分隔 */}
      <line x1="320" y1="20" x2="320" y2="280" stroke="#1f2735" />
      <line x1="640" y1="20" x2="640" y2="280" stroke="#1f2735" />

      {/* ① Co-harness：teacher / student 同一 loop 并行轨道 */}
      <text x="160" y="40" textAnchor="middle" fill="#38bdf8" fontSize="14" fontWeight="600">
        ① Co-harness 同轨
      </text>
      <text x="40" y="80" fill="#9aa7b8" fontSize="11">teacher</text>
      <line x1="90" y1="76" x2="280" y2="76" stroke="#38bdf8" strokeWidth="2" markerEnd="url(#arrow)" />
      <circle cx="120" cy="76" r="4" fill="#38bdf8" />
      <circle cx="180" cy="76" r="4" fill="#38bdf8" />
      <circle cx="240" cy="76" r="4" fill="#38bdf8" />
      <text x="40" y="130" fill="#9aa7b8" fontSize="11">student</text>
      <line x1="90" y1="126" x2="280" y2="126" stroke="#22d3ee" strokeWidth="2" markerEnd="url(#arrow)" />
      <circle cx="120" cy="126" r="4" fill="#22d3ee" />
      <circle cx="180" cy="126" r="4" fill="#22d3ee" />
      <circle cx="240" cy="126" r="4" fill="#22d3ee" />
      <text x="160" y="180" textAnchor="middle" fill="#64748b" fontSize="11">
        同一工具集 / 同一 loop / 同一冻结 preset
      </text>
      <text x="160" y="198" textAnchor="middle" fill="#64748b" fontSize="11">
        唯一变量：模型
      </text>
      <rect x="70" y="220" width="180" height="34" rx="6" fill="#141a26" stroke="#1f2735" />
      <text x="160" y="241" textAnchor="middle" fill="#9aa7b8" fontSize="11">
        append-only ledger 逐 step 可比
      </text>

      {/* ② Targeted Distillation：fork，teacher 落在 student 状态 */}
      <text x="480" y="40" textAnchor="middle" fill="#38bdf8" fontSize="14" fontWeight="600">
        ② Targeted Distillation 分叉标注
      </text>
      {/* student prefix */}
      <line x1="350" y1="110" x2="470" y2="110" stroke="#22d3ee" strokeWidth="2" />
      <circle cx="380" cy="110" r="4" fill="#22d3ee" />
      <circle cx="430" cy="110" r="4" fill="#22d3ee" />
      <text x="410" y="98" textAnchor="middle" fill="#22d3ee" fontSize="10">student 先跑</text>
      {/* fork 点 */}
      <circle cx="470" cy="110" r="5" fill="#fbbf24" />
      <text x="478" y="100" fill="#fbbf24" fontSize="10">fork</text>
      {/* teacher dropped in */}
      <line x1="470" y1="110" x2="610" y2="80" stroke="#38bdf8" strokeWidth="2" strokeDasharray="4 3" markerEnd="url(#arrow)" />
      <line x1="470" y1="110" x2="610" y2="140" stroke="#22d3ee" strokeWidth="2" strokeDasharray="4 3" markerEnd="url(#arrow)" />
      <text x="616" y="82" fill="#38bdf8" fontSize="10">teacher 决策=label</text>
      <text x="616" y="142" fill="#22d3ee" fontSize="10">student 实际状态</text>
      <text x="480" y="190" textAnchor="middle" fill="#64748b" fontSize="11">
        在 student 到达的状态处分叉
      </text>
      <text x="480" y="208" textAnchor="middle" fill="#64748b" fontSize="11">
        分歧处 teacher 决策成为监督信号
      </text>
      <rect x="380" y="225" width="200" height="30" rx="6" fill="#141a26" stroke="#1f2735" />
      <text x="480" y="244" textAnchor="middle" fill="#9aa7b8" fontSize="11">
        监督落在需要帮助的状态，而非 teacher 舒适区
      </text>

      {/* ③ Distractor Stripping：剥离相关但非因果的指标 */}
      <text x="800" y="40" textAnchor="middle" fill="#38bdf8" fontSize="14" fontWeight="600">
        ③ Distractor Stripping 剥离
      </text>
      {/* 一堆指标 */}
      {[0, 1, 2, 3, 4].map((i) => (
        <g key={i}>
          <rect x="670" y={70 + i * 30} width="120" height="20" rx="4" fill="#1a2231" stroke="#1f2735" />
          <text x="730" y={84 + i * 30} textAnchor="middle" fill="#9aa7b8" fontSize="10">
            {['长度', '延迟', 'token 花费', '无关子任务', '目标能力'][i]}
          </text>
        </g>
      ))}
      {/* 剥离箭头：前四个划掉，最后保留 */}
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <line x1="795" y1={80 + i * 30} x2="870" y2={80 + i * 30} stroke="#f87171" strokeWidth="1.5" markerEnd="url(#arrow)" />
          <line x1="800" y1={76 + i * 30} x2="780" y2={88 + i * 30} stroke="#f87171" strokeWidth="1" />
        </g>
      ))}
      <line x1="795" y1="190" x2="870" y2="190" stroke="#34d399" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <text x="888" y="84" fill="#f87171" fontSize="10">剥离</text>
      <text x="888" y="194" fill="#34d399" fontSize="10">保留</text>
      <text x="800" y="240" textAnchor="middle" fill="#64748b" fontSize="11">
        另用 stripping ladder 去 harness 复测
      </text>
      <text x="800" y="258" textAnchor="middle" fill="#64748b" fontSize="11">
        仅在训练轨道成立 → 记为 scaffold overfit
      </text>
    </svg>
  )
}
