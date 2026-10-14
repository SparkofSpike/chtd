// PLAN §9.4 色板集中导出，供 Recharts 与自绘 SVG 复用。
export const colors = {
  base: '#0b0e14',
  panel: '#141a26',
  raised: '#1a2231',
  line: '#1f2735',
  ink: '#e6edf6',
  inkDim: '#9aa7b8',
  inkFaint: '#64748b',
  accent: '#38bdf8',
  accentSoft: '#22d3ee',
  ok: '#34d399',
  bad: '#f87171',
  warn: '#fbbf24',
} as const

// 臂（arm）配色：base 灰蓝、chtd 青、control 琥珀、v5 绿。
export const armColors: Record<string, string> = {
  base: '#94a3b8',
  chtd: '#22d3ee',
  'chtd-seed0': '#38bdf8',
  'chtd-seed1': '#22d3ee',
  ctrl: '#fbbf24',
  control: '#fbbf24',
  v5: '#34d399',
}

export function armColor(arm: string): string {
  const k = arm.toLowerCase()
  if (k.includes('chtd')) return armColors.chtd
  if (k.includes('base')) return armColors.base
  if (k.includes('control') || k.includes('ctrl')) return armColors.ctrl
  if (k.includes('v5')) return armColors.v5
  return '#a78bfa'
}
