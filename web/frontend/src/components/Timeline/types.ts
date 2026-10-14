import type { Step } from '../../api/types'

export interface TimedStep extends Step {
  // 前端叠加：是否与对侧路分歧（PLAN §5.4）
  divergent?: boolean
  divergenceNote?: string
}
