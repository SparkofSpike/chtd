// 与后端 FastAPI 契约（PLAN §6）对应的类型。snake_case。

export interface HealthEndpoint {
  url: string
  ok: boolean
  model: string | null
  detail: string
}

export interface HealthRoute {
  route: string
  ready: boolean
  kind: 'hosted' | 'local'
  model: string
  base_url: string
}

export interface Health {
  mode: 'replay' | 'live'
  endpoints: {
    base: HealthEndpoint
    adapter: HealthEndpoint
  }
  /** Live 能不能跑，看的是这几条路由。旧后端可能不返回，故可选。 */
  routes?: HealthRoute[]
  runnable_routes?: string[]
  reasons?: string[]
  agnes_cli: { path: string; exists: boolean }
  chtd_root: string
  server_time: string
}

export interface TaskItem {
  id: string
  prompt: string
  why?: string
}

export interface TaskSet {
  name: string
  count: number
  tasks: TaskItem[]
}

export interface TasksResponse {
  sets: TaskSet[]
}

// PLAN §5.1 标准化轨迹
export interface ToolCall {
  name: string
  args: Record<string, unknown>
  ordinal?: number
  result?: { content: string; isError: boolean }
  duration_ms?: number
  policy?: { isReadOnly?: boolean; [k: string]: unknown }
}

export interface Step {
  index: number
  turn?: number
  plan: string
  calls: ToolCall[]
  verifier?: { scope?: string; verdict?: string; reasons?: string[] }
  recovery?: boolean
}

export interface Session {
  session_key: string
  arm: string
  arm_group?: string
  task_id: string
  preset?: string
  track_sha256?: string
  profile_hash?: string
  model_settings?: { slot?: string; route?: string; model?: string; switched?: boolean }[]
  declared_model_settings?: { slot?: string; route?: string; model?: string }[]
  model_switches?: { slot?: string; from?: Record<string, unknown>; to?: Record<string, unknown> }[]
  user_prompt?: string
  steps: Step[]
  final_answer?: string
  answer?: string
  verdict?: string
  passed?: boolean
  check_output?: string
  seconds?: number
  exit?: string
  timed_out?: boolean
  batch?: string
  source_file?: string
}

export interface BatchArm {
  arm: string
  records: number
  passed: number
  source_file: string
}

export interface Batch {
  id: string
  label: string
  source: string
  arms: BatchArm[]
}

export interface SessionIndexItem {
  batch: string
  arm: string
  arm_group?: string
  shard?: string | null
  task_id: string
  passed: boolean
  seconds?: number
  exit?: string
  timed_out?: boolean
  session_key: string
  track_sha256?: string
  preset?: string
  paired?: Record<string, string>
}

export interface SessionsResponse {
  batches: Batch[]
  sessions: SessionIndexItem[]
  filters: { arms: string[]; results: string[]; tasks: string[] }
}

// 导出 JSON 顶层都带 _meta（PLAN §3.6）
export interface ExportMeta {
  generated_at: string
  chtd_root: string
  sources: string[]
  loader_version?: string
  status?: string
}

export interface SummaryCard {
  key: string
  label: string
  value: string
  sub?: string
  support: string
  not_support: string
  source: string
  reproduce?: string
}

export interface SummaryResponse {
  _meta?: ExportMeta
  cards?: SummaryCard[]
  supports?: string[]
  not_supports?: string[]
  [k: string]: unknown
}

export interface ExperimentIndexItem {
  name: string
  title: string
  summary?: string
}

// Arena SSE 帧（PLAN §6.6）
export type ArenaEvent =
  | { event: 'begin'; mode: string; effective_mode: string; lanes: string[]; task_id: string; track_sha256?: string; replay_note?: string }
  | { event: 'step'; lane: string; step_index: number; plan: string }
  | { event: 'tool_call'; lane: string; step_index: number; name: string; args: Record<string, unknown>; ordinal?: number }
  | { event: 'tool_result'; lane: string; step_index: number; name: string; content: string; isError: boolean; duration_ms?: number }
  | { event: 'recovery'; lane: string; step_index: number; after_error: boolean }
  | { event: 'divergence'; step_index: number; kind: string; lanes: Record<string, unknown> }
  | { event: 'final'; lane: string; answer: string; verdict: string }
  | { event: 'end'; lane: string; session: { passed: boolean; check_output?: string; seconds?: number }; track_sha256?: string }
  | { event: 'error'; lane?: string; message: string }
  | { event: 'mode_fallback'; from: string; to: string; reason: string }
