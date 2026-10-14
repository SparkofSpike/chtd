// Live 模式的类型与客户端。
//
// 与 Arena 的回放共用事件词汇（begin/step/tool_call/tool_result/final/end），但多出两样
// 运行前才知道的东西：这台机器能不能跑（能力探测），以及能跑哪些任务。这两样决定页面上
// 什么可以点、什么只能看。

import { getJsonPublic, openSse } from './client'

export interface CredentialStatus {
  source: 'env' | 'file' | 'none'
  env: string
  ok: boolean
  path?: string
}

export interface RouteStatus {
  route: string
  kind: 'hosted' | 'local'
  base_url: string
  model: string
  credential: CredentialStatus
  reachable: boolean | null
  ready: boolean
  note: string
  endpoint?: { url: string; ok: boolean; model: string | null; detail: string }
}

export interface LiveCapabilities {
  mode: 'live' | 'replay'
  agnes_cli: { path: string; exists: boolean }
  node: { bin: string; ok: boolean; version: string; detail: string }
  routes: RouteStatus[]
  runnable_routes: string[]
  reasons: string[]
  defaults: { route: string; model: string; preset: string; timeout: number }
  scratch: string
  tasks: number
  tasks_with_fixtures: number
}

export interface LiveTask {
  task_id: string
  set: string
  prompt: string
  check: string | null
  expect: string[]
  forbid: string[]
  why: string
  fixture_ready: boolean
  fixture_path: string | null
  files: string[]
}

export interface LiveTasksResponse {
  tasks: LiveTask[]
  count: number
  runnable: number
}

/** 一条来自 /api/live/run 的 SSE 帧。事件名在 `event` 上，其余字段随事件而定。 */
export interface LiveEvent {
  event: string
  [key: string]: unknown
}

export interface LiveRunOptions {
  task_id?: string
  prompt?: string
  route?: string
  model?: string
  base_url?: string
  api_key?: string
  preset?: string
  timeout?: number
  run_check?: boolean
}

export const liveApi = {
  capabilities: () => getJsonPublic<LiveCapabilities>('/api/live/capabilities'),
  tasks: () => getJsonPublic<LiveTasksResponse>('/api/live/tasks'),
}

export async function runLive(
  opts: LiveRunOptions,
  onEvent: (ev: LiveEvent) => void,
  onFatal: (msg: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  return openSse<LiveEvent>('/api/live/run', opts, onEvent, onFatal, signal)
}
