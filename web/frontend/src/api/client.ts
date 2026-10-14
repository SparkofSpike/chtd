import type {
  ArenaEvent,
  Health,
  SessionsResponse,
  Session,
  TasksResponse,
} from './types'

// 开发时走 vite 代理；生产由后端静态托管，同源。
const BASE = ''

export class ApiError extends Error {
  status: number
  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`)
    this.status = status
  }
}

async function getJson<T>(path: string): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(BASE + path, { headers: { Accept: 'application/json' } })
  } catch (e) {
    throw new ApiError(0, `网络不可达：${(e as Error).message}`)
  }
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const j = await resp.json()
      detail = j.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new ApiError(resp.status, detail)
  }
  return (await resp.json()) as T
}

export const api = {
  health: () => getJson<Health>('/api/health'),
  tasks: () => getJson<TasksResponse>('/api/tasks'),
  experiments: (name: string) =>
    getJson<Record<string, unknown>>(`/api/experiments/${name}`),
  experimentsIndex: () => getJson<{ items?: unknown[] }>('/api/experiments'),
  sessions: (params?: { q?: string; arm?: string; result?: string; batch?: string }) => {
    const qs = new URLSearchParams()
    if (params?.q) qs.set('q', params.q)
    if (params?.arm) qs.set('arm', params.arm)
    if (params?.result) qs.set('result', params.result)
    if (params?.batch) qs.set('batch', params.batch)
    const s = qs.toString()
    return getJson<SessionsResponse>(`/api/sessions${s ? '?' + s : ''}`)
  },
  session: (sessionKey: string) =>
    getJson<Session>(`/api/sessions/${encodeURIComponent(sessionKey)}`),
}

export interface ArenaRunOptions {
  mode?: 'replay' | 'live'
  task_id: string
  arms?: string[]
  collection?: string
  speed?: number
}

// POST /api/arena/run 返回 text/event-stream；EventSource 不支持 POST，
// 因此用 fetch + ReadableStream 手动解析 SSE 帧。
export async function runArena(
  opts: ArenaRunOptions,
  onEvent: (ev: ArenaEvent) => void,
  onFatal: (msg: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  let resp: Response
  try {
    resp = await fetch(BASE + '/api/arena/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        arms: ['base', 'chtd'],
        speed: 1,
        ...opts,
      }),
      signal,
    })
  } catch (e) {
    onFatal(`无法连接后端：${(e as Error).message}`)
    return
  }
  if (!resp.ok || !resp.body) {
    let detail = `HTTP ${resp.status}`
    try {
      const j = await resp.json()
      detail = j.detail ?? detail
    } catch {
      /* ignore */
    }
    onFatal(detail)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const dispatch = (rawBlock: string) => {
    // 一个 SSE block 可能含多行 event:/data:
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of rawBlock.split('\n')) {
      const t = line.trimEnd()
      if (t.startsWith(':')) continue // 心跳注释帧
      if (t.startsWith('event:')) eventName = t.slice(6).trim()
      else if (t.startsWith('data:')) dataLines.push(t.slice(5).trim())
    }
    if (dataLines.length === 0) return
    const dataStr = dataLines.join('\n')
    let data: Record<string, unknown> = {}
    try {
      data = JSON.parse(dataStr)
    } catch {
      return
    }
    onEvent({ event: eventName, ...data } as ArenaEvent)
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        dispatch(block)
      }
    }
    if (buffer.trim()) dispatch(buffer)
  } catch (e) {
    if ((e as Error).name !== 'AbortError') {
      onFatal(`流中断：${(e as Error).message}`)
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      /* ignore */
    }
  }
}
