import { Outlet } from 'react-router-dom'
import TopBar from './TopBar'

export default function Layout() {
  return (
    <div className="min-h-full">
      <TopBar />
      <main className="mx-auto max-w-[1400px] px-4 py-6">
        <Outlet />
      </main>
      <footer className="mx-auto max-w-[1400px] px-4 pb-8 text-xs text-ink-faint">
        Experiments / Arena 读的是 checked-in 的真实 ledger，数字可溯源；Live 在同一页上真实执行
        —— 需要 agnes CLI 与一条就绪的路由（托管 teacher 或自备的 OpenAI 兼容端点）。
      </footer>
    </div>
  )
}
