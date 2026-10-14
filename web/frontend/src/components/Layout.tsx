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
        只读演示：数据来自 checked-in 的 runs/ 与 data/ ledger，数字可溯源；Live 需在装有
        agnes-harness 构建的机器并启动 base/adapter 两端点。
      </footer>
    </div>
  )
}
