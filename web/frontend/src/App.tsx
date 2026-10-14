import { HashRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import Arena from './pages/Arena'
import Experiments from './pages/Experiments'
import Live from './pages/Live'
import Reproduce from './pages/Reproduce'

// 用 HashRouter 保证 dist/ 静态托管（含 file:// 或后端 SPA fallback）均可路由。
export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Overview />} />
          <Route path="/arena" element={<Arena />} />
          <Route path="/live" element={<Live />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/reproduce" element={<Reproduce />} />
          <Route path="*" element={<Overview />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
