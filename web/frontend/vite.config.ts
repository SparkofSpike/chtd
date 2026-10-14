import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 离线打包：base 相对路径，产物不引用任何外网 CDN/字体。
// 开发时 /api 代理到后端 FastAPI（默认 127.0.0.1:8088）。
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8088',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    assetsInlineLimit: 0,
  },
})
