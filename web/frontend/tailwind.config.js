/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // PLAN §9.4 暗色专业风：近黑深蓝灰面板，避免重靛蓝/紫。
        bg: {
          base: '#0b0e14',
          panel: '#141a26',
          raised: '#1a2231',
        },
        line: '#1f2735',
        ink: {
          DEFAULT: '#e6edf6',
          dim: '#9aa7b8',
          faint: '#64748b',
        },
        accent: {
          DEFAULT: '#38bdf8',
          soft: '#22d3ee',
        },
        ok: '#34d399',
        bad: '#f87171',
        warn: '#fbbf24',
      },
      fontFamily: {
        // 系统字体栈，无外网字体外链。
        sans: [
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Microsoft YaHei',
          'PingFang SC',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'Cascadia Code',
          'Consolas',
          'Menlo',
          'Liberation Mono',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
}
