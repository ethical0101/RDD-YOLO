import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev server proxies API and file requests to the FastAPI backend.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', 'VITE_')
  const target = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'
  return {
    plugins: [react(), tailwindcss()],
    envDir: '..',
    server: {
      port: 5173,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/files': { target, changeOrigin: true },
      },
    },
    build: { chunkSizeWarningLimit: 1500 },
  }
})
