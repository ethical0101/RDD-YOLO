import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  Activity, BarChart3, Brain, Camera, FlaskConical, History, LayoutDashboard, Map, Menu, ScanSearch, Video, X,
} from 'lucide-react'
import { apiGet } from '../lib/api'
import { STATIC } from '../lib/staticMode'
import type { Health } from '../lib/types'
import { cx } from './ui'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/detect', label: 'Image Detection', icon: ScanSearch },
  { to: '/video', label: 'Video Detection', icon: Video },
  { to: '/live', label: 'Live Camera', icon: Camera },
  { to: '/map', label: 'Damage Map', icon: Map },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/history', label: 'Detection History', icon: History },
  { to: '/model', label: 'Model', icon: Brain },
  { to: '/training', label: 'Training & Experiments', icon: FlaskConical },
]

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null)
  const [down, setDown] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const poll = () => apiGet<Health>('/health').then((h) => { setHealth(h); setDown(false) }).catch(() => setDown(true))
    poll()
    const t = setInterval(poll, 15000)
    return () => clearInterval(t)
  }, [])

  const status = down
    ? { dot: 'bg-red-500', text: 'Backend offline' }
    : !health ? { dot: 'bg-slate-400', text: 'Connecting…' }
    : health.model_loaded ? { dot: 'bg-emerald-500', text: health.model_version }
    : { dot: 'bg-amber-500', text: 'No model loaded' }

  const sidebar = (
    <nav className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-orange-500 text-white">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <div className="text-sm font-semibold text-white">RDD-YOLO</div>
          <div className="text-[11px] text-slate-400">Road Damage Detection</div>
        </div>
      </div>
      <div className="flex-1 space-y-0.5 px-3">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === '/'} onClick={() => setOpen(false)}
            className={({ isActive }) => cx('flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
              isActive ? 'bg-white/10 font-medium text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white')}>
            <Icon className="h-4 w-4" /> {label}
          </NavLink>
        ))}
      </div>
      <a href="https://github.com/ethical0101/RDD-YOLO" target="_blank" rel="noreferrer"
        className="mx-3 mb-1 flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-white">
        <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
        Source code on GitHub
      </a>
      <div className="m-3 rounded-lg bg-white/5 p-3 text-xs text-slate-300">
        <div className="flex items-center gap-2">
          <span className={cx('h-2 w-2 rounded-full', status.dot)} />
          <span className="truncate font-medium text-white" title={status.text}>{status.text}</span>
        </div>
        {health && (
          <div className="mt-1.5 text-slate-400">
            {health.cuda_available ? 'CUDA GPU' : 'CPU'} · DB {health.database} · v{health.version}
          </div>
        )}
      </div>
    </nav>
  )

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 bg-slate-900 lg:block">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-[2000] flex lg:hidden">
          <div className="w-64 bg-slate-900">{sidebar}</div>
          <div className="flex-1 bg-black/40" onClick={() => setOpen(false)} />
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
          <button onClick={() => setOpen(!open)} aria-label="Menu">{open ? <X /> : <Menu />}</button>
          <span className="font-semibold">RDD-YOLO</span>
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            {STATIC && (
              <div className="mb-5 rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
                <b>Live browser demo.</b> The trained YOLO26s model runs entirely in your browser (WebGPU / WebAssembly) — no
                server; your images and detections stay in this browser. First use downloads the 38 MB model. For GPU speed and the
                full Python backend, run the project locally — <a className="font-medium underline" href="https://github.com/ethical0101/RDD-YOLO" target="_blank" rel="noreferrer">source on GitHub</a>.
              </div>
            )}
            {health && !health.model_loaded && (
              <div className="mb-5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                <b>Inference unavailable:</b> {health.model_error}
              </div>
            )}
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
