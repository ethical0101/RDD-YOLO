import type { ReactNode } from 'react'
import { AlertTriangle, Inbox, Loader2 } from 'lucide-react'
import { CLASS_COLORS, CLASS_NAMES, SEVERITY_COLORS, SOURCE_LABELS } from '../lib/format'
import type { ClassCode, LocationSource, SeverityLevel } from '../lib/types'

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(' ')
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-slate-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Card({ title, subtitle, actions, children, className, bodyClass }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string
}) {
  return (
    <section className={cx('rounded-xl border border-slate-200 bg-white shadow-sm', className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-800">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={cx('p-5', bodyClass)}>{children}</div>
    </section>
  )
}

export function StatCard({ label, value, hint, accent, icon }: {
  label: string; value: ReactNode; hint?: ReactNode; accent?: string; icon?: ReactNode
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        {accent && <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: accent }} />}
        <span className="text-2xl font-semibold tabular-nums text-slate-900">{value}</span>
      </div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  )
}

export function ClassBadge({ code, showName = true }: { code: ClassCode; showName?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-slate-200">
      <span className="h-2 w-2 rounded-full" style={{ background: CLASS_COLORS[code] }} />
      {code}{showName && <span className="text-slate-500">· {CLASS_NAMES[code]}</span>}
    </span>
  )
}

export function SeverityBadge({ level, score }: { level: SeverityLevel; score?: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-semibold text-slate-800 ring-1 ring-slate-200">
      <span className="h-2 w-2 rounded-sm" style={{ background: SEVERITY_COLORS[level] }} />
      {level}{score !== undefined && <span className="font-normal text-slate-500">{score.toFixed(0)}</span>}
    </span>
  )
}

export function SourceBadge({ source }: { source: LocationSource | null }) {
  if (!source) return <span className="text-xs text-slate-400">No location</span>
  return (
    <span className="inline-flex rounded-md bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-800 ring-1 ring-sky-200">
      {SOURCE_LABELS[source]}
    </span>
  )
}

export function Button({ children, variant = 'primary', className, ...p }: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
}) {
  const styles = {
    primary: 'bg-slate-900 text-white hover:bg-slate-800 disabled:bg-slate-300',
    secondary: 'bg-white text-slate-800 ring-1 ring-slate-300 hover:bg-slate-50 disabled:text-slate-400',
    ghost: 'text-slate-600 hover:bg-slate-100',
    danger: 'bg-white text-red-700 ring-1 ring-red-200 hover:bg-red-50',
  }[variant]
  return (
    <button {...p} className={cx('inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed', styles, className)}>
      {children}
    </button>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1">{message}</div>
      {onRetry && <button onClick={onRetry} className="font-medium underline">Retry</button>}
    </div>
  )
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 px-6 py-12 text-center">
      <div className="text-slate-400">{icon ?? <Inbox className="h-8 w-8" />}</div>
      <div className="mt-3 text-sm font-medium text-slate-700">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-slate-500">{children}</div>}
    </div>
  )
}

export function Notice({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warn' }) {
  return (
    <div className={cx('rounded-lg border px-4 py-3 text-sm', tone === 'warn'
      ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-sky-200 bg-sky-50 text-sky-900')}>
      {children}
    </div>
  )
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  )
}

export const inputCls = 'w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200'
