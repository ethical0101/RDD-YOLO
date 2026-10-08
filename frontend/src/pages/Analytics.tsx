import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../lib/api'
import { CLASS_CODES, CLASS_COLORS, CLASS_NAMES, SEVERITIES, SEVERITY_COLORS, intFmt, pct } from '../lib/format'
import type { Stats } from '../lib/types'
import { Card, EmptyState, ErrorState, PageHeader, Spinner } from '../components/ui'

const axis = { stroke: '#c3c2b7', tick: { fill: '#898781', fontSize: 11 }, tickLine: false }
const grid = <CartesianGrid stroke="#e1e0d9" strokeDasharray="0" vertical={false} />
const tooltipStyle = { contentStyle: { borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 }, cursor: { fill: 'rgba(15,23,42,0.04)' } }

export default function Analytics() {
  const [days, setDays] = useState(30)
  const { data: s, error, reload } = useApi<Stats>('/stats', { days })
  if (error) return <ErrorState message={error} onRetry={reload} />
  if (!s) return <Spinner />

  const dist = CLASS_CODES.map((c) => ({ code: c, name: `${c} ${CLASS_NAMES[c]}`, count: s.per_class[c].count, fill: CLASS_COLORS[c] }))
  const sevByClass = CLASS_CODES.map((c) => ({ code: c, ...s.per_class[c].severity }))
  const timeline = s.over_time.map((d) => ({ ...d, label: d.date.slice(5) }))
  const none = s.total_detections === 0

  return (
    <>
      <PageHeader title="Analytics" subtitle="Aggregates computed from the stored detections in the database."
        actions={<select value={days} onChange={(e) => setDays(Number(e.target.value))} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm">
          <option value={7}>Last 7 days</option><option value={30}>Last 30 days</option><option value={90}>Last 90 days</option></select>} />
      {none ? <EmptyState title="No detections stored yet">Analytics appear once images or videos have been analysed and saved.</EmptyState> : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="Damage type distribution" subtitle={`${intFmt(s.total_detections)} detections`}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={dist} margin={{ top: 16, right: 8, left: -12, bottom: 0 }} barCategoryGap="28%">
                {grid}
                <XAxis dataKey="code" {...axis} />
                <YAxis allowDecimals={false} {...axis} axisLine={false} />
                <Tooltip {...tooltipStyle} formatter={(v) => [v, 'Detections']} labelFormatter={(c) => `${c} · ${CLASS_NAMES[c as keyof typeof CLASS_NAMES]}`} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} label={{ position: 'top', fill: '#52514e', fontSize: 11 }} isAnimationActive={false}>
                  {dist.map((d) => <Cell key={d.code} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <ClassLegend />
          </Card>

          <Card title="Confidence distribution" subtitle={`Mean confidence ${pct(s.average_confidence)}`}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={s.confidence_histogram} margin={{ top: 16, right: 8, left: -12, bottom: 0 }} barCategoryGap={2}>
                {grid}
                <XAxis dataKey="bin" {...axis} interval={0} fontSize={10} />
                <YAxis allowDecimals={false} {...axis} axisLine={false} />
                <Tooltip {...tooltipStyle} formatter={(v) => [v, 'Detections']} labelFormatter={(b) => `Confidence ${b}`} />
                <Bar dataKey="count" fill="#2a78d6" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Severity by damage type" subtitle="Heuristic levels (see Model page for the formula)">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={sevByClass} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barCategoryGap="28%">
                {grid}
                <XAxis dataKey="code" {...axis} />
                <YAxis allowDecimals={false} {...axis} axisLine={false} />
                <Tooltip {...tooltipStyle} />
                <Legend iconType="square" wrapperStyle={{ fontSize: 12 }} />
                {SEVERITIES.map((lv) => (
                  <Bar key={lv} dataKey={lv} stackId="s" fill={SEVERITY_COLORS[lv]} stroke="#fff" strokeWidth={2} isAnimationActive={false} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Detections over time" subtitle={`Per day, last ${days} days`}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={timeline} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                {grid}
                <XAxis dataKey="label" {...axis} minTickGap={24} />
                <YAxis allowDecimals={false} {...axis} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle.contentStyle} cursor={{ stroke: '#94a3b8', strokeWidth: 1 }} />
                <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
                {CLASS_CODES.map((c) => (
                  <Line key={c} type="monotone" dataKey={c} name={`${c} ${CLASS_NAMES[c]}`} stroke={CLASS_COLORS[c]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Class-wise statistics" className="lg:col-span-2" bodyClass="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs text-slate-500">
                  <tr><th className="px-5 py-2 font-medium">Class</th><th className="px-3 py-2 text-right font-medium">Detections</th><th className="px-3 py-2 text-right font-medium">Share</th>
                    <th className="px-3 py-2 text-right font-medium">Avg confidence</th>{SEVERITIES.map((l) => <th key={l} className="px-3 py-2 text-right font-medium">{l}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {CLASS_CODES.map((c) => {
                    const p = s.per_class[c]
                    return (
                      <tr key={c}>
                        <td className="px-5 py-2"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: CLASS_COLORS[c] }} />{c} · {CLASS_NAMES[c]}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{intFmt(p.count)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{pct(p.count / s.total_detections)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{pct(p.avg_confidence)}</td>
                        {SEVERITIES.map((l) => <td key={l} className="px-3 py-2 text-right tabular-nums">{p.severity[l]}</td>)}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}

function ClassLegend() {
  return (
    <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-slate-600">
      {CLASS_CODES.map((c) => <span key={c} className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: CLASS_COLORS[c] }} />{c} {CLASS_NAMES[c]}</span>)}
    </div>
  )
}
