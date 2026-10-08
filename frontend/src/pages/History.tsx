import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, MapPinned, Trash2 } from 'lucide-react'
import { apiSend, useApi } from '../lib/api'
import { CLASS_CODES, CLASS_NAMES, SEVERITIES, coord, dateTime, pct } from '../lib/format'
import type { StoredDetection } from '../lib/types'
import { Button, Card, ClassBadge, EmptyState, ErrorState, PageHeader, SeverityBadge, SourceBadge, Spinner, inputCls } from '../components/ui'

const PAGE = 25

export default function History() {
  const [cls, setCls] = useState('')
  const [sev, setSev] = useState('')
  const [kind, setKind] = useState('')
  const [minConf, setMinConf] = useState('')
  const [sort, setSort] = useState('created_at')
  const [page, setPage] = useState(0)
  const params = { classes: cls, severity: sev, kind, min_conf: minConf, sort, order: 'desc', limit: PAGE, offset: page * PAGE }
  const { data, error, loading, reload } = useApi<{ total: number; items: StoredDetection[] }>('/detections', params)

  const remove = async (id: number) => {
    if (!confirm(`Delete detection #${id}?`)) return
    await apiSend(`/detections/${id}`, 'DELETE')
    reload()
  }

  const exportCsv = async () => {
    const res = await fetch(`/api/detections?${new URLSearchParams({ ...Object.fromEntries(Object.entries(params).filter(([, v]) => v !== '').map(([k, v]) => [k, String(v)])), limit: '1000', offset: '0' })}`)
    const rows: StoredDetection[] = (await res.json()).items
    const head = ['id', 'created_at', 'class_code', 'class_name', 'confidence', 'severity', 'severity_score', 'latitude', 'longitude', 'location_source', 'kind', 'video_time_s', 'model_version']
    const csv = [head.join(','), ...rows.map((r) => head.map((h) => JSON.stringify((r as any)[h] ?? '')).join(','))].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = 'rdd_yolo_detections.csv'
    a.click()
  }

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1
  const reset = (fn: (v: string) => void) => (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => { fn(e.target.value); setPage(0) }

  return (
    <>
      <PageHeader title="Detection History" subtitle="Every stored detection, newest first."
        actions={<Button variant="secondary" onClick={exportCsv} disabled={!data?.total}><Download className="h-4 w-4" /> Export CSV</Button>} />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-5">
        <select className={inputCls} value={cls} onChange={reset(setCls)}><option value="">All classes</option>{CLASS_CODES.map((c) => <option key={c} value={c}>{c} {CLASS_NAMES[c]}</option>)}</select>
        <select className={inputCls} value={sev} onChange={reset(setSev)}><option value="">All severities</option>{SEVERITIES.map((s) => <option key={s}>{s}</option>)}</select>
        <select className={inputCls} value={kind} onChange={reset(setKind)}><option value="">All sources</option><option value="image">Image</option><option value="video">Video</option><option value="webcam">Webcam</option></select>
        <input className={inputCls} type="number" min={0} max={1} step={0.05} placeholder="Min confidence (0-1)" value={minConf} onChange={reset(setMinConf)} />
        <select className={inputCls} value={sort} onChange={reset(setSort)}><option value="created_at">Sort: newest</option><option value="confidence">Sort: confidence</option><option value="severity_score">Sort: severity</option></select>
      </div>
      {error && <ErrorState message={error} onRetry={reload} />}
      <Card bodyClass="p-0">
        {loading && !data ? <Spinner /> : !data?.items.length ? <div className="p-5"><EmptyState title="No detections match">Analyse an image or video to populate the history.</EmptyState></div> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-slate-500">
                <tr>{['', 'Timestamp', 'Class', 'Confidence', 'Severity', 'Coordinates', 'Location source', 'Input', 'Model', ''].map((h, i) => <th key={i} className="whitespace-nowrap px-3 py-2 font-medium first:pl-5">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.items.map((d) => (
                  <tr key={d.id} className="hover:bg-slate-50">
                    <td className="py-1.5 pl-5 pr-2">{d.crop_url ? <a href={d.image_url ?? d.crop_url} target="_blank" rel="noreferrer"><img src={d.crop_url} alt="" className="h-10 w-10 rounded object-cover" /></a> : null}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-slate-600">{dateTime(d.created_at)}</td>
                    <td className="px-3 py-2"><ClassBadge code={d.class_code} /></td>
                    <td className="px-3 py-2 tabular-nums">{pct(d.confidence)}</td>
                    <td className="px-3 py-2"><SeverityBadge level={d.severity} score={d.severity_score} /></td>
                    <td className="whitespace-nowrap px-3 py-2 font-mono text-xs">{coord(d.latitude, d.longitude)}</td>
                    <td className="px-3 py-2"><SourceBadge source={d.location_source} /></td>
                    <td className="px-3 py-2 text-xs text-slate-600">{d.kind}{d.video_time_s !== null && ` @ ${d.video_time_s.toFixed(1)}s`}</td>
                    <td className="max-w-[160px] truncate px-3 py-2 text-xs text-slate-500" title={d.model_version}>{d.model_version}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-right">
                      {d.latitude !== null && <Link to={`/map?focus=${d.inference_id}`} title="Show on map" className="mr-2 inline-flex text-slate-500 hover:text-slate-900"><MapPinned className="h-4 w-4" /></Link>}
                      <button onClick={() => remove(d.id)} title="Delete" className="text-slate-400 hover:text-red-600"><Trash2 className="h-4 w-4" /></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data && data.total > 0 && (
          <div className="flex items-center justify-between border-t border-slate-100 px-5 py-3 text-sm text-slate-600">
            <span>{data.total} detection(s)</span>
            <div className="flex items-center gap-2">
              <Button variant="secondary" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
              <span>Page {page + 1} / {pages}</span>
              <Button variant="secondary" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</Button>
            </div>
          </div>
        )}
      </Card>
    </>
  )
}
