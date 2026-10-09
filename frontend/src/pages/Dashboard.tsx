import { Link } from 'react-router-dom'
import { AlertOctagon, Gauge, MapPinned, ScanSearch } from 'lucide-react'
import { useApi } from '../lib/api'
import { CLASS_CODES, CLASS_COLORS, CLASS_NAMES, dateTime, intFmt, pct } from '../lib/format'
import { deviceLabel } from '../lib/format'
import type { ExperimentSummary, Stats, StoredDetection } from '../lib/types'
import { Button, Card, ClassBadge, EmptyState, ErrorState, PageHeader, SeverityBadge, SourceBadge, Spinner, StatCard } from '../components/ui'

export default function Dashboard() {
  const stats = useApi<Stats>('/stats')
  const recent = useApi<{ items: StoredDetection[] }>('/detections', { limit: 8 })
  const model = useApi<{ loaded: boolean; info: any }>('/model')
  const exps = useApi<ExperimentSummary[]>('/training/experiments')

  if (stats.error) return <ErrorState message={`Cannot reach the API: ${stats.error}`} onRetry={stats.reload} />
  const s = stats.data
  const best = exps.data?.find((e) => e.name === model.data?.info?.run_name)

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Overview of every detection stored in the database. All numbers come from real model inference."
        actions={<Link to="/detect"><Button><ScanSearch className="h-4 w-4" /> Analyse a road image</Button></Link>} />
      {!s ? <Spinner /> : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard label="Total detections" value={intFmt(s.total_detections)} hint={`${intFmt(s.total_inferences)} analysed images/videos`} icon={<ScanSearch className="h-4 w-4" />} />
            <StatCard label="Average confidence" value={pct(s.average_confidence)} icon={<Gauge className="h-4 w-4" />} />
            <StatCard label="High severity" value={intFmt(s.severity.HIGH)} hint="heuristic score ≥ 65" icon={<AlertOctagon className="h-4 w-4" />} accent="#d03b3b" />
            <StatCard label="Geolocated" value={intFmt(s.geolocated_detections)} hint="shown on the map" icon={<MapPinned className="h-4 w-4" />} />
          </div>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {CLASS_CODES.map((c) => (
              <StatCard key={c} label={`${c} · ${CLASS_NAMES[c]}`} value={intFmt(s.per_class[c].count)} accent={CLASS_COLORS[c]}
                hint={s.per_class[c].avg_confidence !== null ? `avg conf ${pct(s.per_class[c].avg_confidence)}` : 'no detections yet'} />
            ))}
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card title="Recent detections" className="lg:col-span-2" bodyClass="p-0"
              actions={<Link to="/history" className="text-xs font-medium text-slate-600 hover:text-slate-900">View all →</Link>}>
              {recent.loading ? <Spinner /> : !recent.data?.items.length ? (
                <div className="p-5"><EmptyState title="No detections yet">Upload a road image on the Image Detection page to create the first record.</EmptyState></div>
              ) : (
                <ul className="divide-y divide-slate-100">
                  {recent.data.items.map((d) => (
                    <li key={d.id} className="flex items-center gap-3 px-5 py-2.5">
                      {d.crop_url ? <img src={d.crop_url} className="h-10 w-10 rounded object-cover" alt="" /> : <div className="h-10 w-10 rounded bg-slate-100" />}
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2"><ClassBadge code={d.class_code} /><SeverityBadge level={d.severity} /></div>
                        <div className="mt-0.5 text-xs text-slate-500">{dateTime(d.created_at)} · conf {pct(d.confidence)}</div>
                      </div>
                      <SourceBadge source={d.location_source} />
                    </li>
                  ))}
                </ul>
              )}
            </Card>
            <Card title="Active model">
              {model.loading ? <Spinner /> : !model.data?.loaded ? (
                <EmptyState title="No trained model loaded">Run the training pipeline; the best checkpoint is picked up automatically.</EmptyState>
              ) : (
                <dl className="space-y-2 text-sm">
                  <Row k="Checkpoint" v={model.data.info.weights} />
                  <Row k="Architecture" v={model.data.info.architecture ?? '—'} />
                  <Row k="Parameters" v={intFmt(model.data.info.parameters)} />
                  <Row k="Device" v={deviceLabel(model.data.info.device)} />
                  {best?.test_metrics && <>
                    <Row k="Test mAP@50" v={pct(best.test_metrics.mAP50)} />
                    <Row k="Test mAP@50-95" v={pct(best.test_metrics.mAP50_95)} />
                  </>}
                  <Link to="/model" className="block pt-2 text-xs font-medium text-slate-600 hover:text-slate-900">Model details →</Link>
                </dl>
              )}
            </Card>
          </div>
        </div>
      )}
    </>
  )
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="flex justify-between gap-4"><dt className="text-slate-500">{k}</dt><dd className="truncate text-right font-medium text-slate-800">{v}</dd></div>
}
