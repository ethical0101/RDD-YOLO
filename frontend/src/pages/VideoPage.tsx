import { useEffect, useRef, useState } from 'react'
import { Film, Route as RouteIcon, Upload } from 'lucide-react'
import { apiSend, useApi } from '../lib/api'
import { CLASS_CODES, CLASS_COLORS, dateTime, pct } from '../lib/format'
import type { Inference } from '../lib/types'
import LocationPicker, { NO_LOCATION, type PickedLocation } from '../components/LocationPicker'
import { Button, Card, ClassBadge, EmptyState, ErrorState, Field, PageHeader, SeverityBadge, SourceBadge, Spinner, inputCls } from '../components/ui'

export default function VideoPage() {
  const [file, setFile] = useState<File | null>(null)
  const [route, setRoute] = useState<File | null>(null)
  const [conf, setConf] = useState(0.25)
  const [stride, setStride] = useState(3)
  const [locMode, setLocMode] = useState<'none' | 'route' | 'fixed'>('none')
  const [loc, setLoc] = useState<PickedLocation>(NO_LOCATION)
  const [jobId, setJobId] = useState<number | null>(null)
  const [job, setJob] = useState<Inference | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const history = useApi<Inference[]>('/videos', { limit: 10 }, [job?.status])

  useEffect(() => {
    if (!jobId) return
    let stop = false
    const tick = async () => {
      try {
        const j = await (await fetch(`/api/inference/video/${jobId}`)).json()
        if (stop) return
        setJob(j)
        if (j.status === 'queued' || j.status === 'processing') setTimeout(tick, 1000)
      } catch (e: any) { setError(e.message) }
    }
    tick()
    return () => { stop = true }
  }, [jobId])

  const submit = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setJob(null)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('conf', String(conf))
    fd.append('stride', String(stride))
    if (locMode === 'route' && route) fd.append('route_file', route)
    if (locMode === 'fixed' && loc.lat !== null && loc.lon !== null) {
      fd.append('latitude', String(loc.lat))
      fd.append('longitude', String(loc.lon))
      fd.append('location_source', loc.source ?? 'manual')
    }
    try {
      const r = await apiSend<{ job_id: number }>('/inference/video', 'POST', fd)
      setJobId(r.job_id)
    } catch (e: any) { setError(e.message) } finally { setBusy(false) }
  }

  const seek = (t: number | null) => { if (videoRef.current && t !== null) { videoRef.current.currentTime = t; videoRef.current.play() } }
  const running = job && (job.status === 'queued' || job.status === 'processing')
  const duration = job?.extra?.duration_s as number | undefined

  return (
    <>
      <PageHeader title="Video Detection" subtitle="Process a dash-cam / phone video frame by frame. Output: annotated H.264 video, timestamped detections, optional GPS route mapping." />
      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <Card title="Video & settings">
            <div className="space-y-4">
              <Field label="Video file" hint="MP4, MOV, AVI, MKV, WebM">
                <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v" className={inputCls} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              </Field>
              <Field label={`Confidence threshold: ${conf.toFixed(2)}`}>
                <input type="range" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} className="w-full accent-slate-900" />
              </Field>
              <Field label={`Frame stride: every ${stride} frame(s)`} hint="Higher = faster processing; boxes are carried between analysed frames.">
                <input type="range" min={1} max={15} value={stride} onChange={(e) => setStride(Number(e.target.value))} className="w-full accent-slate-900" />
              </Field>
            </div>
          </Card>
          <Card title="Location (optional)">
            <div className="mb-3 flex gap-1 rounded-lg bg-slate-100 p-1 text-sm">
              {(['none', 'route', 'fixed'] as const).map((m) => (
                <button key={m} onClick={() => setLocMode(m)} className={`flex-1 rounded-md px-2 py-1.5 ${locMode === m ? 'bg-white font-medium shadow-sm' : 'text-slate-600'}`}>
                  {m === 'none' ? 'None' : m === 'route' ? 'GPS route' : 'Single point'}
                </button>
              ))}
            </div>
            {locMode === 'route' && (
              <Field label="Route file (CSV or GPX)" hint={<>CSV columns <code>time_s,lat,lon</code> (seconds from video start) or GPX track with timestamps. Positions are interpolated per detection; frames outside the track get no location.</>}>
                <input type="file" accept=".csv,.gpx" className={inputCls} onChange={(e) => setRoute(e.target.files?.[0] ?? null)} />
              </Field>
            )}
            {locMode === 'fixed' && <LocationPicker value={loc} onChange={setLoc} compact />}
            {locMode === 'none' && <p className="text-xs text-slate-500">Detections will be stored without coordinates (they won't appear on the map).</p>}
          </Card>
          <Button className="w-full py-2.5" onClick={submit} disabled={!file || busy || !!running || (locMode === 'route' && !route)}>
            <Upload className="h-4 w-4" /> {busy ? 'Uploading…' : running ? 'Processing…' : 'Process video'}
          </Button>
          {error && <ErrorState message={error} />}
        </div>

        <div className="space-y-6">
          <Card title="Output" subtitle={job ? `Job #${job.id} · ${job.status}` : undefined}>
            {!job ? (
              <EmptyState title="No video processed yet" icon={<Film className="h-8 w-8" />}>Upload a video to run frame-by-frame detection.</EmptyState>
            ) : running ? (
              <div className="py-8">
                <div className="mb-2 flex justify-between text-sm text-slate-600"><span>Processing frames on the {job.status === 'queued' ? 'queue' : 'model'}…</span><span>{pct(job.progress, 0)}</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full bg-slate-900 transition-all" style={{ width: `${job.progress * 100}%` }} /></div>
              </div>
            ) : job.status === 'failed' ? <ErrorState message={job.error ?? 'Processing failed'} /> : (
              <div className="space-y-4">
                <video ref={videoRef} src={job.output_video_url ?? undefined} controls className="w-full rounded-lg bg-black" />
                <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                  <Mini k="Analysed frames" v={`${job.extra?.processed_frames} / ${job.extra?.frames}`} />
                  <Mini k="Unique detections" v={job.num_detections} />
                  <Mini k="Mean inference" v={`${job.extra?.mean_inference_ms ?? '—'} ms`} />
                  <Mini k="Inference FPS" v={job.extra?.inference_fps ?? '—'} />
                </div>
                {duration && job.detections && job.detections.length > 0 && (
                  <div>
                    <div className="mb-1 text-xs font-medium text-slate-500">Detection timeline (click to seek)</div>
                    <div className="relative h-8 rounded bg-slate-100">
                      {job.detections.map((d) => (
                        <button key={d.id} title={`${d.class_code} @ ${d.video_time_s?.toFixed(1)}s`} onClick={() => seek(d.video_time_s)}
                          className="absolute top-1 h-6 w-1.5 -translate-x-1/2 rounded-sm"
                          style={{ left: `${((d.video_time_s ?? 0) / duration) * 100}%`, background: CLASS_COLORS[d.class_code] }} />
                      ))}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-3">{CLASS_CODES.map((c) => <ClassBadge key={c} code={c} />)}</div>
                  </div>
                )}
              </div>
            )}
          </Card>
          {job?.status === 'completed' && (
            <Card title="Timestamped detections" subtitle="Repeated sightings of the same defect across consecutive frames are merged (best view kept)." bodyClass="p-0">
              {!job.detections?.length ? <div className="p-5"><EmptyState title="No damage detected in this video" /></div> : (
                <div className="max-h-96 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-slate-50 text-left text-xs text-slate-500">
                      <tr><th className="px-5 py-2 font-medium">Time</th><th className="px-3 py-2 font-medium">View</th><th className="px-3 py-2 font-medium">Class</th>
                        <th className="px-3 py-2 font-medium">Conf.</th><th className="px-3 py-2 font-medium">Severity</th><th className="px-3 py-2 font-medium">Location</th></tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {job.detections.map((d) => (
                        <tr key={d.id} className="cursor-pointer hover:bg-slate-50" onClick={() => seek(d.video_time_s)}>
                          <td className="px-5 py-2 font-mono text-xs">{d.video_time_s?.toFixed(2)} s</td>
                          <td className="px-3 py-1.5">{d.crop_url && <img src={d.crop_url} className="h-9 w-9 rounded object-cover" alt="" />}</td>
                          <td className="px-3 py-2"><ClassBadge code={d.class_code} showName={false} /></td>
                          <td className="px-3 py-2 tabular-nums">{pct(d.confidence)}</td>
                          <td className="px-3 py-2"><SeverityBadge level={d.severity} /></td>
                          <td className="px-3 py-2"><SourceBadge source={d.location_source} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}
          <Card title="Previous videos" bodyClass="p-0">
            {history.loading ? <Spinner /> : !history.data?.length ? <div className="p-5 text-sm text-slate-500">None yet.</div> : (
              <ul className="divide-y divide-slate-100 text-sm">
                {history.data.map((v) => (
                  <li key={v.id} className="flex items-center justify-between px-5 py-2.5">
                    <div><div className="font-medium text-slate-800">{v.source_filename}</div><div className="text-xs text-slate-500">{dateTime(v.created_at)} · {v.num_detections} detections · {v.status}</div></div>
                    <div className="flex items-center gap-2">{v.location_source === 'route' && <RouteIcon className="h-4 w-4 text-slate-400" />}
                      <Button variant="ghost" onClick={() => setJobId(v.id)}>Open</Button></div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </>
  )
}

function Mini({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-xs text-slate-500">{k}</div><div className="font-semibold tabular-nums">{v}</div></div>
}
