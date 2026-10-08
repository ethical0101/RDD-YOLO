import { useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Save } from 'lucide-react'
import { apiSend } from '../lib/api'
import { CLASS_COLORS, pct } from '../lib/format'
import { useBrowserLocation } from '../lib/geolocation'
import type { DetectionResult } from '../lib/types'
import { Button, Card, ClassBadge, EmptyState, ErrorState, Field, Notice, PageHeader, SeverityBadge } from '../components/ui'

/** Webcam detection: frames are captured in the browser and sent to the API (~1 request in flight at a time). */
export default function Live() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const runningRef = useRef(false)
  const [running, setRunning] = useState(false)
  const [dets, setDets] = useState<DetectionResult[]>([])
  const [stats, setStats] = useState({ fps: 0, ms: 0 })
  const [conf, setConf] = useState(0.3)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)
  const geo = useBrowserLocation()
  const confRef = useRef(conf)
  useEffect(() => { confRef.current = conf }, [conf])

  const grab = (): Promise<Blob | null> => {
    const v = videoRef.current!
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    return new Promise((r) => c.toBlob(r, 'image/jpeg', 0.85))
  }

  const draw = (d: DetectionResult[], w: number, h: number) => {
    const c = canvasRef.current
    if (!c) return
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)
    ctx.lineWidth = Math.max(2, w / 300)
    ctx.font = `600 ${Math.max(14, w / 45)}px Inter, sans-serif`
    for (const x of d) {
      const [x1, y1, x2, y2] = x.bbox
      ctx.strokeStyle = ctx.fillStyle = CLASS_COLORS[x.class_code]
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const label = `${x.class_code} ${(x.confidence * 100).toFixed(0)}%`
      const tw = ctx.measureText(label).width
      ctx.fillRect(x1, Math.max(0, y1 - 22), tw + 8, 22)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x1 + 4, Math.max(16, y1 - 6))
    }
  }

  const loop = async () => {
    let frames = 0
    let t0 = performance.now()
    while (runningRef.current) {
      const v = videoRef.current
      if (!v || v.readyState < 2) { await new Promise((r) => setTimeout(r, 100)); continue }
      const blob = await grab()
      if (!blob) continue
      const fd = new FormData()
      fd.append('file', blob, 'frame.jpg')
      fd.append('conf', String(confRef.current))
      try {
        const r = await apiSend<{ detections: DetectionResult[]; inference_ms: number; width: number; height: number }>('/inference/frame', 'POST', fd)
        if (!runningRef.current) break
        setDets(r.detections)
        draw(r.detections, r.width, r.height)
        frames++
        const dt = (performance.now() - t0) / 1000
        if (dt > 1) { setStats({ fps: frames / dt, ms: r.inference_ms }); frames = 0; t0 = performance.now() }
      } catch (e: any) { setError(e.message); stop(); break }
    }
  }

  const start = async () => {
    setError(null)
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false })
      streamRef.current = s
      videoRef.current!.srcObject = s
      await videoRef.current!.play()
      runningRef.current = true
      setRunning(true)
      loop()
    } catch (e: any) {
      setError(e.name === 'NotAllowedError' ? 'Camera permission denied.' : `Camera unavailable: ${e.message}`)
    }
  }

  const stop = () => {
    runningRef.current = false
    setRunning(false)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    canvasRef.current?.getContext('2d')?.clearRect(0, 0, 9999, 9999)
  }
  useEffect(() => () => stop(), [])

  const snapshot = async () => {
    const blob = await grab()
    if (!blob) return
    const fix = geo.fix ?? await geo.request()
    const fd = new FormData()
    fd.append('file', blob, 'webcam.jpg')
    fd.append('conf', String(conf))
    fd.append('save', 'true')
    if (fix) { fd.append('latitude', String(fix.lat)); fd.append('longitude', String(fix.lon)); fd.append('location_accuracy_m', String(fix.accuracy)) }
    try {
      const r = await apiSend<{ inference_id: number; detections: unknown[] }>('/inference/frame', 'POST', fd)
      setSaved(`Saved snapshot #${r.inference_id} with ${r.detections.length} detection(s)${fix ? ' and browser GPS' : ' (no location)'}.`)
    } catch (e: any) { setError(e.message) }
  }

  return (
    <>
      <PageHeader title="Live Camera" subtitle="Real-time detection from a webcam or phone camera (frames are analysed by the backend model)."
        actions={running
          ? <><Button variant="secondary" onClick={snapshot}><Save className="h-4 w-4" /> Save snapshot</Button><Button variant="danger" onClick={stop}><CameraOff className="h-4 w-4" /> Stop</Button></>
          : <Button onClick={start}><Camera className="h-4 w-4" /> Start camera</Button>} />
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <Card bodyClass="p-3">
          <div className="relative overflow-hidden rounded-lg bg-slate-900">
            <video ref={videoRef} muted playsInline className={`block w-full ${running ? '' : 'hidden'}`} />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            {!running && <div className="p-8"><EmptyState title="Camera is off" icon={<Camera className="h-8 w-8" />}>Start the camera and allow access. Point it at the road surface.</EmptyState></div>}
          </div>
        </Card>
        <div className="space-y-4">
          <Card title="Live stats">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-xs text-slate-500">Throughput</div><div className="font-semibold tabular-nums">{stats.fps.toFixed(1)} fps</div></div>
              <div className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-xs text-slate-500">Inference</div><div className="font-semibold tabular-nums">{stats.ms.toFixed(0)} ms</div></div>
            </div>
            <div className="mt-4"><Field label={`Confidence: ${conf.toFixed(2)}`}><input type="range" min={0.1} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} className="w-full accent-slate-900" /></Field></div>
          </Card>
          <Card title="Current frame" bodyClass="p-0">
            {!dets.length ? <div className="p-4 text-sm text-slate-500">No detections.</div> : (
              <ul className="divide-y divide-slate-100">{dets.map((d, i) => (
                <li key={i} className="flex items-center justify-between px-4 py-2"><ClassBadge code={d.class_code} showName={false} /><span className="text-sm tabular-nums">{pct(d.confidence)}</span><SeverityBadge level={d.severity} /></li>
              ))}</ul>
            )}
          </Card>
          {saved && <Notice>{saved}</Notice>}
          {error && <ErrorState message={error} />}
          <p className="text-xs text-slate-500">Camera access needs <code>localhost</code> or HTTPS. Frames are not stored unless you save a snapshot.</p>
        </div>
      </div>
    </>
  )
}
