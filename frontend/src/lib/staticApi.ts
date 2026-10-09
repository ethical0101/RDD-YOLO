/**
 * Browser-only implementation of the REST API for the static deployment.
 * Same routes and response shapes as backend/app/api/routes.py; detections are stored in IndexedDB
 * (per browser), model/training/dataset pages read a JSON snapshot exported from the real artefacts.
 */
import exifr from 'exifr'
import { get, set } from 'idb-keyval'
import { backend, detect, loadModel } from './engine'
import { asset } from './staticMode'
import type { ClassCode, DetectionResult, Inference, StoredDetection } from './types'

const CODES: ClassCode[] = ['D00', 'D10', 'D20', 'D40']
const NAMES: Record<ClassCode, string> = { D00: 'Longitudinal Crack', D10: 'Transverse Crack', D20: 'Alligator Crack', D40: 'Pothole' }
const COLORS: Record<ClassCode, string> = { D00: '#2a78d6', D10: '#1baf7a', D20: '#4a3aa7', D40: '#eb6834' }
const MODEL_VERSION = 'yolo26s_full_best'
const KEY = 'rdd-yolo-db-v1'

interface DB { nextInf: number; nextDet: number; inferences: Inference[] }
let cache: DB | null = null
async function db(): Promise<DB> {
  if (!cache) cache = ((await get(KEY)) as DB | undefined) ?? { nextInf: 1, nextDet: 1, inferences: [] }
  return cache
}
async function save() { if (cache) await set(KEY, cache) }

class HttpError extends Error {
  status: number
  constructor(status: number, m: string) { super(m); this.status = status }
}
const fail = (status: number, msg: string): never => { throw new HttpError(status, msg) }

// ---------------------------------------------------------------- helpers
function haversine(a: number, b: number, c: number, d: number) {
  const R = 6371008.8, p1 = a * Math.PI / 180, p2 = c * Math.PI / 180
  const dp = p2 - p1, dl = (d - b) * Math.PI / 180
  const x = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(x))
}
const validCoords = (lat?: number | null, lon?: number | null) =>
  lat != null && lon != null && !Number.isNaN(lat) && !Number.isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180

async function toBitmap(blob: Blob): Promise<ImageBitmap> {
  try { return await createImageBitmap(blob, { imageOrientation: 'from-image' }) } catch { return fail(400, 'Could not decode image (use JPEG, PNG, WebP or AVIF in the browser demo)') }
}
function canvasFrom(src: CanvasImageSource, w: number, h: number, maxSide = 1280) {
  const s = Math.min(1, maxSide / Math.max(w, h))
  const c = document.createElement('canvas')
  c.width = Math.round(w * s)
  c.height = Math.round(h * s)
  c.getContext('2d')!.drawImage(src, 0, 0, c.width, c.height)
  return { canvas: c, scale: s }
}
function drawBoxes(base: HTMLCanvasElement, dets: DetectionResult[], s: number) {
  const c = document.createElement('canvas')
  c.width = base.width
  c.height = base.height
  const ctx = c.getContext('2d')!
  ctx.drawImage(base, 0, 0)
  const lw = Math.max(2, Math.round(Math.min(c.width, c.height) / 300))
  ctx.font = `600 ${Math.max(12, lw * 6)}px Inter, sans-serif`
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox.map((v) => v * s)
    ctx.strokeStyle = ctx.fillStyle = COLORS[d.class_code]
    ctx.lineWidth = lw
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    const label = `${d.class_code} ${d.confidence.toFixed(2)} ${d.severity}`
    const tw = ctx.measureText(label).width
    ctx.fillRect(x1, Math.max(0, y1 - lw * 8), tw + 8, lw * 8)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x1 + 4, Math.max(lw * 6, y1 - lw * 2))
  }
  return c.toDataURL('image/jpeg', 0.85)
}
function crop(base: HTMLCanvasElement, b: number[], s: number) {
  const [x1, y1, x2, y2] = b.map((v) => v * s)
  const px = (x2 - x1) * 0.15, py = (y2 - y1) * 0.15
  const xa = Math.max(0, x1 - px), ya = Math.max(0, y1 - py), xb = Math.min(base.width, x2 + px), yb = Math.min(base.height, y2 + py)
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(xb - xa))
  c.height = Math.max(1, Math.round(yb - ya))
  c.getContext('2d')!.drawImage(base, xa, ya, xb - xa, yb - ya, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.85)
}
async function nearbyCounter(lat: number | null, lon: number | null) {
  const all = (await db()).inferences.flatMap((i) => i.detections ?? [])
  return (code: ClassCode) => lat == null || lon == null ? 0
    : all.filter((d) => d.class_code === code && d.latitude != null && haversine(lat, lon, d.latitude!, d.longitude!) <= 15).length
}
function num(fd: FormData, k: string) {
  const v = fd.get(k)
  return v === null || v === '' ? null : Number(v)
}

function stored(inf: Inference, d: DetectionResult, id: number, cropUrl: string | null, extra: Partial<StoredDetection> = {}): StoredDetection {
  return {
    id, inference_id: inf.id, kind: inf.kind, class_code: d.class_code, class_name: d.class_name, confidence: d.confidence,
    bbox: d.bbox, severity: d.severity, severity_score: d.severity_score, severity_detail: d.severity_detail as Record<string, number>,
    latitude: inf.latitude, longitude: inf.longitude, location_source: inf.location_source, frame_index: null, video_time_s: null,
    crop_url: cropUrl, image_url: inf.annotated_url, model_version: MODEL_VERSION, created_at: inf.created_at, ...extra,
  }
}

// ---------------------------------------------------------------- image / frame inference
async function inferImage(fd: FormData, kind: 'image' | 'webcam') {
  const file = fd.get('file') as File | null
  if (!file || !file.size) fail(400, 'Empty upload')
  const conf = num(fd, 'conf') ?? 0.25
  const saveIt = (fd.get('save') ?? (kind === 'image' ? 'true' : 'false')) === 'true'
  let lat = num(fd, 'latitude'), lon = num(fd, 'longitude')
  if ((lat == null) !== (lon == null) || (lat != null && !validCoords(lat, lon))) fail(422, 'Provide both latitude and longitude within valid ranges')
  let source = lat != null ? ((fd.get('location_source') as string) || (kind === 'webcam' ? 'browser' : 'manual')) : null
  let exifGps: { lat: number; lon: number } | null = null
  if (kind === 'image') {
    try {
      const g = await exifr.gps(file!)
      if (g && validCoords(g.latitude, g.longitude) && !(g.latitude === 0 && g.longitude === 0)) {
        exifGps = { lat: +g.latitude.toFixed(7), lon: +g.longitude.toFixed(7) }
        if ((fd.get('prefer_exif') ?? 'true') === 'true' || lat == null) { lat = exifGps.lat; lon = exifGps.lon; source = 'exif' }
      }
    } catch { /* no EXIF */ }
  }
  const accuracy = num(fd, 'location_accuracy_m')
  const bmp = await toBitmap(file!)
  const { detections, ms } = await detect(bmp, conf, saveIt ? await nearbyCounter(lat, lon) : undefined)
  const { canvas, scale } = canvasFrom(bmp, bmp.width, bmp.height)
  const location = lat != null ? { lat, lon: lon!, source: source as never, accuracy_m: accuracy } : null
  const res: Record<string, unknown> = { detections, inference_ms: +ms.toFixed(1), width: bmp.width, height: bmp.height,
    model_version: MODEL_VERSION, location, saved: false, inference_id: null, exif_gps: exifGps }
  if (!saveIt) return kind === 'image' ? { ...res, annotated_base64: drawBoxes(canvas, detections, scale).split(',')[1] } : res
  const d = await db()
  const inf: Inference = {
    id: d.nextInf++, kind, status: 'completed', progress: 1, error: null, source_filename: file!.name,
    image_url: canvas.toDataURL('image/jpeg', 0.85), annotated_url: drawBoxes(canvas, detections, scale), video_url: null,
    output_video_url: null, width: bmp.width, height: bmp.height, latitude: lat, longitude: lon, location_source: source as never,
    model_version: MODEL_VERSION, conf_threshold: conf, inference_ms: +ms.toFixed(1), num_detections: detections.length,
    extra: { location_accuracy_m: accuracy }, created_at: new Date().toISOString(), detections: [],
  }
  inf.detections = detections.map((x) => stored(inf, x, d.nextDet++, crop(canvas, x.bbox, scale)))
  d.inferences.push(inf)
  await save()
  return { ...res, saved: true, inference_id: inf.id, annotated_url: inf.annotated_url, image_url: inf.image_url }
}

// ---------------------------------------------------------------- video (processed in the browser)
function parseRoute(text: string): { t: number; lat: number; lon: number }[] {
  const pts: { t: number; lat: number; lon: number }[] = []
  if (text.trim().startsWith('<')) {
    const docx = new DOMParser().parseFromString(text, 'application/xml')
    let t0: number | null = null
    docx.querySelectorAll('trkpt, rtept').forEach((p) => {
      const tm = p.querySelector('time')?.textContent
      if (!tm) fail(422, 'GPX points need <time>')
      const t = Date.parse(tm!) / 1000
      t0 = t0 ?? t
      pts.push({ t: t - t0, lat: +p.getAttribute('lat')!, lon: +p.getAttribute('lon')! })
    })
  } else {
    const [head, ...rows] = text.trim().split(/\r?\n/)
    const cols = head.split(',').map((c) => c.trim().toLowerCase())
    const li = cols.findIndex((c) => c === 'lat' || c === 'latitude'), oi = cols.findIndex((c) => c === 'lon' || c === 'longitude')
    const ti = cols.indexOf('time_s'), si = cols.indexOf('timestamp')
    if (li < 0 || oi < 0 || (ti < 0 && si < 0)) fail(422, 'Route CSV needs time_s (or timestamp), lat, lon columns')
    let t0: number | null = null
    for (const r of rows) {
      const v = r.split(',')
      let t = ti >= 0 ? +v[ti] : Date.parse(v[si]) / 1000
      if (si >= 0 && ti < 0) { t0 = t0 ?? t; t -= t0 }
      pts.push({ t, lat: +v[li], lon: +v[oi] })
    }
  }
  if (!pts.length || pts.some((p) => !validCoords(p.lat, p.lon))) fail(422, 'Invalid route coordinates')
  return pts.sort((a, b) => a.t - b.t)
}
function routeAt(pts: { t: number; lat: number; lon: number }[], t: number) {
  if (t < pts[0].t || t > pts[pts.length - 1].t) return null
  const i = pts.findIndex((p) => p.t >= t)
  if (i <= 0) return { lat: pts[0].lat, lon: pts[0].lon }
  const a = pts[i - 1], b = pts[i], f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t)
  return { lat: +(a.lat + f * (b.lat - a.lat)).toFixed(7), lon: +(a.lon + f * (b.lon - a.lon)).toFixed(7) }
}
const blobUrls = new Map<number, string>()

async function startVideo(fd: FormData) {
  const file = fd.get('file') as File | null
  if (!file || !file.size) fail(400, 'Empty upload')
  const conf = num(fd, 'conf') ?? 0.25
  const stride = num(fd, 'stride') ?? 5
  const routeFile = fd.get('route_file') as File | null
  const route = routeFile && routeFile.size ? parseRoute(await routeFile.text()) : null
  const lat = num(fd, 'latitude'), lon = num(fd, 'longitude')
  const fixedSrc = lat != null ? ((fd.get('location_source') as string) || 'manual') : null
  await loadModel()
  const d = await db()
  const url = URL.createObjectURL(file!)
  const inf: Inference = {
    id: d.nextInf++, kind: 'video', status: 'processing', progress: 0, error: null, source_filename: file!.name, image_url: null,
    annotated_url: null, video_url: url, output_video_url: url, width: null, height: null,
    latitude: route ? null : lat, longitude: route ? null : lon, location_source: (route ? 'route' : fixedSrc) as never,
    model_version: MODEL_VERSION, conf_threshold: conf, inference_ms: null, num_detections: 0,
    extra: { stride, route_points: route?.length ?? 0, browser_processed: true }, created_at: new Date().toISOString(), detections: [],
  }
  d.inferences.push(inf)
  blobUrls.set(inf.id, url)
  void processVideo(inf, url, conf, stride, route, lat, lon, fixedSrc)
  return { job_id: inf.id, status: inf.status }
}

async function processVideo(inf: Inference, url: string, conf: number, stride: number,
  route: { t: number; lat: number; lon: number }[] | null, lat: number | null, lon: number | null, fixedSrc: string | null) {
  try {
    const v = document.createElement('video')
    v.src = url
    v.muted = true
    v.preload = 'auto'
    await new Promise((ok, bad) => { v.onloadeddata = ok; v.onerror = () => bad(new Error('Cannot decode this video in the browser (try MP4/H.264 or WebM)')) })
    const fps = 30, dt = stride / fps, duration = v.duration
    const W = v.videoWidth, H = v.videoHeight
    const frames: { t: number; dets: DetectionResult[] }[] = []
    let inferMs = 0
    const tracks: { t: number; tLast: number; det: DetectionResult; crop: string }[] = []
    for (let t = 0; t < duration; t += dt) {
      await new Promise((ok) => { v.onseeked = ok; v.currentTime = Math.min(t, duration - 0.001) })
      const { canvas, scale } = canvasFrom(v, W, H)
      const { detections, ms } = await detect(canvas, conf)
      const dets = detections.map((x) => ({ ...x, bbox: x.bbox.map((b) => b / scale) as typeof x.bbox }))
      inferMs += ms
      frames.push({ t: +t.toFixed(3), dets })
      for (const x of dets) {
        const m = [...tracks].reverse().find((tr) => t - tr.tLast <= 1 && tr.det.class_code === x.class_code && iouBox(tr.det.bbox, x.bbox) >= 0.3)
        if (!m) tracks.push({ t, tLast: t, det: x, crop: crop(canvas, x.bbox, scale) })
        else {
          m.tLast = t
          if (x.confidence > m.det.confidence) { m.det = x; m.t = t; m.crop = crop(canvas, x.bbox, scale) }
        }
      }
      inf.progress = Math.min(0.99, t / duration)
    }
    const d = await db()
    inf.detections = tracks.map((tr) => {
      const pos = route ? routeAt(route, tr.t) : lat != null ? { lat, lon: lon! } : null
      return stored(inf, tr.det, d.nextDet++, tr.crop, {
        latitude: pos?.lat ?? null, longitude: pos?.lon ?? null,
        location_source: (pos ? (route ? 'route' : fixedSrc) : null) as never, frame_index: null, video_time_s: +tr.t.toFixed(3),
      })
    })
    inf.width = W
    inf.height = H
    inf.num_detections = inf.detections.length
    inf.inference_ms = frames.length ? +(inferMs / frames.length).toFixed(1) : null
    // Browsers do not expose the source frame rate, so frame counts are not reported - only the sampling interval.
    inf.extra = { ...inf.extra, fps: null, frames: null, sample_interval_s: +dt.toFixed(3), duration_s: duration, processed_frames: frames.length,
      mean_inference_ms: inf.inference_ms, inference_fps: inferMs ? +(1000 * frames.length / inferMs).toFixed(1) : null,
      raw_frame_detections: frames.reduce((a, f) => a + f.dets.length, 0), overlay_frames: frames }
    inf.status = 'completed'
    inf.progress = 1
  } catch (e) {
    inf.status = 'failed'
    inf.error = (e as Error).message
  }
  await save()
}
function iouBox(a: number[], b: number[]) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return u > 0 ? inter / u : 0
}

// ---------------------------------------------------------------- queries
function allDetections(d: DB) {
  return d.inferences.flatMap((i) => (i.detections ?? []).map((x) => ({ ...x, kind: i.kind, image_url: i.annotated_url ?? x.image_url })))
}
function filterDets(list: StoredDetection[], q: URLSearchParams, onlyGeo = false) {
  const csv = (k: string) => q.get(k)?.split(',').filter(Boolean)
  const classes = csv('classes'), sev = csv('severity')?.map((s) => s.toUpperCase()), src = csv('source')
  const minConf = q.get('min_conf') ? +q.get('min_conf')! : null
  const days = q.get('days') ? +q.get('days')! : null, kind = q.get('kind')
  const since = days ? Date.now() - days * 864e5 : null
  const hasLoc = q.get('has_location')
  return list.filter((d) => (!classes || classes.includes(d.class_code)) && (!sev || sev.includes(d.severity))
    && (!src || (d.location_source && src.includes(d.location_source))) && (minConf == null || d.confidence >= minConf)
    && (!since || Date.parse(d.created_at) >= since) && (!kind || d.kind === kind)
    && (!(onlyGeo || hasLoc === 'true') || d.latitude != null) && (hasLoc !== 'false' || d.latitude == null))
}
function stats(d: DB, days: number) {
  const dets = allDetections(d)
  const per: Record<string, unknown> = {}
  for (const c of CODES) {
    const xs = dets.filter((x) => x.class_code === c)
    per[c] = { name: NAMES[c], count: xs.length, avg_confidence: xs.length ? xs.reduce((a, x) => a + x.confidence, 0) / xs.length : null,
      severity: { LOW: xs.filter((x) => x.severity === 'LOW').length, MEDIUM: xs.filter((x) => x.severity === 'MEDIUM').length, HIGH: xs.filter((x) => x.severity === 'HIGH').length } }
  }
  const bins = Array(10).fill(0)
  dets.forEach((x) => { bins[Math.min(9, Math.floor(x.confidence * 10))]++ })
  const over: Record<string, Record<string, number>> = {}
  for (let i = days - 1; i >= 0; i--) over[new Date(Date.now() - i * 864e5).toISOString().slice(0, 10)] = {}
  dets.forEach((x) => { const k = x.created_at.slice(0, 10); if (over[k]) over[k][x.class_code] = (over[k][x.class_code] ?? 0) + 1 })
  const kinds: Record<string, number> = {}
  d.inferences.forEach((i) => { kinds[i.kind] = (kinds[i.kind] ?? 0) + 1 })
  return {
    total_detections: dets.length, total_inferences: d.inferences.length, inferences_by_kind: kinds,
    geolocated_detections: dets.filter((x) => x.latitude != null).length, per_class: per,
    average_confidence: dets.length ? dets.reduce((a, x) => a + x.confidence, 0) / dets.length : null,
    severity: { LOW: dets.filter((x) => x.severity === 'LOW').length, MEDIUM: dets.filter((x) => x.severity === 'MEDIUM').length, HIGH: dets.filter((x) => x.severity === 'HIGH').length },
    confidence_histogram: bins.map((count, i) => ({ bin: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`, count })),
    over_time: Object.entries(over).map(([date, c]) => ({ date, D00: c.D00 ?? 0, D10: c.D10 ?? 0, D20: c.D20 ?? 0, D40: c.D40 ?? 0, total: Object.values(c).reduce((a, b) => a + b, 0) })),
  }
}

const snapshot = (name: string) => fetch(asset(`/data/api/${name}.json`)!).then((r) => { if (!r.ok) fail(404, 'Not found'); return r.json() })

// ---------------------------------------------------------------- router
export async function staticRequest(method: string, path: string, body?: FormData | object): Promise<unknown> {
  const [p, qs] = path.split('?')
  const q = new URLSearchParams(qs ?? '')
  const d = await db()
  let m: RegExpMatchArray | null
  if (method === 'GET') {
    if (p === '/health') return { status: 'ok', model_loaded: true, model_version: `${MODEL_VERSION} (in-browser)`, model_error: null, database: 'browser', cuda_available: false, version: '1.0.0' }
    if (p === '/model') {
      const s = await snapshot('model')
      return { ...s, info: { ...s.info, device: backend === 'not loaded' ? 'browser' : backend } }
    }
    if (p === '/classes' || p === '/severity/rules' || p === '/dataset/stats' || p === '/training/comparison' || p === '/training/experiments')
      return snapshot(p.slice(1).replace(/\//g, '_'))
    if ((m = p.match(/^\/training\/experiments\/([\w-]+)$/))) return snapshot(`training_experiment_${m[1]}`)
    if (p === '/system/hardware') return { hardware: { gpu_name: `Your browser (${backend})`, vram_gb: null, cpu_cores: navigator.hardwareConcurrency, ram_gb: null, torch_version: 'onnxruntime-web 1.30' } }
    if (p === '/stats') return stats(d, +(q.get('days') ?? 30))
    if (p === '/detections') {
      let list = filterDets(allDetections(d), q)
      const sort = (q.get('sort') ?? 'created_at') as keyof StoredDetection, dir = q.get('order') === 'asc' ? 1 : -1
      list = list.sort((a, b) => ((a[sort] as number | string) > (b[sort] as number | string) ? 1 : -1) * dir || b.id - a.id)
      const limit = +(q.get('limit') ?? 50), offset = +(q.get('offset') ?? 0)
      return { total: list.length, limit, offset, items: list.slice(offset, offset + limit) }
    }
    if ((m = p.match(/^\/detections\/(\d+)$/))) return allDetections(d).find((x) => x.id === +m![1]) ?? fail(404, 'Detection not found')
    if (p === '/map/detections') {
      const list = filterDets(allDetections(d), q, true).sort((a, b) => b.created_at.localeCompare(a.created_at))
      return { type: 'FeatureCollection', features: list.map((x) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: [x.longitude, x.latitude] }, properties: x })) }
    }
    if ((m = p.match(/^\/inferences\/(\d+)$/)) || (m = p.match(/^\/inference\/video\/(\d+)$/))) {
      const inf = d.inferences.find((i) => i.id === +m![1]) ?? fail(404, 'Not found')
      return { ...inf, output_video_url: blobUrls.get(inf.id) ?? null, video_url: blobUrls.get(inf.id) ?? null }
    }
    if (p === '/videos') return d.inferences.filter((i) => i.kind === 'video').reverse().slice(0, +(q.get('limit') ?? 20))
      .map((i) => ({ ...i, detections: undefined, output_video_url: blobUrls.get(i.id) ?? null }))
  }
  if (method === 'POST') {
    if (p === '/inference/image') return inferImage(body as FormData, 'image')
    if (p === '/inference/frame') return inferImage(body as FormData, (body as FormData).get('save') === 'true' ? 'webcam' : 'image')
    if (p === '/inference/video') return startVideo(body as FormData)
    if (p === '/model/select') fail(400, 'The browser demo ships one model (YOLO26s). Run the full app locally to switch models.')
  }
  if (method === 'PATCH' && (m = p.match(/^\/inferences\/(\d+)\/location$/))) {
    const inf = d.inferences.find((i) => i.id === +m![1]) ?? fail(404, 'Inference not found')
    const b = body as { latitude: number; longitude: number; source?: string; accuracy_m?: number }
    if (!validCoords(b.latitude, b.longitude)) fail(422, 'Invalid coordinates')
    Object.assign(inf, { latitude: b.latitude, longitude: b.longitude, location_source: b.source ?? 'manual' })
    inf.detections?.forEach((x) => Object.assign(x, { latitude: b.latitude, longitude: b.longitude, location_source: b.source ?? 'manual' }))
    await save()
    return inf
  }
  if (method === 'DELETE') {
    if ((m = p.match(/^\/inferences\/(\d+)$/))) { d.inferences = d.inferences.filter((i) => i.id !== +m![1]); await save(); return undefined }
    if ((m = p.match(/^\/detections\/(\d+)$/))) {
      for (const i of d.inferences) {
        const before = i.detections?.length ?? 0
        i.detections = i.detections?.filter((x) => x.id !== +m![1])
        if ((i.detections?.length ?? 0) < before) i.num_detections = Math.max(0, i.num_detections - 1)
      }
      await save()
      return undefined
    }
  }
  return fail(404, `Not available in the browser demo: ${method} ${p}`)
}
