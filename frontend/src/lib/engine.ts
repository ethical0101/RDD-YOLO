/**
 * In-browser inference with ONNX Runtime Web (static deployment).
 * Mirrors the server pipeline: letterbox to 640 (pad 114) -> YOLO26 head [1, 8, 8400]
 * (cx, cy, w, h + 4 class scores) -> confidence filter -> class-wise NMS (IoU 0.7) -> map to image.
 * Verified against the PyTorch model: identical classes/confidences, boxes within 1 px.
 */
import * as ort from 'onnxruntime-web/webgpu'
import { asset } from './staticMode'
import type { ClassCode, DetectionResult, SeverityLevel } from './types'

const CODES: ClassCode[] = ['D00', 'D10', 'D20', 'D40']
const NAMES: Record<ClassCode, string> = { D00: 'Longitudinal Crack', D10: 'Transverse Crack', D20: 'Alligator Crack', D40: 'Pothole' }
const SIZE = 640

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'
ort.env.wasm.numThreads = 1 // GitHub Pages cannot enable cross-origin isolation (needed for threads)

let session: ort.InferenceSession | null = null
let loading: Promise<ort.InferenceSession> | null = null
export let backend = 'not loaded'

export function loadModel(): Promise<ort.InferenceSession> {
  if (session) return Promise.resolve(session)
  if (!loading) {
    loading = (async () => {
      const url = asset('/data/model/yolo26s_full_best.onnx')!
      const providers: string[][] = 'gpu' in navigator ? [['webgpu'], ['wasm']] : [['wasm']]
      let last: unknown
      for (const ep of providers) {
        try {
          session = await ort.InferenceSession.create(url, { executionProviders: ep, graphOptimizationLevel: 'all' })
          backend = ep[0] === 'webgpu' ? 'WebGPU' : 'WebAssembly (CPU)'
          return session
        } catch (e) { last = e }
      }
      loading = null
      throw last
    })()
  }
  return loading
}

// ---------------------------------------------------------------- severity (same rules as rdd_yolo/severity.py)
const CLASS_WEIGHT: Record<ClassCode, number> = { D00: 0.45, D10: 0.5, D20: 0.8, D40: 1.0 }
export function severity(code: ClassCode, b: number[], w: number, h: number, conf: number, nearby = 0) {
  const areaRatio = Math.max(0, (b[2] - b[0]) * (b[3] - b[1])) / Math.max(w * h, 1)
  const areaScore = Math.min(1, Math.sqrt(areaRatio / 0.25))
  const bonus = Math.min(15, 5 * Math.max(0, nearby))
  const score = Math.round(Math.min(100, 100 * (0.45 * CLASS_WEIGHT[code] + 0.4 * areaScore + 0.15 * conf) + bonus) * 10) / 10
  const level: SeverityLevel = score >= 65 ? 'HIGH' : score >= 40 ? 'MEDIUM' : 'LOW'
  return {
    level, score,
    detail: { level, score, class_weight: CLASS_WEIGHT[code], area_ratio: +areaRatio.toFixed(5), area_score: +areaScore.toFixed(4),
      confidence: +conf.toFixed(4), repeat_count: nearby, repeat_bonus: bonus },
  }
}

// ---------------------------------------------------------------- detection
function iou(a: number[], b: number[]) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = ix * iy
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

export async function detect(source: CanvasImageSource & { width: number; height: number }, conf = 0.25,
  nearby?: (code: ClassCode) => number): Promise<{ detections: DetectionResult[]; ms: number }> {
  const s = await loadModel()
  const W = source.width, H = source.height
  const r = Math.min(SIZE / H, SIZE / W)
  const nw = Math.round(W * r), nh = Math.round(H * r)
  const left = Math.floor((SIZE - nw) / 2), top = Math.floor((SIZE - nh) / 2)
  const c = document.createElement('canvas')
  c.width = SIZE
  c.height = SIZE
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = 'rgb(114,114,114)'
  ctx.fillRect(0, 0, SIZE, SIZE)
  ctx.drawImage(source, left, top, nw, nh)
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data
  const plane = SIZE * SIZE
  const input = new Float32Array(3 * plane)
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4] / 255
    input[plane + i] = px[i * 4 + 1] / 255
    input[2 * plane + i] = px[i * 4 + 2] / 255
  }
  const t0 = performance.now()
  const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) })
  const ms = performance.now() - t0
  const data = out[s.outputNames[0]].data as Float32Array
  const n = 8400
  const cand: { b: number[]; c: number; s: number }[] = []
  for (let i = 0; i < n; i++) {
    let best = 0, k = 0
    for (let j = 0; j < 4; j++) {
      const v = data[(4 + j) * n + i]
      if (v > best) { best = v; k = j }
    }
    if (best < conf) continue
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i]
    cand.push({ b: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], c: k, s: best })
  }
  cand.sort((a, b) => b.s - a.s)
  const kept: typeof cand = []
  for (const d of cand) if (!kept.some((k) => k.c === d.c && iou(k.b, d.b) > 0.7)) kept.push(d)
  const detections = kept.slice(0, 300).map((d) => {
    const b = [(d.b[0] - left) / r, (d.b[1] - top) / r, (d.b[2] - left) / r, (d.b[3] - top) / r]
      .map((v, i) => Math.round(Math.min(Math.max(v, 0), i % 2 ? H : W) * 10) / 10)
    const code = CODES[d.c]
    const sev = severity(code, b, W, H, d.s, nearby ? nearby(code) : 0)
    return {
      class_id: d.c, class_code: code, class_name: NAMES[code], confidence: Math.round(d.s * 1e4) / 1e4,
      bbox: b as [number, number, number, number], bbox_norm: [b[0] / W, b[1] / H, b[2] / W, b[3] / H] as [number, number, number, number],
      severity: sev.level, severity_score: sev.score, severity_detail: sev.detail,
    }
  })
  return { detections, ms }
}
