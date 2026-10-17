import { apiGet } from './api'
import { STATIC } from './staticMode'
import type { StoredDetection } from './types'

export const GPS_NOTE = 'Coordinates are where the photo or video frame was taken (browser GPS, EXIF, manual pin or route ' +
  'interpolation), not the measured position of the damage itself; expect an offset of a few to tens of metres.'

function save(blob: Blob, name: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 10000)
}

const COLS = ['id', 'created_at', 'class_code', 'class_name', 'confidence', 'severity', 'severity_score', 'latitude', 'longitude',
  'location_source', 'kind', 'inference_id', 'video_time_s', 'model_version'] as const

/** Download stored detections as GeoJSON or CSV, using the same filters as the History / Map pages. */
export async function exportDetections(format: 'geojson' | 'csv', filters: Record<string, string | number | undefined>, geolocatedOnly = true) {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15)
  if (!STATIC) {
    const qs = new URLSearchParams(Object.entries({ ...filters, format, geolocated_only: String(geolocatedOnly) })
      .filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => [k, String(v)]))
    const res = await fetch(`/api/export/detections?${qs}`)
    if (!res.ok) throw new Error(`Export failed (HTTP ${res.status})`)
    return save(await res.blob(), `rdd_yolo_detections_${stamp}.${format}`)
  }
  const rows = (await apiGet<{ items: StoredDetection[] }>('/detections', { ...filters, limit: 100000, offset: 0 })).items
    .filter((r) => !geolocatedOnly || r.latitude != null)
  if (format === 'csv') {
    const csv = [`# ${GPS_NOTE}`, COLS.join(','), ...rows.map((r) => COLS.map((c) => JSON.stringify((r as any)[c] ?? '')).join(','))].join('\n')
    return save(new Blob([csv], { type: 'text/csv' }), `rdd_yolo_detections_${stamp}.csv`)
  }
  const gj = {
    type: 'FeatureCollection',
    metadata: { generator: 'RDD-YOLO browser demo', exported_utc: stamp, count: rows.length, location_note: GPS_NOTE },
    features: rows.map((r) => ({ type: 'Feature', geometry: r.latitude != null ? { type: 'Point', coordinates: [r.longitude, r.latitude] } : null, properties: r })),
  }
  save(new Blob([JSON.stringify(gj)], { type: 'application/geo+json' }), `rdd_yolo_detections_${stamp}.geojson`)
}

/** Save an annotated result image (data: or same-origin URL). */
export async function downloadImage(src: string, name: string) {
  save(await (await fetch(src)).blob(), name)
}
