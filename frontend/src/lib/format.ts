import type { ClassCode, LocationSource, SeverityLevel } from './types'

export const CLASS_CODES: ClassCode[] = ['D00', 'D10', 'D20', 'D40']
export const CLASS_NAMES: Record<ClassCode, string> = {
  D00: 'Longitudinal Crack',
  D10: 'Transverse Crack',
  D20: 'Alligator Crack',
  D40: 'Pothole',
}
export const CLASS_COLORS: Record<ClassCode, string> = {
  D00: '#2a78d6',
  D10: '#1baf7a',
  D20: '#4a3aa7',
  D40: '#eb6834',
}
export const SEVERITIES: SeverityLevel[] = ['LOW', 'MEDIUM', 'HIGH']
export const SEVERITY_COLORS: Record<SeverityLevel, string> = { LOW: '#0ca30c', MEDIUM: '#fab219', HIGH: '#d03b3b' }
export const SOURCE_LABELS: Record<LocationSource, string> = {
  browser: 'Browser GPS',
  exif: 'EXIF GPS',
  manual: 'Manual',
  route: 'Route',
}

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`
export const num = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits)
export const coord = (lat: number | null, lon: number | null) =>
  lat === null || lon === null ? '—' : `${lat.toFixed(5)}, ${lon.toFixed(5)}`
export const dateTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—'
export const intFmt = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toLocaleString())

/** 'cpu' -> CPU, '0' -> CUDA:0, anything else (e.g. WebGPU in the browser demo) as is. */
export const deviceLabel = (d: string | null | undefined) =>
  !d ? '—' : d === 'cpu' ? 'CPU' : /^\d+$/.test(d) ? `CUDA:${d}` : d
