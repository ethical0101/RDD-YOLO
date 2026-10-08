import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import L from '../lib/leaflet-global'
import 'leaflet.markercluster'
import 'leaflet.heat'
import { Flame, Layers, RefreshCw } from 'lucide-react'
import { useApi } from '../lib/api'
import { CLASS_CODES, CLASS_COLORS, CLASS_NAMES, SEVERITIES, SEVERITY_COLORS, SOURCE_LABELS, dateTime, pct } from '../lib/format'
import type { ClassCode, LocationSource, SeverityLevel, StoredDetection } from '../lib/types'
import { Button, Card, ClassBadge, EmptyState, ErrorState, PageHeader, SeverityBadge, SourceBadge, cx } from '../components/ui'

interface FC { type: 'FeatureCollection'; features: { geometry: { coordinates: [number, number] }; properties: StoredDetection }[] }

const SEV_SIZE: Record<SeverityLevel, number> = { LOW: 22, MEDIUM: 26, HIGH: 31 }
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!))

function icon(d: StoredDetection) {
  const s = SEV_SIZE[d.severity]
  return L.divIcon({
    className: '',
    html: `<div class="damage-marker" style="width:${s}px;height:${s}px;background:${CLASS_COLORS[d.class_code]};outline:3px solid ${SEVERITY_COLORS[d.severity]}">${d.class_code.slice(1)}</div>`,
    iconSize: [s, s],
    iconAnchor: [s / 2, s / 2],
  })
}

function popupHtml(d: StoredDetection) {
  const img = d.crop_url ? `<img src="${d.crop_url}" style="width:100%;max-height:140px;object-fit:cover;border-radius:6px;margin-bottom:6px"/>` : ''
  const full = d.image_url ? `<a href="${d.image_url}" target="_blank" rel="noreferrer">Open full annotated image</a>` : ''
  const rows: [string, string][] = [
    ['Damage', `${d.class_code} · ${CLASS_NAMES[d.class_code]}`],
    ['Confidence', pct(d.confidence)],
    ['Severity', `${d.severity} (score ${d.severity_score.toFixed(0)})`],
    ['Latitude', d.latitude!.toFixed(6)],
    ['Longitude', d.longitude!.toFixed(6)],
    ['Location source', d.location_source ? SOURCE_LABELS[d.location_source] : '—'],
    ['Detected', dateTime(d.created_at)],
    ...(d.video_time_s !== null ? [['Video time', `${d.video_time_s.toFixed(2)} s`] as [string, string]] : []),
    ['Model', d.model_version],
    ['Record', `#${d.id} (inference #${d.inference_id})`],
  ]
  return `<div style="width:230px;font-size:12px">${img}<table style="width:100%">${rows
    .map(([k, v]) => `<tr><td style="color:#64748b;padding:1px 6px 1px 0">${k}</td><td style="font-weight:600">${esc(v)}</td></tr>`).join('')}</table>
    <div style="margin-top:6px">${full}</div></div>`
}

function Layers_({ fc, cluster, heat, focus, onReady }: {
  fc: FC | null; cluster: boolean; heat: boolean; focus: number | null; onReady: (m: Map<number, L.Marker>, map: L.Map, group: L.LayerGroup) => void
}) {
  const map = useMap()
  const fitted = useRef(false)
  useEffect(() => {
    if (!fc) return
    const markers = new Map<number, L.Marker>()
    const group: L.LayerGroup = cluster ? L.markerClusterGroup({ maxClusterRadius: 45, showCoverageOnHover: false }) : L.layerGroup()
    const pts: [number, number, number][] = []
    for (const f of fc.features) {
      const d = f.properties
      const [lon, lat] = f.geometry.coordinates
      const m = L.marker([lat, lon], { icon: icon(d), title: `${d.class_code} ${pct(d.confidence)}` }).bindPopup(popupHtml(d))
      markers.set(d.id, m)
      group.addLayer(m)
      pts.push([lat, lon, d.severity === 'HIGH' ? 1 : d.severity === 'MEDIUM' ? 0.6 : 0.3])
    }
    map.addLayer(group)
    const heatLayer = heat && pts.length ? L.heatLayer(pts, { radius: 28, blur: 22, maxZoom: 17 }) : null
    if (heatLayer) map.addLayer(heatLayer)
    if (!fitted.current && pts.length) {
      const focused = focus !== null ? fc.features.filter((f) => f.properties.inference_id === focus) : []
      const src = focused.length ? focused : fc.features
      map.fitBounds(L.latLngBounds(src.map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]])), { maxZoom: 17, padding: [40, 40] })
      if (focused.length) {
        const m = markers.get(focused[0].properties.id)
        const g = group as L.MarkerClusterGroup
        // a clustered marker is not on the map until its cluster is expanded
        if (m) setTimeout(() => (typeof g.zoomToShowLayer === 'function' ? g.zoomToShowLayer(m, () => m.openPopup()) : m.openPopup()), 400)
      }
      fitted.current = true
    }
    onReady(markers, map, group)
    return () => { map.removeLayer(group); if (heatLayer) map.removeLayer(heatLayer) }
  }, [fc, cluster, heat, map]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

export default function MapPage() {
  const [params] = useSearchParams()
  const focus = params.get('focus') ? Number(params.get('focus')) : null
  const [classes, setClasses] = useState<ClassCode[]>([...CLASS_CODES])
  const [sev, setSev] = useState<SeverityLevel[]>([...SEVERITIES])
  const [sources, setSources] = useState<LocationSource[]>(['browser', 'exif', 'manual', 'route'])
  const [minConf, setMinConf] = useState(0)
  const [days, setDays] = useState<number | ''>('')
  const [cluster, setCluster] = useState(true)
  const [heat, setHeat] = useState(false)
  const markersRef = useRef<{ markers: Map<number, L.Marker>; map: L.Map; group: L.LayerGroup } | null>(null)

  const q = useMemo(() => ({ classes: classes.join(','), severity: sev.join(','), source: sources.join(','), min_conf: minConf || undefined, days: days || undefined }),
    [classes, sev, sources, minConf, days])
  const empty = !classes.length || !sev.length || !sources.length
  const { data, error, reload, loading } = useApi<FC>(empty ? null : '/map/detections', q)
  const fc = empty ? { type: 'FeatureCollection' as const, features: [] } : data
  const recent = (fc?.features ?? []).slice(0, 12).map((f) => f.properties)

  const toggle = <T,>(arr: T[], v: T, set: (x: T[]) => void) => set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v])
  const flyTo = (d: StoredDetection) => {
    const r = markersRef.current
    const m = r?.markers.get(d.id)
    if (!r || !m) return
    const g = r.group as L.MarkerClusterGroup
    if (typeof g.zoomToShowLayer === 'function') g.zoomToShowLayer(m, () => m.openPopup())
    else { r.map.flyTo(m.getLatLng(), 18, { duration: 0.6 }); setTimeout(() => m.openPopup(), 700) }
  }

  return (
    <>
      <PageHeader title="Damage Map" subtitle="OpenStreetMap view of every geolocated detection. Marker colour = damage class, ring = severity, size grows with severity."
        actions={<Button variant="secondary" onClick={reload}><RefreshCw className={cx('h-4 w-4', loading && 'animate-spin')} /> Refresh</Button>} />
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs font-medium text-slate-500">Class</span>
          {CLASS_CODES.map((c) => (
            <button key={c} onClick={() => toggle(classes, c, setClasses)} title={CLASS_NAMES[c]}
              className={cx('inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs ring-1', classes.includes(c) ? 'bg-slate-50 ring-slate-300' : 'text-slate-400 ring-slate-200')}>
              <span className="h-2 w-2 rounded-full" style={{ background: classes.includes(c) ? CLASS_COLORS[c] : '#cbd5e1' }} />{c} {CLASS_NAMES[c]}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-xs font-medium text-slate-500">Severity</span>
          {SEVERITIES.map((s) => (
            <button key={s} onClick={() => toggle(sev, s, setSev)}
              className={cx('inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs ring-1', sev.includes(s) ? 'bg-slate-50 ring-slate-300' : 'text-slate-400 ring-slate-200')}>
              <span className="h-2 w-2 rounded-sm" style={{ background: sev.includes(s) ? SEVERITY_COLORS[s] : '#cbd5e1' }} />{s}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-xs font-medium text-slate-500">Source</span>
          {(Object.keys(SOURCE_LABELS) as LocationSource[]).map((s) => (
            <button key={s} onClick={() => toggle(sources, s, setSources)}
              className={cx('rounded-md px-2 py-1 text-xs ring-1', sources.includes(s) ? 'bg-slate-50 ring-slate-300' : 'text-slate-400 ring-slate-200')}>{SOURCE_LABELS[s]}</button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-slate-600">Min conf {minConf.toFixed(2)}
          <input type="range" min={0} max={0.9} step={0.05} value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} className="w-24 accent-slate-900" /></label>
        <select value={days} onChange={(e) => setDays(e.target.value ? Number(e.target.value) : '')} className="rounded-md border border-slate-300 px-2 py-1 text-xs">
          <option value="">All time</option><option value="1">Last 24 h</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option>
        </select>
        <div className="ml-auto flex gap-1.5">
          <button onClick={() => setCluster(!cluster)} className={cx('inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs ring-1', cluster ? 'bg-slate-900 text-white ring-slate-900' : 'ring-slate-300')}><Layers className="h-3.5 w-3.5" /> Cluster</button>
          <button onClick={() => setHeat(!heat)} className={cx('inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs ring-1', heat ? 'bg-slate-900 text-white ring-slate-900' : 'ring-slate-300')}><Flame className="h-3.5 w-3.5" /> Heatmap</button>
        </div>
      </div>
      {error && <div className="mb-4"><ErrorState message={error} onRetry={reload} /></div>}
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="h-[620px] overflow-hidden rounded-xl border border-slate-200 shadow-sm">
          <MapContainer center={[20.5937, 78.9629]} zoom={5} className="h-full w-full">
            <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19} />
            <Layers_ fc={fc} cluster={cluster} heat={heat} focus={focus} onReady={(markers, map, group) => { markersRef.current = { markers, map, group } }} />
          </MapContainer>
        </div>
        <Card title="Recent geolocated detections" subtitle={fc ? `${fc.features.length} marker(s) match the filters` : undefined} bodyClass="p-0">
          {!fc?.features.length ? (
            <div className="p-5"><EmptyState title="No geolocated detections">Run image detection with browser GPS, a manual point, an EXIF-tagged photo, or a video with a GPS route.</EmptyState></div>
          ) : (
            <ul className="max-h-[560px] divide-y divide-slate-100 overflow-y-auto">
              {recent.map((d) => (
                <li key={d.id}><button onClick={() => flyTo(d)} className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-slate-50">
                  {d.crop_url ? <img src={d.crop_url} className="h-10 w-10 rounded object-cover" alt="" /> : <div className="h-10 w-10 rounded bg-slate-100" />}
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex flex-wrap gap-1.5"><ClassBadge code={d.class_code} showName={false} /><SeverityBadge level={d.severity} /></div>
                    <div className="flex items-center gap-2 text-xs text-slate-500">{pct(d.confidence)} · {dateTime(d.created_at)}</div>
                  </div>
                  <SourceBadge source={d.location_source} />
                </button></li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </>
  )
}
