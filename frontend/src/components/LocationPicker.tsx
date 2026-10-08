import { useEffect } from 'react'
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import { Crosshair, MapPin } from 'lucide-react'
import { useBrowserLocation } from '../lib/geolocation'
import type { LocationSource } from '../lib/types'
import { Button, cx, inputCls } from './ui'

export interface PickedLocation { lat: number | null; lon: number | null; source: LocationSource | null; accuracy: number | null }
export const NO_LOCATION: PickedLocation = { lat: null, lon: null, source: null, accuracy: null }

export const pinIcon = L.divIcon({
  className: '',
  html: '<div style="width:18px;height:18px;border-radius:9999px;background:#0f172a;border:3px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>',
  iconSize: [18, 18],
  iconAnchor: [9, 9],
})

function ClickToSet({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({ click: (e) => onPick(e.latlng.lat, e.latlng.lng) })
  return null
}

function Recenter({ lat, lon }: { lat: number | null; lon: number | null }) {
  const map = useMap()
  useEffect(() => { if (lat !== null && lon !== null) map.setView([lat, lon], Math.max(map.getZoom(), 15)) }, [lat, lon, map])
  return null
}

/**
 * Lets the user attach a real location: browser GPS (with permission) or a manual
 * point (typed or clicked on the OpenStreetMap). EXIF GPS is applied server-side.
 */
export default function LocationPicker({ value, onChange, compact }: {
  value: PickedLocation; onChange: (v: PickedLocation) => void; compact?: boolean
}) {
  const geo = useBrowserLocation()
  const useBrowser = async () => {
    const f = await geo.request()
    if (f) onChange({ lat: f.lat, lon: f.lon, source: 'browser', accuracy: f.accuracy })
  }
  const setManual = (lat: number | null, lon: number | null) => onChange({ lat, lon, source: lat !== null && lon !== null ? 'manual' : null, accuracy: null })
  const center: [number, number] = value.lat !== null && value.lon !== null ? [value.lat, value.lon] : [20.5937, 78.9629]

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="secondary" onClick={useBrowser} disabled={geo.pending}>
          <Crosshair className="h-4 w-4" /> {geo.pending ? 'Locating…' : 'Use my location'}
        </Button>
        {value.source && <Button type="button" variant="ghost" onClick={() => onChange({ lat: null, lon: null, source: null, accuracy: null })}>Clear</Button>}
      </div>
      {geo.error && <p className="text-xs text-red-600">{geo.error}. You can set the point manually instead.</p>}
      <div className="grid grid-cols-2 gap-2">
        <input className={inputCls} type="number" step="any" placeholder="Latitude" value={value.lat ?? ''}
          onChange={(e) => setManual(e.target.value === '' ? null : Number(e.target.value), value.lon)} />
        <input className={inputCls} type="number" step="any" placeholder="Longitude" value={value.lon ?? ''}
          onChange={(e) => setManual(value.lat, e.target.value === '' ? null : Number(e.target.value))} />
      </div>
      <div className={cx('overflow-hidden rounded-lg border border-slate-200', compact ? 'h-44' : 'h-56')}>
        <MapContainer center={center} zoom={value.lat !== null ? 15 : 4} className="h-full w-full">
          <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <ClickToSet onPick={(la, lo) => setManual(Number(la.toFixed(7)), Number(lo.toFixed(7)))} />
          <Recenter lat={value.lat} lon={value.lon} />
          {value.lat !== null && value.lon !== null && <Marker position={[value.lat, value.lon]} icon={pinIcon} />}
        </MapContainer>
      </div>
      <p className="flex items-center gap-1.5 text-xs text-slate-500">
        <MapPin className="h-3.5 w-3.5" />
        {value.source
          ? <>Source: <b className="text-slate-700">{value.source === 'browser' ? `Browser GPS (±${Math.round(value.accuracy ?? 0)} m)` : 'Manual'}</b></>
          : 'Click the map, type coordinates, or use browser GPS. If the photo has EXIF GPS it is used automatically.'}
      </p>
    </div>
  )
}
