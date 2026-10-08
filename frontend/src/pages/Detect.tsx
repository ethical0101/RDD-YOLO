import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { CheckCircle2, ImageUp, Info, MapPinned, ScanSearch } from 'lucide-react'
import { apiSend } from '../lib/api'
import { pct } from '../lib/format'
import type { ImageInferenceResponse } from '../lib/types'
import DetectionViewer from '../components/DetectionViewer'
import LocationPicker, { NO_LOCATION, type PickedLocation } from '../components/LocationPicker'
import { Button, Card, ClassBadge, EmptyState, ErrorState, Field, Notice, PageHeader, SeverityBadge, SourceBadge } from '../components/ui'

export default function Detect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [conf, setConf] = useState(0.25)
  const [save, setSave] = useState(true)
  const [loc, setLoc] = useState<PickedLocation>(NO_LOCATION)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<ImageInferenceResponse | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [locSaved, setLocSaved] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const pick = (f: File | undefined) => {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setRes(null)
    setError(null)
  }

  const run = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setLocSaved(false)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('conf', String(conf))
    fd.append('save', String(save))
    if (loc.lat !== null && loc.lon !== null) {
      fd.append('latitude', String(loc.lat))
      fd.append('longitude', String(loc.lon))
      fd.append('location_source', loc.source ?? 'manual')
      if (loc.accuracy !== null) fd.append('location_accuracy_m', String(loc.accuracy))
    }
    try {
      setRes(await apiSend<ImageInferenceResponse>('/inference/image', 'POST', fd))
    } catch (e: any) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const attachLocation = async () => {
    if (!res?.inference_id || loc.lat === null || loc.lon === null) return
    try {
      await apiSend(`/inferences/${res.inference_id}/location`, 'PATCH',
        { latitude: loc.lat, longitude: loc.lon, source: loc.source ?? 'manual', accuracy_m: loc.accuracy })
      setRes({ ...res, location: { lat: loc.lat, lon: loc.lon, source: loc.source ?? 'manual', accuracy_m: loc.accuracy } })
      setLocSaved(true)
    } catch (e: any) { setError(e.message) }
  }

  const imgSrc = res?.annotated_base64 ? null : preview
  return (
    <>
      <PageHeader title="Image Detection" subtitle="Upload a road photo. The trained model localises damage, classifies it (D00/D10/D20/D40) and estimates a heuristic severity." />
      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <Card title="1 · Image">
            <div onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]) }}
              onClick={() => inputRef.current?.click()}
              className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 px-4 py-8 text-center hover:border-slate-400 hover:bg-slate-50">
              <ImageUp className="h-8 w-8 text-slate-400" />
              <div className="mt-2 text-sm font-medium text-slate-700">{file ? file.name : 'Drop an image or click to browse'}</div>
              <div className="text-xs text-slate-500">JPEG / PNG / WebP</div>
            </div>
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
            <div className="mt-4 space-y-3">
              <Field label={`Confidence threshold: ${conf.toFixed(2)}`}>
                <input type="range" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} className="w-full accent-slate-900" />
              </Field>
              <label className="flex items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} className="accent-slate-900" />
                Save detections to the database
              </label>
            </div>
          </Card>
          <Card title="2 · Location" subtitle="Optional. Never guessed — only real GPS or a point you choose.">
            <LocationPicker value={loc} onChange={setLoc} compact />
          </Card>
          <Button onClick={run} disabled={!file || busy} className="w-full py-2.5">
            <ScanSearch className="h-4 w-4" /> {busy ? 'Running inference…' : 'Detect road damage'}
          </Button>
          {error && <ErrorState message={error} />}
        </div>

        <div className="space-y-6">
          <Card title="Result" subtitle={res ? `${res.detections.length} detection(s) · ${res.inference_ms.toFixed(0)} ms · model ${res.model_version}` : undefined}
            actions={res?.saved && <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700"><CheckCircle2 className="h-4 w-4" /> Saved #{res.inference_id}</span>}>
            {!preview ? (
              <EmptyState title="No image selected" icon={<ImageUp className="h-8 w-8" />}>Choose a road image to start.</EmptyState>
            ) : res ? (
              res.annotated_base64 && !imgSrc
                ? <img src={`data:image/jpeg;base64,${res.annotated_base64}`} className="w-full rounded-lg" alt="Annotated" />
                : <DetectionViewer src={preview} width={res.width} height={res.height} detections={res.detections} highlight={hover} onHover={setHover} />
            ) : (
              <img src={preview} className="w-full rounded-lg" alt="Preview" />
            )}
          </Card>

          {res && (
            <Card title="Detections" bodyClass="p-0">
              {res.detections.length === 0 ? (
                <div className="p-5"><EmptyState title="No road damage detected">Nothing above the {conf.toFixed(2)} confidence threshold. Try lowering it.</EmptyState></div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-left text-xs text-slate-500">
                      <tr><th className="px-5 py-2 font-medium">#</th><th className="px-3 py-2 font-medium">Class</th><th className="px-3 py-2 font-medium">Confidence</th>
                        <th className="px-3 py-2 font-medium">Severity</th><th className="px-3 py-2 font-medium">Box area</th><th className="px-3 py-2 font-medium">BBox (px)</th></tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {res.detections.map((d, i) => (
                        <tr key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className={hover === i ? 'bg-slate-50' : ''}>
                          <td className="px-5 py-2 text-slate-500">{i + 1}</td>
                          <td className="px-3 py-2"><ClassBadge code={d.class_code} /></td>
                          <td className="px-3 py-2 tabular-nums">{pct(d.confidence)}</td>
                          <td className="px-3 py-2"><SeverityBadge level={d.severity} score={d.severity_score} /></td>
                          <td className="px-3 py-2 tabular-nums">{pct(Number(d.severity_detail.area_ratio), 2)}</td>
                          <td className="px-3 py-2 font-mono text-xs text-slate-500">{d.bbox.map((v) => v.toFixed(0)).join(', ')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 px-5 py-3 text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">Location:</span>
                  {res.location ? <><SourceBadge source={res.location.source} /><span className="font-mono text-xs">{res.location.lat.toFixed(6)}, {res.location.lon.toFixed(6)}</span></>
                    : <span className="text-slate-400">none attached</span>}
                  {res.exif_gps && res.location?.source !== 'exif' && <span className="text-xs text-slate-500">(EXIF GPS present)</span>}
                </div>
                <div className="flex gap-2">
                  {res.saved && !res.location && loc.lat !== null && (
                    <Button variant="secondary" onClick={attachLocation}><MapPinned className="h-4 w-4" /> Attach selected location</Button>
                  )}
                  {res.saved && res.location && res.detections.length > 0 && <Link to={`/map?focus=${res.inference_id}`}><Button variant="secondary"><MapPinned className="h-4 w-4" /> View on map</Button></Link>}
                </div>
              </div>
              {locSaved && <div className="px-5 pb-3 text-xs text-emerald-700">Location attached to all detections of this image.</div>}
            </Card>
          )}
          {res && res.detections.length > 0 && (
            <Notice><Info className="mr-1 inline h-4 w-4" />Severity is a transparent heuristic (class, relative box area, confidence, repeat reports) for prioritising review — not an engineering-certified road condition rating.</Notice>
          )}
        </div>
      </div>
    </>
  )
}
