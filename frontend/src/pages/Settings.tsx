import { useState } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { DEFAULTS, loadSettings, saveSettings, type Settings as S } from '../lib/settings'
import { Button, Card, Field, Notice, PageHeader } from '../components/ui'

export default function Settings() {
  const [s, setS] = useState<S>(loadSettings)
  const [saved, setSaved] = useState(false)
  const set = (patch: Partial<S>) => { setS({ ...s, ...patch }); setSaved(false) }
  return (
    <>
      <PageHeader title="Settings" subtitle="Stored in this browser and used as defaults on the detection pages." />
      <div className="max-w-2xl space-y-6">
        <Card title="Detection defaults">
          <div className="space-y-5">
            <Field label={`Default confidence threshold: ${s.defaultConf.toFixed(2)}`}
              hint="Lower values find more damage but add false alarms. The precision and recall figures in the reports use 0.25.">
              <input type="range" min={0.05} max={0.9} step={0.05} value={s.defaultConf} onChange={(e) => set({ defaultConf: Number(e.target.value) })} className="w-full accent-slate-900" />
            </Field>
            <label className="flex items-start gap-3 text-sm">
              <input type="checkbox" checked={s.preferExif} onChange={(e) => set({ preferExif: e.target.checked })} className="mt-0.5 accent-slate-900" />
              <span><b>Prefer the photo's EXIF GPS</b> over the location chosen on the page when both exist.</span>
            </label>
            <label className="flex items-start gap-3 text-sm">
              <input type="checkbox" checked={s.saveResults} onChange={(e) => set({ saveResults: e.target.checked })} className="mt-0.5 accent-slate-900" />
              <span><b>Save image results</b> to the history and map by default.</span>
            </label>
          </div>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button onClick={() => { saveSettings(s); setSaved(true) }}>Save settings</Button>
            <Button variant="secondary" onClick={() => set({ ...DEFAULTS })}>Reset to defaults</Button>
            {saved && <span className="inline-flex items-center gap-1 text-sm text-emerald-700"><CheckCircle2 className="h-4 w-4" /> Saved</span>}
          </div>
        </Card>
        <Notice>Non-maximum suppression uses a fixed IoU of 0.7, the value the model was validated with. It is not adjustable, because changing it would change what the reported metrics mean.</Notice>
      </div>
    </>
  )
}
