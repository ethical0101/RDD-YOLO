import { useState } from 'react'
import { CheckCircle2, Cpu } from 'lucide-react'
import { apiSend, useApi } from '../lib/api'
import { CLASS_CODES, CLASS_COLORS, CLASS_NAMES, intFmt, pct } from '../lib/format'
import type { EvalMetrics, ExperimentDetail } from '../lib/types'
import { Button, Card, EmptyState, ErrorState, Notice, PageHeader, Spinner } from '../components/ui'

export default function ModelPage() {
  const model = useApi<{ loaded: boolean; error: string | null; info: any; available: any[] }>('/model')
  const ds = useApi<any>('/dataset/stats')
  const rules = useApi<any>('/severity/rules')
  const hw = useApi<any>('/system/hardware')
  const runName: string | undefined = model.data?.info?.run_name
  const exp = useApi<ExperimentDetail>(runName ? `/training/experiments/${runName}` : null)
  const [switching, setSwitching] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const select = async (w: string) => {
    setSwitching(w)
    setErr(null)
    try { await apiSend('/model/select', 'POST', { weights: w }); await model.reload() } catch (e: any) { setErr(e.message) } finally { setSwitching(null) }
  }

  if (model.loading) return <Spinner />
  if (model.error) return <ErrorState message={model.error} onRetry={model.reload} />
  const info = model.data?.info
  const ri = exp.data?.run_info
  const test: EvalMetrics | null | undefined = exp.data?.evaluations?.test?.metrics

  return (
    <>
      <PageHeader title="Model" subtitle="The trained detector currently serving inference, the data it was trained on, and its measured performance." />
      {err && <div className="mb-4"><ErrorState message={err} /></div>}
      {!model.data?.loaded ? (
        <EmptyState title="No trained model is loaded">{model.data?.error}</EmptyState>
      ) : (
        <div className="grid gap-6 lg:grid-cols-3">
          <Card title="Active model" className="lg:col-span-2">
            <dl className="grid grid-cols-1 gap-x-8 gap-y-2.5 text-sm sm:grid-cols-2">
              <KV k="Model name" v={info.model_name ?? info.weights} />
              <KV k="Checkpoint" v={info.weights} mono />
              <KV k="Architecture file" v={ri?.architecture_yaml ?? info.architecture} mono />
              <KV k="Base framework" v={`Ultralytics ${info.ultralytics_version ?? ri?.ultralytics_version ?? ''} · YOLO26${info.scale ?? ''}`} />
              <KV k="Training status" v={ri?.finished_utc ? <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle2 className="h-4 w-4" />Completed</span> : 'Unknown'} />
              <KV k="Epochs trained" v={ri?.results?.epochs_completed ? `${ri.results.epochs_completed} / ${ri.args?.epochs}` : '—'} />
              <KV k="Best epoch (val)" v={ri?.results?.best_epoch_val?.epoch ?? '—'} />
              <KV k="Training time" v={ri?.train_time_hours ? `${ri.train_time_hours.toFixed(2)} h on ${ri.hardware?.gpu_name ?? 'CPU'}` : '—'} />
              <KV k="Image size" v={`${info.imgsz} × ${info.imgsz}`} />
              <KV k="Parameters" v={intFmt(info.parameters)} />
              <KV k="Inference device" v={info.device === 'cpu' ? 'CPU' : `CUDA:${info.device}`} />
              <KV k="Initialisation" v={ri?.initialisation?.pretrained_source ? `Backbone from COCO ${ri.initialisation.pretrained_source}; neck/head trained from scratch` : '—'} />
            </dl>
          </Card>
          <Card title="Test-split performance" subtitle={test ? `${test.images} held-out images` : undefined}>
            {!test ? <EmptyState title="Evaluation pending">Run training/evaluate.py for this checkpoint.</EmptyState> : (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <Big k="mAP@50" v={pct(test.overall.mAP50)} />
                <Big k="mAP@50-95" v={pct(test.overall.mAP50_95)} />
                <Big k="Precision" v={pct(test.overall.precision)} />
                <Big k="Recall" v={pct(test.overall.recall)} />
                <Big k="F1" v={pct(test.overall.f1)} />
                <Big k="FPS (batch 1)" v={test.speed_benchmark.fps_end_to_end.toFixed(1)} />
              </div>
            )}
          </Card>

          {test && (
            <Card title="Class-wise test metrics" className="lg:col-span-3" bodyClass="p-0">
              <div className="overflow-x-auto"><table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr>
                  {['Class', 'Instances', 'Precision', 'Recall', 'F1', 'mAP@50', 'mAP@50-95'].map((h) => <th key={h} className="px-4 py-2 font-medium first:pl-5">{h}</th>)}</tr></thead>
                <tbody className="divide-y divide-slate-100">
                  {CLASS_CODES.map((c) => { const m = test.per_class[c]; return m ? (
                    <tr key={c}><td className="py-2 pl-5 pr-4"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: CLASS_COLORS[c] }} />{c} · {CLASS_NAMES[c]}</td>
                      <td className="px-4 py-2 tabular-nums">{m.instances ?? '—'}</td><td className="px-4 py-2 tabular-nums">{pct(m.precision)}</td><td className="px-4 py-2 tabular-nums">{pct(m.recall)}</td>
                      <td className="px-4 py-2 tabular-nums">{pct(m.f1)}</td><td className="px-4 py-2 tabular-nums">{pct(m.mAP50)}</td><td className="px-4 py-2 tabular-nums">{pct(m.mAP50_95)}</td></tr>) : null })}
                </tbody></table></div>
            </Card>
          )}

          <Card title="Dataset" className="lg:col-span-2">
            {!ds.data?.available ? <EmptyState title="Dataset statistics unavailable">{ds.data?.message}</EmptyState> : (
              <div className="space-y-4 text-sm">
                <p className="text-slate-600">RDD2022 (CRDDC'2022, CC BY 4.0) — countries: <b>{ds.data.config.countries.join(', ')}</b>. Official train split re-split {ds.data.config.split.map((v: number) => `${v * 100}%`).join(' / ')} (seed {ds.data.config.seed}); the official test split has no public labels.</p>
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-slate-500"><tr><th className="py-1 font-medium">Split</th><th className="font-medium">Images</th><th className="font-medium">Boxes</th>{CLASS_CODES.map((c) => <th key={c} className="font-medium">{c}</th>)}</tr></thead>
                  <tbody className="divide-y divide-slate-100">{(['train', 'val', 'test'] as const).map((s) => { const st = ds.data.splits[s]; return (
                    <tr key={s}><td className="py-1.5 font-medium">{s}</td><td className="tabular-nums">{intFmt(st.images)}</td><td className="tabular-nums">{intFmt(st.instances)}</td>
                      {CLASS_CODES.map((c) => <td key={c} className="tabular-nums">{intFmt(st.instances_per_class[c])}</td>)}</tr>) })}</tbody>
                </table>
                <p className="text-xs text-slate-500">Validation: {intFmt(ds.data.validation.images_valid)} / {intFmt(ds.data.validation.images_total)} images valid · {ds.data.validation.issue_count} issue(s) · non-target labels dropped: {Object.entries(ds.data.validation.ignored_labels ?? {}).map(([k, v]) => `${k} (${v})`).join(', ')}</p>
              </div>
            )}
          </Card>
          <Card title="Detected classes">
            <ul className="space-y-2 text-sm">{CLASS_CODES.map((c, i) => (
              <li key={c} className="flex items-center gap-2"><span className="w-5 text-xs text-slate-400">{i}</span><span className="h-3 w-3 rounded-full" style={{ background: CLASS_COLORS[c] }} /><b>{c}</b> {CLASS_NAMES[c]}</li>))}
            </ul>
          </Card>

          <Card title="Severity estimation rules" className="lg:col-span-2">
            {rules.data && (
              <div className="space-y-3 text-sm">
                <Notice tone="warn">{rules.data.disclaimer}</Notice>
                <code className="block rounded-lg bg-slate-900 p-3 text-xs leading-relaxed text-slate-100">{rules.data.formula}</code>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div><div className="mb-1 text-xs font-medium text-slate-500">Class weights</div>{Object.entries(rules.data.class_weight).map(([k, v]) => <div key={k} className="flex justify-between"><span>{k} {CLASS_NAMES[k as keyof typeof CLASS_NAMES]}</span><span className="tabular-nums">{String(v)}</span></div>)}</div>
                  <div><div className="mb-1 text-xs font-medium text-slate-500">Levels</div>{Object.entries(rules.data.levels).map(([k, v]) => <div key={k} className="flex justify-between"><span>{k}</span><span>score {String(v)}</span></div>)}</div>
                </div>
              </div>
            )}
          </Card>
          <Card title="Available checkpoints">
            <ul className="space-y-2">{model.data.available.map((w) => (
              <li key={w.file} className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm">
                <div className="min-w-0"><div className="truncate font-medium">{w.file}</div><div className="text-xs text-slate-500">{w.size_mb} MB{w.results?.best_epoch_val ? ` · val mAP50 ${pct(w.results.best_epoch_val.mAP50)}` : ''}</div></div>
                {info.weights === w.file ? <span className="text-xs font-medium text-emerald-700">Active</span>
                  : <Button variant="secondary" onClick={() => select(w.file)} disabled={!!switching}>{switching === w.file ? '…' : 'Use'}</Button>}
              </li>))}
            </ul>
            {hw.data && <p className="mt-4 flex items-start gap-1.5 text-xs text-slate-500"><Cpu className="mt-0.5 h-3.5 w-3.5 shrink-0" />{hw.data.hardware.gpu_name ?? 'No CUDA GPU'} · {hw.data.hardware.vram_gb ?? '—'} GB VRAM · {hw.data.hardware.cpu_cores} CPU threads · {hw.data.hardware.ram_gb} GB RAM · torch {hw.data.hardware.torch_version}</p>}
          </Card>
        </div>
      )}
    </>
  )
}

function KV({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return <div className="flex justify-between gap-4 border-b border-slate-100 pb-2"><dt className="text-slate-500">{k}</dt><dd className={`text-right font-medium text-slate-800 ${mono ? 'font-mono text-xs' : ''}`}>{v}</dd></div>
}
function Big({ k, v }: { k: string; v: string }) {
  return <div className="rounded-lg bg-slate-50 px-3 py-2.5"><div className="text-xs text-slate-500">{k}</div><div className="text-xl font-semibold tabular-nums">{v}</div></div>
}
