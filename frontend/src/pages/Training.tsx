import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../lib/api'
import { CLASS_CODES, CLASS_NAMES, intFmt, pct } from '../lib/format'
import type { ExperimentDetail, ExperimentSummary } from '../lib/types'
import { asset } from '../lib/staticMode'
import { Card, EmptyState, ErrorState, Notice, PageHeader, Spinner, cx } from '../components/ui'

const axis = { stroke: '#c3c2b7', tick: { fill: '#898781', fontSize: 11 }, tickLine: false }
const grid = <CartesianGrid stroke="#e1e0d9" vertical={false} />
const tip = { contentStyle: { borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12 } }
const BASE = '#2a78d6'
const RDD = '#eb6834'

export default function Training() {
  const exps = useApi<ExperimentSummary[]>('/training/experiments')
  const cmp = useApi<any>('/training/comparison')
  const [sel, setSel] = useState<string | null>(null)
  const real = (exps.data ?? []).filter((e) => !e.name.startsWith('smoke'))
  useEffect(() => { if (!sel && real.length) setSel(real.find((e) => e.experiment === 'rdd')?.name ?? real[0].name) }, [real.length]) // eslint-disable-line
  const detail = useApi<ExperimentDetail>(sel ? `/training/experiments/${sel}` : null, undefined, [sel])

  return (
    <>
      <PageHeader title="Training & Experiments" subtitle="Results produced by the actual training and evaluation runs in experiments/. Nothing here is typed in by hand." />
      <div className="space-y-6">
        <Architecture />
        <Comparison cmp={cmp.data} loading={cmp.loading} />
        <Card title="Experiment runs" bodyClass="p-0">
          {exps.loading ? <Spinner /> : exps.error ? <div className="p-5"><ErrorState message={exps.error} /></div> : !real.length ? (
            <div className="p-5"><EmptyState title="No training runs yet">Start training with scripts\run_experiments.ps1.</EmptyState></div>
          ) : (
            <div className="overflow-x-auto"><table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr>{['Run', 'Status', 'Epochs', 'Time', 'Best val mAP@50', 'Best val mAP@50-95', 'Test mAP@50', 'Test F1'].map((h) => <th key={h} className="px-4 py-2 font-medium first:pl-5">{h}</th>)}</tr></thead>
              <tbody className="divide-y divide-slate-100">{real.map((e) => (
                <tr key={e.name} onClick={() => setSel(e.name)} className={cx('cursor-pointer hover:bg-slate-50', sel === e.name && 'bg-slate-50')}>
                  <td className="py-2 pl-5 pr-4"><div className="font-medium">{e.name}</div><div className="text-xs text-slate-500">{e.title}</div></td>
                  <td className="px-4 py-2"><Status s={e.status} /></td>
                  <td className="px-4 py-2 tabular-nums">{e.epochs_completed}{e.epochs_requested ? ` / ${e.epochs_requested}` : ''}</td>
                  <td className="px-4 py-2 tabular-nums">{e.train_time_hours ? `${e.train_time_hours.toFixed(2)} h` : '—'}</td>
                  <td className="px-4 py-2 tabular-nums">{pct(e.best_val?.mAP50)}</td><td className="px-4 py-2 tabular-nums">{pct(e.best_val?.mAP50_95)}</td>
                  <td className="px-4 py-2 tabular-nums">{e.test_metrics ? pct(e.test_metrics.mAP50) : <span className="text-xs text-slate-400">pending</span>}</td>
                  <td className="px-4 py-2 tabular-nums">{e.test_metrics ? pct(e.test_metrics.f1) : <span className="text-xs text-slate-400">pending</span>}</td>
                </tr>))}</tbody>
            </table></div>
          )}
        </Card>
        {sel && (detail.loading && !detail.data ? <Spinner /> : detail.data && <RunDetail d={detail.data} />)}
      </div>
    </>
  )
}

function Status({ s }: { s: string }) {
  const m: Record<string, string> = { completed: 'bg-emerald-50 text-emerald-800 ring-emerald-200', running_or_interrupted: 'bg-amber-50 text-amber-900 ring-amber-200', pending: 'bg-slate-50 text-slate-600 ring-slate-200' }
  return <span className={cx('rounded-md px-2 py-0.5 text-xs font-medium ring-1', m[s] ?? m.pending)}>{s === 'running_or_interrupted' ? 'running / interrupted' : s}</span>
}

function Architecture() {
  return (
    <Card title="Baseline vs RDD-YOLO: what changed" subtitle="Both models: YOLO26n, same data, same hyper-parameters, same seed and the same initialisation (COCO-pretrained backbone, neck + head from scratch).">
      <div className="grid gap-4 md:grid-cols-3">
        {[
          ['SimAM attention', 'Parameter-free 3-D attention appended after the backbone (C2PSA). Each neuron is weighted by sigmoid(1/E), where E is its energy vs. its channel; distinctive (damage) activations are emphasised. 0 extra parameters.'],
          ['Bilinear upsampling', 'Both neck upsamplers use corner-aligned bilinear interpolation instead of nearest-neighbour, to keep thin crack edges smooth when P5/P4 features are enlarged.'],
          ['GhostConv in the neck', 'The two 3×3 stride-2 downsampling convolutions of the PAN path are replaced by GhostConv (half the maps from a conv, half from cheap depth-wise ops), reducing parameters and FLOPs.'],
        ].map(([t, d]) => (
          <div key={t} className="rounded-lg border border-slate-200 p-4"><div className="text-sm font-semibold">{t}</div><p className="mt-1 text-sm text-slate-600">{d}</p></div>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-500">Architectures: <code>models/architectures/yolo26-baseline.yaml</code> and <code>yolo26-rdd.yaml</code>. The original paper applied these changes to YOLOv8x on an RTX 4090; this project ports them to YOLO26n to fit a 4 GB RTX 3050, so absolute numbers are not comparable to the paper.</p>
    </Card>
  )
}

function Comparison({ cmp, loading }: { cmp: any; loading: boolean }) {
  if (loading) return <Spinner />
  if (!cmp?.available) return <Notice tone="warn"><b>Comparison pending.</b> {cmp?.message}</Notice>
  const rows = [['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1'], ['mAP50', 'mAP@50'], ['mAP50_95', 'mAP@50-95']] as const
  const chart = rows.map(([k, l]) => ({ metric: l, Baseline: cmp.overall[k].baseline, 'RDD-YOLO': cmp.overall[k].rdd }))
  const e = cmp.efficiency
  return (
    <Card title={`Experiment A vs B — ${cmp.split} split`} subtitle="Generated by training/compare.py from both evaluation runs.">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="overflow-x-auto"><table className="w-full text-sm">
          <thead className="text-left text-xs text-slate-500"><tr><th className="py-1.5 font-medium">Metric</th><th className="font-medium">Baseline</th><th className="font-medium">RDD-YOLO</th><th className="font-medium">Δ</th></tr></thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(([k, l]) => { const o = cmp.overall[k]; const d = Math.round(o.delta * 10000) / 100; return (
              <tr key={k}><td className="py-1.5">{l}</td><td className="tabular-nums">{pct(o.baseline)}</td><td className="tabular-nums">{pct(o.rdd)}</td>
                <td className={cx('tabular-nums font-medium', d > 0 ? 'text-emerald-700' : d < 0 ? 'text-red-700' : 'text-slate-500')}>{d > 0 ? '+' : ''}{(d === 0 ? 0 : d).toFixed(2)} pp</td></tr>) })}
            <tr><td className="py-1.5">Parameters</td><td className="tabular-nums">{intFmt(e.parameters.baseline)}</td><td className="tabular-nums">{intFmt(e.parameters.rdd)}</td><td className="tabular-nums">{intFmt(e.parameters.rdd - e.parameters.baseline)}</td></tr>
            <tr><td className="py-1.5">GFLOPs</td><td className="tabular-nums">{e.gflops.baseline}</td><td className="tabular-nums">{e.gflops.rdd}</td><td /></tr>
            <tr><td className="py-1.5">Model size</td><td className="tabular-nums">{e.model_size_mb.baseline} MB</td><td className="tabular-nums">{e.model_size_mb.rdd} MB</td><td /></tr>
            <tr><td className="py-1.5">FPS (batch 1, end-to-end)</td><td className="tabular-nums">{e.fps_end_to_end.baseline}</td><td className="tabular-nums">{e.fps_end_to_end.rdd}</td><td /></tr>
            <tr><td className="py-1.5">Epochs trained</td><td className="tabular-nums">{cmp.epochs.baseline}</td><td className="tabular-nums">{cmp.epochs.rdd}</td><td /></tr>
          </tbody></table></div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={2}>
            {grid}<XAxis dataKey="metric" {...axis} /><YAxis {...axis} axisLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip {...tip} formatter={(v) => pct(Number(v), 2)} cursor={{ fill: 'rgba(15,23,42,0.04)' }} /><Legend iconType="square" wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Baseline" fill={BASE} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            <Bar dataKey="RDD-YOLO" fill={RDD} radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-6 overflow-x-auto"><table className="w-full text-sm">
        <thead className="text-left text-xs text-slate-500"><tr><th className="py-1.5 font-medium">Class</th><th className="font-medium">Base mAP@50</th><th className="font-medium">RDD mAP@50</th><th className="font-medium">Base mAP@50-95</th><th className="font-medium">RDD mAP@50-95</th></tr></thead>
        <tbody className="divide-y divide-slate-100">{CLASS_CODES.map((c) => (
          <tr key={c}><td className="py-1.5">{c} · {CLASS_NAMES[c]}</td><td className="tabular-nums">{pct(cmp.per_class_mAP50[c].baseline)}</td><td className="tabular-nums">{pct(cmp.per_class_mAP50[c].rdd)}</td>
            <td className="tabular-nums">{pct(cmp.per_class_mAP50_95[c].baseline)}</td><td className="tabular-nums">{pct(cmp.per_class_mAP50_95[c].rdd)}</td></tr>))}</tbody>
      </table></div>
      {cmp.plots?.length > 0 && <div className="mt-6 grid gap-4 md:grid-cols-2">{cmp.plots.map((p: any) => <Figure key={p.name} {...p} />)}</div>}
    </Card>
  )
}

function RunDetail({ d }: { d: ExperimentDetail }) {
  const h = d.history.map((r) => ({ ...r, epoch: r.epoch }))
  const test = d.evaluations.test?.metrics
  const lines = (keys: [string, string, string][]) => keys.map(([k, n, c]) => <Line key={k} dataKey={k} name={n} stroke={c} strokeWidth={2} dot={false} isAnimationActive={false} />)
  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title={`${d.name}: loss curves`} subtitle="Training vs validation (box + classification loss)">
          {!h.length ? <EmptyState title="No epochs logged yet" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={h} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>{grid}<XAxis dataKey="epoch" {...axis} /><YAxis {...axis} axisLine={false} />
                <Tooltip {...tip} formatter={(v) => Number(v).toFixed(4)} /><Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                {lines([['train/box_loss', 'train box', '#2a78d6'], ['val/box_loss', 'val box', '#1baf7a'], ['train/cls_loss', 'train cls', '#eb6834'], ['val/cls_loss', 'val cls', '#4a3aa7']])}
              </LineChart>
            </ResponsiveContainer>)}
        </Card>
        <Card title={`${d.name}: validation metrics per epoch`}>
          {!h.length ? <EmptyState title="No epochs logged yet" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={h} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>{grid}<XAxis dataKey="epoch" {...axis} /><YAxis {...axis} axisLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip {...tip} formatter={(v) => pct(Number(v), 2)} /><Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                {lines([['metrics/mAP50(B)', 'mAP@50', '#2a78d6'], ['metrics/mAP50-95(B)', 'mAP@50-95', '#eb6834'], ['metrics/precision(B)', 'precision', '#1baf7a'], ['metrics/recall(B)', 'recall', '#4a3aa7']])}
              </LineChart>
            </ResponsiveContainer>)}
        </Card>
      </div>
      {test && (
        <Card title={`${d.name}: test-split evaluation`} subtitle={`${test.images} images · evaluated ${new Date(test.evaluated_utc).toLocaleString()} on ${test.gpu}`}>
          <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-6">
            {([['Precision', test.overall.precision], ['Recall', test.overall.recall], ['F1', test.overall.f1], ['mAP@50', test.overall.mAP50], ['mAP@50-95', test.overall.mAP50_95]] as const).map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-xs text-slate-500">{k}</div><div className="text-lg font-semibold tabular-nums">{pct(v)}</div></div>))}
            <div className="rounded-lg bg-slate-50 px-3 py-2"><div className="text-xs text-slate-500">FPS / params</div><div className="text-lg font-semibold tabular-nums">{test.speed_benchmark.fps_end_to_end.toFixed(0)} · {(test.parameters / 1e6).toFixed(2)}M</div></div>
          </div>
        </Card>
      )}
      <Card title={`${d.name}: plots`} subtitle="Ultralytics outputs (confusion matrix, PR / F1 curves, batches) from the training and test evaluation runs">
        {![...d.plots, ...(d.evaluations.test?.plots ?? [])].length ? <EmptyState title="No plots yet" /> : (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {(d.evaluations.test?.plots ?? []).map((p) => <Figure key={'t' + p.name} {...p} name={`test · ${p.name}`} />)}
            {d.plots.map((p) => <Figure key={p.name} {...p} name={`train/val · ${p.name}`} />)}
          </div>)}
      </Card>
    </div>
  )
}

function Figure({ name, url }: { name: string; url: string }) {
  return (
    <a href={asset(url)} target="_blank" rel="noreferrer" className="group block overflow-hidden rounded-lg border border-slate-200 bg-white">
      <img src={asset(url)} alt={name} loading="lazy" className="aspect-[4/3] w-full object-contain bg-white transition group-hover:opacity-90" />
      <div className="border-t border-slate-100 px-3 py-1.5 text-xs text-slate-600">{name}</div>
    </a>
  )
}
