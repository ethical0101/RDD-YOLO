import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { useApi } from '../lib/api'
import { STATIC } from '../lib/staticMode'
import { Button, Card, EmptyState, ErrorState, Notice, PageHeader, Spinner, StatCard, cx } from '../components/ui'

interface Item { path: string; size_gb: number; kind: string; removable: boolean; reason: string }
interface Report {
  generated_utc: string; project: string; drive: { total_gb: number; used_gb: number; free_gb: number }
  reserve_gb: number; dataset_budget_gb: number; dataset_raw_gb: number; items: Item[]; removable_total_gb: number
}

export default function Storage() {
  const [refresh, setRefresh] = useState(0)
  const { data, error, loading } = useApi<Report>(STATIC ? null : '/storage/report', refresh ? { refresh: true } : undefined, [refresh])
  if (STATIC) {
    return (
      <>
        <PageHeader title="Storage" />
        <EmptyState title="Only in the local app">Storage monitoring reads the project drive, which the browser demo cannot access.</EmptyState>
      </>
    )
  }
  return (
    <>
      <PageHeader title="Storage" subtitle="Live measurement of the project drive and of every dataset, checkpoint and cache folder."
        actions={<Button variant="secondary" onClick={() => setRefresh((x) => x + 1)} disabled={loading}><RefreshCw className={cx('h-4 w-4', loading && 'animate-spin')} /> Rescan</Button>} />
      {loading && !data ? <Spinner label="Scanning folders…" /> : error ? <ErrorState message={error} /> : data && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Free on project drive" value={`${data.drive.free_gb} GB`} hint={`of ${data.drive.total_gb} GB`} />
            <StatCard label="Download reserve" value={`${data.reserve_gb} GB`} hint="downloads stop before crossing it" />
            <StatCard label="Raw datasets" value={`${data.dataset_raw_gb} GB`} hint={`budget ${data.dataset_budget_gb} GB`} />
            <StatCard label="Regenerable" value={`${data.removable_total_gb} GB`} hint="caches and build outputs" />
          </div>
          {data.drive.free_gb < data.reserve_gb + 5 && <Notice tone="warn">Free space is close to the download reserve; new dataset downloads will be refused.</Notice>}
          <Card title="Drive usage">
            <div className="h-3 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full bg-slate-800" style={{ width: `${(100 * data.drive.used_gb) / data.drive.total_gb}%` }} />
            </div>
            <div className="mt-2 text-xs text-slate-500">{data.drive.used_gb} GB used · measured {new Date(data.generated_utc).toLocaleString()}</div>
          </Card>
          <Card title="Inventory" bodyClass="p-0">
            <div className="overflow-x-auto"><table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr>{['Path', 'Size', 'Class', 'Removable', 'Why'].map((h) => <th key={h} className="px-4 py-2 font-medium first:pl-5">{h}</th>)}</tr></thead>
              <tbody className="divide-y divide-slate-100">{[...data.items].sort((a, b) => b.size_gb - a.size_gb).map((i) => (
                <tr key={i.path}>
                  <td className="py-2 pl-5 pr-4 font-mono text-xs">{i.path}</td>
                  <td className="whitespace-nowrap px-4 py-2 tabular-nums">{i.size_gb.toFixed(2)} GB</td>
                  <td className="px-4 py-2 text-xs">{i.kind}</td>
                  <td className="px-4 py-2 text-xs">{i.removable ? 'quarantine only' : 'no'}</td>
                  <td className="px-4 py-2 text-xs text-slate-600">{i.reason}</td>
                </tr>))}</tbody>
            </table></div>
          </Card>
          <Notice>Nothing is deleted from this page. Regenerable items can be moved to <code>_quarantine/</code> with <code>scripts/storage_report.py --quarantine &lt;path&gt; --apply</code>. Source datasets, final checkpoints, results and documents are never eligible. Every check is logged in <code>docs/storage/storage_log.jsonl</code>.</Notice>
        </div>
      )}
    </>
  )
}
