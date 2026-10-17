import { ExternalLink } from 'lucide-react'
import { useApi } from '../lib/api'
import { intFmt } from '../lib/format'
import { Card, EmptyState, ErrorState, Notice, PageHeader, Spinner, cx } from '../components/ui'

const SPLITS = ['train', 'val', 'test', 'test_clean', 'test_rdd_new', 'test_external', 'test_closeup', 'test_pavement', 'test_rome']

interface Entry {
  id: string; name: string; source_url: string; license: string; attribution: string; publication: string
  original_classes: string; mapping: string; used_in: string; limitations: string; status: string
  local_images: number; local_bytes: number; downloaded: boolean; images_in_training_set_v3: Record<string, number>
}

interface Catalog {
  available: boolean; message?: string; generated_utc?: string; training_set?: string
  datasets?: Entry[]; excluded?: { name: string; source_url: string; reason: string }[]
}

export default function Datasets() {
  const { data, error, loading } = useApi<Catalog>('/datasets/catalog')
  if (loading) return <Spinner />
  if (error) return <ErrorState message={error} />
  if (!data?.available) return <EmptyState title="Dataset manifest not generated">{data?.message}</EmptyState>
  const ds = data.datasets ?? []
  const total = (s: string) => ds.reduce((a, e) => a + (e.images_in_training_set_v3[s] ?? 0), 0)
  return (
    <>
      <PageHeader title="Datasets" subtitle={<>Generated {new Date(data.generated_utc!).toLocaleString()} from <code>dataset/registry.py</code> and a scan of the files on disk. Image counts are measured, not typed in.</>} />
      <div className="space-y-6">
        <Card title="Images per split in the training set" subtitle={`${data.training_set} (Experiments E and F), after class mapping, de-duplication and leakage removal`} bodyClass="p-0">
          <div className="overflow-x-auto"><table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr>
              <th className="py-2 pl-5 pr-3 font-medium">Dataset</th><th className="px-3 py-2 font-medium">Status</th><th className="px-3 py-2 font-medium">License</th>
              <th className="px-3 py-2 text-right font-medium">On disk</th>
              {SPLITS.map((s) => <th key={s} className="px-3 py-2 text-right font-medium">{s}</th>)}
            </tr></thead>
            <tbody className="divide-y divide-slate-100">
              {ds.map((e) => (
                <tr key={e.id}>
                  <td className="min-w-[16rem] py-2 pl-5 pr-3"><a href={e.source_url} target="_blank" rel="noreferrer" className="font-medium hover:underline">{e.name}</a>
                    <div className="text-xs text-slate-500">Used in: {e.used_in}</div></td>
                  <td className="px-3 py-2"><StatusBadge s={e.status} /></td>
                  <td className="px-3 py-2 text-xs">{e.license}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{intFmt(e.local_images)}<div className="text-xs text-slate-500">{(e.local_bytes / 2 ** 30).toFixed(2)} GB</div></td>
                  {SPLITS.map((s) => <td key={s} className="px-3 py-2 text-right tabular-nums">{e.images_in_training_set_v3[s] ? intFmt(e.images_in_training_set_v3[s]) : <span className="text-slate-300">–</span>}</td>)}
                </tr>))}
              <tr className="bg-slate-50 font-semibold"><td className="py-2 pl-5" colSpan={4}>Total</td>
                {SPLITS.map((s) => <td key={s} className="px-3 py-2 text-right tabular-nums">{intFmt(total(s))}</td>)}</tr>
            </tbody>
          </table></div>
        </Card>
        <div className="grid gap-4 lg:grid-cols-2">
          {ds.map((e) => (
            <Card key={e.id} title={e.name} actions={<a href={e.source_url} target="_blank" rel="noreferrer" className="text-slate-500 hover:text-slate-900" aria-label="Source"><ExternalLink className="h-4 w-4" /></a>}>
              <dl className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1.5 text-sm">
                <dt className="text-slate-500">License</dt><dd>{e.license}</dd>
                <dt className="text-slate-500">Attribution</dt><dd>{e.attribution}</dd>
                <dt className="text-slate-500">Publication</dt><dd>{e.publication}</dd>
                <dt className="text-slate-500">Classes</dt><dd>{e.original_classes}</dd>
                <dt className="text-slate-500">Mapping</dt><dd>{e.mapping}</dd>
                <dt className="text-slate-500">Limitations</dt><dd>{e.limitations}</dd>
              </dl>
            </Card>))}
        </div>
        <Card title="Reviewed and not used" bodyClass="p-0">
          <div className="overflow-x-auto"><table className="w-full text-sm"><tbody className="divide-y divide-slate-100">
            {(data.excluded ?? []).map((x) => (
              <tr key={x.name}><td className="py-2 pl-5 pr-3 font-medium"><a href={x.source_url} target="_blank" rel="noreferrer" className="hover:underline">{x.name}</a></td>
                <td className="px-3 py-2 text-slate-600">{x.reason}</td></tr>))}
          </tbody></table></div>
        </Card>
        <Notice>A dataset counts as used only when its images are present in the training set above. Datasets without a clear license, or behind a registration wall, are not downloaded.</Notice>
      </div>
    </>
  )
}

function StatusBadge({ s }: { s: string }) {
  const tone = s === 'used' ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
    : s.startsWith('downloaded') || s.startsWith('used (') ? 'bg-sky-50 text-sky-800 ring-sky-200' : 'bg-slate-50 text-slate-600 ring-slate-200'
  return <span className={cx('whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-medium ring-1', tone)}>{s}</span>
}
