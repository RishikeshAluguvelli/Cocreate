import { Loader2, ShieldAlert } from 'lucide-react'
import type { Photo } from '../types'
import { headOf } from '../types'
import type { Touched } from '../App'

export default function Gallery({ photos, selectedId, onSelect, activity, touched }:
  { photos: Photo[]; selectedId: string | null; onSelect: (id: string) => void; activity: Record<string, string>; touched: Touched }) {
  return (
    <div className="thin-scroll flex h-full flex-col gap-2 overflow-y-auto p-3">
      {photos.map((p, i) => {
        const h = headOf(p)
        const sel = p.id === selectedId
        const t = touched[p.id]
        const qc = h.params?.qc
        return (
          <button key={p.id} onClick={() => onSelect(p.id)}
            className={`card relative overflow-hidden text-left transition ${sel ? 'border-amber-400 ring-1 ring-amber-400/50' : t === 'edited' ? 'border-emerald-500/60' : 'hover:border-zinc-600'}`}>
            <div className="aspect-[3/2] bg-zinc-800">
              <img src={h.url} alt={p.name} className={`h-full w-full object-cover ${t === 'skipped' ? 'opacity-60' : ''}`} loading="lazy" />
            </div>
            <div className="flex items-center justify-between gap-1 px-2 py-1.5 text-xs">
              <span className="truncate">{i + 1}. {p.name}</span>
              <span className="flex shrink-0 items-center gap-1">
                {qc?.verdict === 'needs_review' && <ShieldAlert size={12} className="text-amber-300" />}
                {h.index > 0
                  ? <span className="rounded-full bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-300">v{h.index}</span>
                  : <span className="text-[10px] text-zinc-500">original</span>}
              </span>
            </div>
            {t && !activity[p.id] && (
              <span className={`absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] ${t === 'edited' ? 'bg-emerald-500/90 text-zinc-950' : 'bg-zinc-900/85 text-zinc-300'}`}>{t === 'edited' ? 'edited' : 'unchanged'}</span>
            )}
            {activity[p.id] && (
              <div className="absolute inset-x-0 top-0 flex items-center gap-1 bg-black/70 px-2 py-1 text-[11px] text-amber-200">
                <Loader2 size={12} className="animate-spin" /><span className="truncate">{activity[p.id]}</span>
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}
