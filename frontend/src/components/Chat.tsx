import { useEffect, useRef, useState } from 'react'
import { Bot, Brain, ChevronDown, ChevronRight, History, Loader2, MessageSquarePlus, Send, ShieldAlert, ShieldCheck, User, Wrench } from 'lucide-react'
import type { ArchivedChat, ChatMsg, Photo, ToolEvent } from '../types'
import type { PickedColor } from '../api'
import { TOOL_LABEL } from '../types'
import type { ScopeMode } from '../App'

/** the planner is asked for plain text; strip stray markdown just in case */
const md = (t: string) => t.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`([^`]+)`/g, '$1').replace(/^#+\s*/gm, '')

function QcBadge({ qc }: { qc?: { verdict: string; reason?: string } | null }) {
  if (!qc || qc.verdict === 'skipped') return null
  return qc.verdict === 'pass'
    ? <span title={qc.reason || 'QC passed'} className="inline-flex items-center gap-0.5 text-emerald-300"><ShieldCheck size={11} /> QC</span>
    : <span title={qc.reason || 'QC flagged this result'} className="inline-flex items-center gap-0.5 text-amber-300"><ShieldAlert size={11} /> check</span>
}

function ToolCard({ ev, photos }: { ev: ToolEvent & { done?: boolean }; photos: Photo[] }) {
  const [open, setOpen] = useState(false)
  const name = ev.name ?? '?'
  const inp = ev.input ?? {}
  const res = ev.result
  const target = inp.target ?? inp.instruction ?? inp.text ?? inp.region ?? (inp.hex ? `${inp.target ?? ''} → ${inp.hex}` : inp.mode ?? '')
  const results: any[] = Array.isArray(res?.results) ? res.results : res ? [res] : []
  const pname = (id?: string) => photos.find(p => p.id === id)?.name ?? id
  const nPhotos = Array.isArray(inp.photo_ids) ? (inp.photo_ids.includes('all') ? photos.length : inp.photo_ids.length) : inp.photo_id ? 1 : 0
  return (
    <div className="card mt-2 p-2 text-xs">
      <button className="flex w-full items-center gap-2 text-left" onClick={() => setOpen(o => !o)}>
        {ev.done ? <Wrench size={13} className="text-amber-300" /> : <Loader2 size={13} className="animate-spin text-amber-300" />}
        <span className="shrink-0 whitespace-nowrap font-medium">{TOOL_LABEL[name] ?? name}</span>
        <span className="truncate text-zinc-400">{typeof target === 'string' ? target : ''}</span>
        {nPhotos > 0 && !ev.done && <span className="shrink-0 rounded bg-zinc-800 px-1 text-[10px] text-zinc-400">{nPhotos} photo{nPhotos > 1 ? 's' : ''}</span>}
        {ev.error && <span className="ml-auto text-red-300">failed</span>}
        {open ? <ChevronDown size={12} className="ml-auto shrink-0" /> : <ChevronRight size={12} className="ml-auto shrink-0" />}
      </button>
      {res?.hex && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="h-5 w-5 rounded border border-zinc-600" style={{ background: res.hex }} /><span className="font-mono">{res.hex}</span><span className="text-zinc-500">from {res.sampled_from}</span>
          {res.hex_exposure_matched && <><span className="h-5 w-5 rounded border border-zinc-600" style={{ background: res.hex_exposure_matched }} /><span className="font-mono">{res.hex_exposure_matched}</span><span className="text-zinc-500">exposure-matched</span></>}
          {res.mask_preview_url && <a href={res.mask_preview_url} target="_blank" rel="noreferrer" className="text-amber-300 underline">mask</a>}
        </div>
      )}
      {results.length > 0 && results.some(r => r.result_url || r.found === false || r.error) && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {results.map((r, i) => (
            <div key={i} title={r.summary ?? r.message ?? r.error ?? ''} className={`flex items-center gap-1.5 rounded-md border px-1.5 py-1 ${r.error ? 'border-red-500/40' : r.found === false ? 'border-zinc-700 opacity-70' : 'border-emerald-500/40'}`}>
              {r.result_url && <img src={r.result_url} alt="" className="h-8 w-12 rounded object-cover" />}
              <span>{r.photo_name ?? pname(r.photo_id)}</span>
              <span className="text-zinc-500">{r.error ? 'error' : r.found === false ? 'skipped · unchanged' : r.method ? r.method.split(' ')[0] : `v${r.version_index ?? ''}`}</span>
              {typeof r.area_pct === 'number' && r.found !== false && <span className="text-zinc-600">{r.area_pct}%</span>}
              <QcBadge qc={r.qc} />
            </div>
          ))}
        </div>
      )}
      {ev.error && <div className="mt-1 text-red-300">{ev.error}</div>}
      {open && <pre className="thin-scroll mt-2 max-h-48 overflow-auto rounded bg-zinc-950 p-2 text-[10px] text-zinc-400">{JSON.stringify({ input: inp, result: res }, null, 1)}</pre>}
    </div>
  )
}

/** pair tool_start / tool_result events into cards */
function mergeEvents(events: ToolEvent[]): (ToolEvent & { done?: boolean })[] {
  const out: (ToolEvent & { done?: boolean })[] = []
  for (const e of events) {
    if (e.type === 'tool_start') out.push({ ...e, done: false })
    else if (e.type === 'tool_result') {
      const c = [...out].reverse().find(x => x.name === e.name && !x.done)
      if (c) { c.done = true; c.result = e.result; c.error = e.error } else out.push({ ...e, done: true })
    }
  }
  return out
}

function turnSummary(events: ToolEvent[]) {
  const edited = new Set<string>(); const skipped = new Set<string>(); let errors = 0; let cost = 0; let flagged = 0
  for (const e of events) {
    if (e.type !== 'tool_result') continue
    const res = e.result; const rs: any[] = Array.isArray(res?.results) ? res.results : res ? [res] : []
    for (const r of rs) {
      const key = r.photo_name ?? r.photo_id
      if (r.error) errors++
      else if (r.result_url && r.found !== false) { edited.add(key); if (r.qc?.verdict === 'needs_review') flagged++ }
      else if (r.found === false) skipped.add(key)
      cost += Number(r.cost_usd ?? 0)
    }
  }
  return { edited, skipped, errors, cost, flagged }
}

function Elapsed({ since }: { since: number }) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    const update = () => setSecs(Math.max(0, Math.round(Date.now() / 1000 - since)))
    update()
    const t = setInterval(update, 1000)
    return () => clearInterval(t)
  }, [since])
  return <span className="tabular-nums text-zinc-500">{secs}s</span>
}

function Message({ m, photos, live, status }: { m: ChatMsg & { thinking?: string }; photos: Photo[]; live?: boolean; status?: string }) {
  const [showThink, setShowThink] = useState(false)
  const tools = mergeEvents(m.events)
  const sum = m.role === 'assistant' && m.events.length ? turnSummary(m.events) : null
  const scopeTag = m.scope ? (m.scope.mode === 'photo' && m.scope.photo_name ? `only ${m.scope.photo_name}` : `whole shoot · ${m.scope.photos ?? photos.length} photos`) : null
  return (
    <div className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : ''}`}>
      {m.role === 'assistant' && <div className="mt-1 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-amber-400 text-zinc-950"><Bot size={14} /></div>}
      <div className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm ${m.role === 'user' ? 'bg-amber-400/15 text-amber-50' : 'bg-zinc-800/80'}`}>
        {m.role === 'user' && scopeTag && <div className="mb-1 text-[10px] uppercase tracking-wide text-amber-300/80">→ {scopeTag}</div>}
        {m.thinking && (
          <button onClick={() => setShowThink(s => !s)} className="mb-1 flex items-center gap-1 text-[11px] text-zinc-400">
            <Brain size={11} /> {showThink ? <ChevronDown size={11} /> : <ChevronRight size={11} />} reasoning
          </button>
        )}
        {m.thinking && showThink && <div className="mb-2 whitespace-pre-wrap border-l-2 border-zinc-700 pl-2 text-[11px] text-zinc-400">{m.thinking}</div>}
        {m.text && <div className="whitespace-pre-wrap leading-relaxed">{md(m.text)}</div>}
        {tools.map((ev, i) => <ToolCard key={i} ev={ev} photos={photos} />)}
        {live && (
          <div className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
            <Loader2 size={13} className="animate-spin text-amber-300" />
            <span className="truncate">{status || 'Working'}</span>
            <Elapsed since={m.created_at} />
          </div>
        )}
        {!live && sum && (sum.edited.size > 0 || sum.skipped.size > 0 || sum.errors > 0) && (
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-zinc-700/60 pt-1.5 text-[11px] text-zinc-400">
            {sum.edited.size > 0 && <span className="text-emerald-300">edited {sum.edited.size} photo{sum.edited.size > 1 ? 's' : ''}</span>}
            {sum.skipped.size > 0 && <span>{sum.skipped.size} unchanged</span>}
            {sum.errors > 0 && <span className="text-red-300">{sum.errors} failed</span>}
            {sum.flagged > 0 && <span className="text-amber-300">{sum.flagged} flagged by QC</span>}
            {m.seconds != null && <span>{m.seconds}s</span>}
            {sum.cost > 0 && <span>${sum.cost.toFixed(2)}</span>}
          </div>
        )}
        {!live && m.role === 'assistant' && m.status_log && m.status_log.length > 0 && !m.events.length && !m.text && <div className="text-xs text-zinc-500">(no reply)</div>}
      </div>
      {m.role === 'user' && <div className="mt-1 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-zinc-700"><User size={14} /></div>}
    </div>
  )
}

function Archived({ chats, photos }: { chats: ArchivedChat[]; photos: Photo[] }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!chats.length) return null
  return (
    <div className="space-y-1">
      {[...chats].reverse().map(c => (
        <div key={c.id} className="rounded-lg border border-zinc-800">
          <button className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs text-zinc-300" onClick={() => setOpen(o => o === c.id ? null : c.id)}>
            {open === c.id ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <History size={12} className="text-zinc-500" />
            <span>{new Date(c.created_at * 1000).toLocaleString()}</span>
            <span className="text-zinc-500">· {c.messages.length} messages</span>
          </button>
          {open === c.id && <div className="space-y-3 border-t border-zinc-800 p-2 opacity-80">{c.messages.map((m, i) => <Message key={i} m={m} photos={photos} />)}</div>}
        </div>
      ))}
    </div>
  )
}

type Props = {
  messages: ChatMsg[]; archived: ArchivedChat[]; memoryNotes: string
  live: (ChatMsg & { thinking?: string }) | null; status: string; busy: boolean; photos: Photo[]
  selected: Photo | null; pointsCount: number; paintedPct?: number | null; picked?: PickedColor | null; scope: ScopeMode; onScope: (s: ScopeMode) => void
  confirmFirst: boolean; onConfirmFirst: (v: boolean) => void; onSend: (text: string) => void; onNewChat: () => void
}

export default function Chat({ messages, archived, memoryNotes, live, status, busy, photos, selected, pointsCount, paintedPct, picked, scope, onScope, confirmFirst, onConfirmFirst, onSend, onNewChat }: Props) {
  const [text, setText] = useState('')
  const [showArchived, setShowArchived] = useState(false)
  const [showMemory, setShowMemory] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, live?.text, live?.events.length, status])

  const send = (t: string) => { if (!t.trim() || busy) return; onSend(t.trim()); setText('') }
  const chips: string[] = []
  if (selected?.dataset_request) chips.push(selected.dataset_request.length > 90 ? selected.dataset_request.slice(0, 87) + '…' : selected.dataset_request)
  chips.push('Remove the cars from every photo', 'Put the reference image on the TV', 'Turn the lamps on with a soft warm glow', 'Add a "Digitally altered" disclosure label to all photos', 'The wall colour drifted — sample it from the original exposure and repaint')
  if (pointsCount >= 3) chips.unshift('Draw the lot lines through the points I placed')
  else if (pointsCount >= 1) chips.unshift('Remove the object at the point I marked')
  if (paintedPct) chips.unshift('Also apply the last edit to the area I painted', 'Remove what I painted over')
  if (picked) chips.unshift(pointsCount >= 1 ? `Repaint the surface at the point I marked to ${picked.hex}` : `Repaint the walls to ${picked.hex}`)
  const effScope: ScopeMode = selected ? scope : 'shoot'

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2 text-xs">
        <span className="font-medium">Chat</span>
        <span className="text-zinc-500">{messages.length ? `${messages.length} messages` : 'new conversation'}</span>
        <span className="ml-auto" />
        {memoryNotes && <button className={`btn btn-ghost !px-2 !py-1 ${showMemory ? 'ring-1 ring-amber-400/60' : ''}`} title="What the assistant remembers from earlier in this conversation" onClick={() => setShowMemory(s => !s)}><Brain size={13} /> memory</button>}
        {archived.length > 0 && <button className={`btn btn-ghost !px-2 !py-1 ${showArchived ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => setShowArchived(s => !s)} title="Earlier conversations on this shoot"><History size={13} /> {archived.length}</button>}
        <button className="btn btn-ghost !px-2 !py-1" disabled={busy} onClick={onNewChat} title="Start a new conversation (photos and their versions are kept)"><MessageSquarePlus size={13} /> New chat</button>
      </div>
      <div className="thin-scroll flex-1 space-y-3 overflow-y-auto p-3">
        {showArchived && <Archived chats={archived} photos={photos} />}
        {showMemory && memoryNotes && <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-2 text-[11px] text-zinc-300"><div className="mb-1 text-zinc-500">Conversation memory (older turns folded in automatically)</div><div className="whitespace-pre-wrap">{memoryNotes}</div></div>}
        {messages.length === 0 && !live && (
          <div className="rounded-xl border border-dashed border-zinc-800 p-4 text-xs text-zinc-400">
            Describe the change like you would to an editor. With <b>Whole shoot</b> I look at every photo and edit only the ones the request applies to; with <b>Only this photo</b> nothing else changes.
            <div className="mt-2">Tips: place <span className="kbd">Points</span> on the photo for lot-line corners or to pinpoint an object · tick <b>Plan first</b> to approve before anything runs · everything is saved, so you can come back later.</div>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} photos={photos} />)}
        {live && <Message m={live} photos={photos} live status={status} />}
        <div ref={endRef} />
      </div>
      <div className="border-t border-zinc-800 p-3">
        <div className="thin-scroll mb-2 flex gap-1.5 overflow-x-auto pb-1">
          {chips.map((c, i) => <button key={i} className="chip shrink-0" onClick={() => send(selected?.dataset_request && c.startsWith(selected.dataset_request.slice(0, 20)) ? selected.dataset_request : c)}>
            {picked && c.includes(picked.hex) && <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm border border-zinc-600 align-middle" style={{ background: picked.hex }} />}{c}</button>)}
        </div>
        <div className="mb-2 flex items-center gap-2 text-[11px]">
          <span className="text-zinc-500">Applies to</span>
          <div className="flex overflow-hidden rounded-lg border border-zinc-700">
            <button onClick={() => onScope('shoot')} className={`px-2 py-1 ${effScope === 'shoot' ? 'bg-amber-400 text-zinc-950' : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-800'}`}>Whole shoot · {photos.length}</button>
            <button onClick={() => onScope('photo')} disabled={!selected} className={`px-2 py-1 ${effScope === 'photo' ? 'bg-amber-400 text-zinc-950' : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-800'} disabled:opacity-40`}>Only {selected?.name ?? 'selected'}</button>
          </div>
          <label className="ml-auto flex cursor-pointer items-center gap-1 text-zinc-400" title="The assistant describes its plan and waits for your OK before editing">
            <input type="checkbox" checked={confirmFirst} onChange={e => onConfirmFirst(e.target.checked)} className="accent-amber-400" /> Plan first
          </label>
        </div>
        <form onSubmit={e => { e.preventDefault(); send(text) }} className="flex items-end gap-2">
          <textarea value={text} onChange={e => setText(e.target.value)} rows={2}
            placeholder={effScope === 'photo' && selected ? `Change only ${selected.name}…` : 'Describe the change; I pick the photos it applies to…'}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(text) } }}
            className="thin-scroll flex-1 resize-none rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm outline-none focus:border-amber-400/60" />
          <button className="btn btn-primary h-10" disabled={busy || !text.trim()}>{busy ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}</button>
        </form>
        <div className="mt-1 text-[10px] text-zinc-500">
          {effScope === 'photo' && selected ? `Only ${selected.name} will change. Say "all photos" to override.` : `The assistant checks all ${photos.length} photos and skips the ones the request does not apply to.`}
        </div>
      </div>
    </div>
  )
}
