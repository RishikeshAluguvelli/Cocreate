import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowLeft, Download, FileText, ImagePlus, Images, Loader2, LogIn, Sparkles, UserPlus } from 'lucide-react'
import { api, chatStream, setOnUnauthorized, type PaintMask, type PickedColor } from './api'
import type { AuthState, ChatMsg, Health, Photo, Shoot, User } from './types'
import Auth from './components/Auth'
import Home from './components/Home'
import Gallery from './components/Gallery'
import Viewer from './components/Viewer'
import Chat from './components/Chat'
import { AccountMenu } from './components/Account'

type Live = ChatMsg & { thinking?: string }
export type Touched = Record<string, 'edited' | 'skipped'>
export type ScopeMode = 'shoot' | 'photo'

/** Everything about one in-flight request, keyed by shoot id. A request started in shoot A keeps streaming into
 *  jobs[A] even if the user opens shoot B meanwhile, so nothing from A can ever render inside B. */
type Job = { pendingUser: ChatMsg | null; live: Live | null; status: string; activity: Record<string, string>; touched: Touched; running: boolean }
const IDLE: Job = { pendingUser: null, live: null, status: '', activity: {}, touched: {}, running: false }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [auth, setAuth] = useState<AuthState | null>(null)          // null = still checking
  const [authNeeded, setAuthNeeded] = useState(false)                // accounts required and nobody is logged in (or a 401 came back)
  const [upgrading, setUpgrading] = useState(false)                  // guest -> real account
  const [resetToken, setResetToken] = useState<string | null>(() => location.hash.startsWith('#reset=') ? location.hash.slice(7) : null)
  const [shoot, setShoot] = useState<Shoot | null>(null)
  const shootIdRef = useRef<string | null>(null)                     // the shoot on screen, readable from async handlers
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [points, setPoints] = useState<number[][]>([])
  const [pointsMode, setPointsMode] = useState(false)
  const [paint, setPaint] = useState<PaintMask | null>(null)
  const [paintMode, setPaintMode] = useState(false)
  const [colorMode, setColorMode] = useState(false)                   // eyedropper / one-click recolour
  const [picked, setPicked] = useState<PickedColor | null>(null)       // the colour on the palette, kept across photos and shoots
  const [authStart, setAuthStart] = useState<'register' | 'login'>('register')
  const [jobs, setJobs] = useState<Record<string, Job>>({})
  const [scope, setScope] = useState<ScopeMode>(() => (localStorage.getItem('cc.scope') === 'photo' ? 'photo' : 'shoot'))
  const [confirmFirst, setConfirmFirst] = useState(() => localStorage.getItem('cc.confirm') === '1')
  const [toast, setToast] = useState<string | null>(null)
  const logoRef = useRef<HTMLInputElement>(null)
  const addRef = useRef<HTMLInputElement>(null)

  useEffect(() => { shootIdRef.current = shoot?.id ?? null }, [shoot?.id])
  const isCurrent = (sid: string) => shootIdRef.current === sid
  const patchJob = (sid: string, fn: (j: Job) => Job) => setJobs(js => ({ ...js, [sid]: fn(js[sid] ?? IDLE) }))

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth(null)) }, [])
  useEffect(() => {
    api.me().then(a => { setAuth(a); setAuthNeeded(a.auth_mode === 'required' && !a.user) }).catch(() => setAuth({ auth_mode: 'open', user: null }))
    setOnUnauthorized(() => { setAuthNeeded(true); setShoot(null) })
  }, [])
  const onUser = (u: User) => { setAuth(a => ({ auth_mode: a?.auth_mode ?? 'required', user: u })); setAuthNeeded(false); setUpgrading(false); setResetToken(null) }
  const logout = async () => {
    await api.logout().catch(() => {})
    setShoot(null); setJobs({}); location.hash = ''
    setAuth(a => a ? { ...a, user: null } : a)
    if (auth?.auth_mode === 'required') setAuthNeeded(true)
  }
  useEffect(() => {
    const fromHash = () => {
      const id = location.hash.replace('#', '')
      if (id.startsWith('reset=')) { setResetToken(id.slice(6)); return }
      if (id.startsWith('s_')) api.getShoot(id).then(s => { setShoot(cur => cur?.id === s.id ? cur : s); setSelectedId(sel => (sel && s.photos.some(p => p.id === sel)) ? sel : (s.photos[0]?.id ?? null)) }).catch(() => { location.hash = '' })
      else setShoot(null)
    }
    fromHash()
    window.addEventListener('hashchange', fromHash)
    return () => window.removeEventListener('hashchange', fromHash)
  }, [])
  useEffect(() => { localStorage.setItem('cc.scope', scope) }, [scope])
  useEffect(() => { localStorage.setItem('cc.confirm', confirmFirst ? '1' : '0') }, [confirmFirst])

  const open = (s: Shoot) => { setShoot(s); setSelectedId(s.photos[0]?.id ?? null); setPoints([]); setPaint(null); setPaintMode(false); setColorMode(false); location.hash = s.id }
  const refresh = useCallback(async () => { if (shoot) setShoot(await api.getShoot(shoot.id)) }, [shoot])
  const patchPhoto = (p: Photo) => setShoot(s => s ? { ...s, photos: s.photos.map(x => x.id === p.id ? p : x) } : s)
  const selected = shoot?.photos.find(p => p.id === selectedId) ?? null
  const flash = (m: string, ms = 6000) => { setToast(m); setTimeout(() => setToast(null), ms) }
  const job: Job = shoot ? (jobs[shoot.id] ?? IDLE) : IDLE
  const runningIds = Object.entries(jobs).filter(([, j]) => j.running).map(([id]) => id)
  const otherRunning = runningIds.filter(id => id !== shoot?.id).length

  const send = async (text: string) => {
    if (!shoot) return
    const sid = shoot.id
    if ((jobs[sid] ?? IDLE).running) { flash('This shoot is still working on the previous request.'); return }
    const effScope: ScopeMode = selected ? scope : 'shoot'
    const userMsg: ChatMsg = { role: 'user', text, created_at: Date.now() / 1000, events: [], scope: { mode: effScope, photo_name: selected?.name ?? null, photos: shoot.photos.length } }
    const l: Live = { role: 'assistant', text: '', created_at: Date.now() / 1000, events: [], thinking: '' }
    patchJob(sid, () => ({ ...IDLE, pendingUser: userMsg, live: { ...l }, status: 'Sending request', running: true }))
    const sentPoints = points
    const sentPaint = paint && paint.photo_id === selectedId ? paint : null
    setPoints([]); setPointsMode(false); setPaint(null); setPaintMode(false)
    // the server appends both chat messages when the turn ends; until then the job holds them for display
    const sync = async () => { const s = await api.getShoot(sid); if (isCurrent(sid)) setShoot(s) }
    try {
      for await (const ev of chatStream(sid, { text, selected_photo_id: selectedId, points: sentPoints.length ? sentPoints : null, mask: sentPaint, picked_color: picked, scope: effScope, confirm_first: confirmFirst })) {
        if (ev.type === 'text') { l.text += ev.data.delta; patchJob(sid, j => ({ ...j, live: { ...l } })) }
        else if (ev.type === 'thinking') { l.thinking = (l.thinking ?? '') + ev.data.delta; patchJob(sid, j => ({ ...j, live: { ...l } })) }
        else if (ev.type === 'status') { patchJob(sid, j => ({ ...j, status: ev.data.message })) }
        else if (ev.type === 'progress') { patchJob(sid, j => ({ ...j, status: ev.data.message, activity: { ...j.activity, [ev.data.photo_id]: ev.data.message } })) }
        else if (ev.type === 'tool_start') { l.events = [...l.events, { type: 'tool_start', name: ev.data.name, input: ev.data.input }]; patchJob(sid, j => ({ ...j, live: { ...l }, status: `Running ${String(ev.data.name).replace('_', ' ')}` })) }
        else if (ev.type === 'tool_result') { l.events = [...l.events, { type: 'tool_result', name: ev.data.name, result: ev.data.result, error: ev.data.error }]; patchJob(sid, j => ({ ...j, live: { ...l }, activity: {} })) }
        else if (ev.type === 'version') {
          const pid = ev.data.photo_id as string
          patchJob(sid, j => { const a = { ...j.activity }; delete a[pid]; return { ...j, activity: a, touched: { ...j.touched, [pid]: 'edited' } } })
          await sync()
        }
        else if (ev.type === 'skipped') { const pid = ev.data.photo_id as string; patchJob(sid, j => { const a = { ...j.activity }; delete a[pid]; return { ...j, activity: a, touched: j.touched[pid] ? j.touched : { ...j.touched, [pid]: 'skipped' } } }) }
        else if (ev.type === 'error') { if (isCurrent(sid)) flash(ev.data.message, 8000) }
        else if (ev.type === 'done') { l.text = ev.data.text || l.text; l.seconds = ev.data.seconds; patchJob(sid, j => ({ ...j, live: { ...l }, status: '' })) }
      }
    } catch (e: any) {
      if (isCurrent(sid)) flash(String(e?.message ?? e), 8000)
    } finally {
      const s = await api.getShoot(sid).catch(() => null)
      if (s && isCurrent(sid)) setShoot(s)
      // keep `touched` so the edited / skipped badges stay until the next request on this shoot
      patchJob(sid, j => ({ ...j, running: false, live: null, pendingUser: null, status: '', activity: {} }))
    }
  }

  const newChat = async () => {
    if (!shoot) return
    if (job.running) { flash('Wait for the current request to finish before starting a new chat.'); return }
    const s = await api.newChat(shoot.id)
    setShoot(s); patchJob(shoot.id, () => IDLE)
    flash(s.archived_chats.length ? 'New chat started. The previous conversation is kept under “Earlier chats”.' : 'New chat started.')
  }
  const addPhotos = async (files: FileList | null) => {
    if (!shoot || !files?.length) return
    const sid = shoot.id
    try { const s = await api.addPhotos(sid, Array.from(files)); if (isCurrent(sid)) { setShoot(s); flash(`Added ${files.length} photo${files.length > 1 ? 's' : ''} to the shoot. Requests now cover ${s.photos.length} photos.`) } }
    catch (e: any) { flash(String(e?.message ?? e)) }
  }

  if (auth === null) return <div className="grid h-full place-items-center text-sm text-zinc-500">Loading…</div>
  // the key remounts the form when a reset link arrives as a hash change while the login form is already on screen
  if (resetToken) return <Auth key="reset" onUser={onUser} resetToken={resetToken} />
  if (authNeeded || upgrading) return <Auth key={upgrading ? 'upgrade-' + authStart : 'auth'} onUser={onUser} upgrading={upgrading} initialMode={upgrading ? authStart : undefined} onCancel={() => setUpgrading(false)} />
  const guest = auth.auth_mode === 'guest' && !!auth.user?.is_guest
  const account = auth.user ? (
    <div className="flex items-center gap-2">
      {guest && <button className="btn btn-primary" onClick={() => { setAuthStart('register'); setUpgrading(true) }} title="Keep this workspace on any device"><UserPlus size={14} /> Sign up</button>}
      {guest && <button className="btn btn-ghost" onClick={() => { setAuthStart('login'); setUpgrading(true) }}><LogIn size={14} /> Log in</button>}
      <AccountMenu user={auth.user} onUpgrade={() => { setAuthStart('register'); setUpgrading(true) }} onLogout={logout} />
    </div>
  ) : null
  if (!shoot) return <Home onOpen={open} health={health} auth={auth} running={runningIds} onUser={onUser} onLogout={logout} />

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-3 py-2 text-sm">
        <button className="btn btn-ghost" onClick={() => { setShoot(null); location.hash = '' }}><ArrowLeft size={14} /> Shoots</button>
        <span className="flex items-center gap-1.5 font-medium"><Sparkles size={14} className="text-amber-300" />{shoot.name}</span>
        <span className="text-xs text-zinc-500">{shoot.photos.length} photos · est. ${shoot.total_cost_usd.toFixed(2)} in model calls · saved automatically</span>
        {otherRunning > 0 && <span className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-200"><Loader2 size={10} className="animate-spin" /> {otherRunning} other shoot{otherRunning > 1 ? 's' : ''} still working</span>}
        <span className="ml-auto" />
        {health && <span className={`rounded-full px-2 py-0.5 text-[10px] ${health.planner && health.image_models ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-200'}`}>
          {health.planner ? health.planner_model : 'keyword planner'} · {health.image_models ? `${health.edit_model} via ${health.image_backend}` : 'mock models'}</span>}
        <button className="btn btn-ghost" disabled={job.running} onClick={() => addRef.current?.click()} title="Add more finished photos to this shoot"><Images size={14} /> Add photos</button>
        <input ref={addRef} type="file" accept="image/*" multiple hidden onChange={e => { addPhotos(e.target.files); e.target.value = '' }} />
        <button className="btn btn-ghost" onClick={() => logoRef.current?.click()} title="Upload a logo for watermarks"><ImagePlus size={14} /> {shoot.has_logo ? 'Logo ✓' : 'Logo'}</button>
        <input ref={logoRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={async e => { const f = e.target.files?.[0]; if (f) { await api.uploadLogo(shoot.id, f); refresh() } }} />
        <a className="btn btn-ghost" href={`/api/shoots/${shoot.id}/report.html`} target="_blank" rel="noreferrer" title="Self-contained HTML report: every request, edit, QC verdict and before/after"><FileText size={14} /> Report</a>
        <a className="btn btn-ghost" href={`/api/shoots/${shoot.id}/download.zip`}><Download size={14} /> All</a>
        {account}
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[200px_minmax(0,1fr)_420px]">
        <aside className="min-h-0 border-r border-zinc-800">
          <Gallery photos={shoot.photos} selectedId={selectedId} onSelect={id => { setSelectedId(id); setPoints([]) }} activity={job.activity} touched={job.touched} />
        </aside>
        <main className="min-h-0">
          {selected && <Viewer key={shoot.id} shootId={shoot.id} photo={selected} points={points} pointsMode={pointsMode} busy={job.running} onPointsMode={setPointsMode} onPoints={setPoints} onPhoto={patchPhoto}
            paintMode={paintMode} onPaintMode={setPaintMode} paint={paint} onPaint={setPaint}
            colorMode={colorMode} onColorMode={setColorMode} picked={picked} onPicked={setPicked}
            onSend={t => { setScope('photo'); send(t) }} />}
        </main>
        <aside className="min-h-0 border-l border-zinc-800">
          <Chat key={shoot.id} messages={job.pendingUser ? [...shoot.chat, job.pendingUser] : shoot.chat} archived={shoot.archived_chats} memoryNotes={shoot.memory_notes}
            live={job.live} status={job.status} busy={job.running}
            photos={shoot.photos} selected={selected} pointsCount={points.length} paintedPct={paint && paint.photo_id === selectedId ? paint.area : null} picked={picked} scope={scope} onScope={setScope}
            confirmFirst={confirmFirst} onConfirmFirst={setConfirmFirst} onSend={send} onNewChat={newChat} />
        </aside>
      </div>
      {toast && <div className="fixed bottom-4 left-1/2 -translate-x-1/2 rounded-lg border border-amber-500/40 bg-zinc-900 px-4 py-2 text-sm text-amber-100 shadow-xl">{toast}</div>}
    </div>
  )
}
