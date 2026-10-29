import { useEffect, useRef, useState } from 'react'
import { ArrowRight, Brush, Clock, Columns2, Eraser, Images, KeyRound, Lamp, Loader2, LogIn, LogOut, MessageSquareText, Paintbrush, Pipette, ShieldCheck, SlidersHorizontal, Sparkles, Stamp, Trash2, Trees, Undo2, Upload, UserPlus } from 'lucide-react'
import { api } from '../api'
import type { AuthState, Health, Shoot, ShootSummary, Template, User } from '../types'
import { AuthForm, type AuthMode } from './Auth'
import { AccountMenu, Avatar, ChangePassword } from './Account'

function ago(ts: number) {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

const STEPS = [
  { n: 1, title: 'Start a shoot', body: 'Upload the finished photos of one listing, or open a demo shoot. A shoot is simply the set of photos one request should apply to.' },
  { n: 2, title: 'Ask in plain words', body: '“Remove the cars from the driveway”, “turn on the lamps”, “put the reference image on the TV”. The assistant checks every photo and edits only the ones the request applies to.' },
  { n: 3, title: 'Check, fine-tune, download', body: 'Slide before / after, see exactly which pixels changed, undo any step, nudge the sliders or pick a colour by hand, then download one photo or the whole shoot.' },
]

const CAN_DO = [
  { icon: Eraser, title: 'Remove things', eg: 'cars, bins, cords, clutter, a lockbox, people' },
  { icon: Paintbrush, title: 'Re-paint surfaces', eg: 'walls or floors to a hex colour or to a reference photo' },
  { icon: Pipette, title: 'Pick & apply colours', eg: 'eyedropper on any wall or object, one click to recolour another' },
  { icon: Lamp, title: 'Switch things on', eg: 'lamps, fan lights, the fireplace, a TV showing your image' },
  { icon: Trees, title: 'Refresh outdoors & fabrics', eg: 'patchy lawn, grey sky, twilight look, wrinkled bedding' },
  { icon: SlidersHorizontal, title: 'Adjust the look', eg: 'brightness, contrast, warmth, shadows, sharpen, match photos' },
  { icon: Stamp, title: 'Mark up', eg: 'lot lines from clicked points, a watermark or logo, blur faces & plates' },
  { icon: Brush, title: 'Fix a missed spot', eg: 'paint the area on the photo, then say what to do there' },
]

type Props = { onOpen: (s: Shoot) => void; health: Health | null; auth: AuthState | null; running?: string[]; onUser: (u: User) => void; onLogout: () => void }

/** The account card in the hero: sign in / sign up for guests, the account summary for members, quick start in open mode. */
function WorkspaceCard({ auth, recent, tab, onTab, onUser, onLogout, onOpen, flash }:
  { auth: AuthState | null; recent: ShootSummary[]; tab: AuthMode; onTab: (m: AuthMode) => void; onUser: (u: User) => void; onLogout: () => void; onOpen: (s: Shoot) => void; flash: boolean }) {
  const [changing, setChanging] = useState(false)
  const user = auth?.user
  const mode = auth?.auth_mode ?? 'open'
  const ring = flash ? 'ring-2 ring-amber-400/80 shadow-[0_0_0_6px_rgba(251,191,36,0.12)]' : ''
  if (mode === 'open' || !user) {
    return (
      <div className={`card p-5 transition ${ring}`}>
        <div className="eyebrow">Quick start</div>
        <ol className="mt-3 space-y-3">
          {STEPS.map(s => (
            <li key={s.n} className="flex gap-3 text-xs"><span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-amber-400 text-[11px] font-semibold text-zinc-950">{s.n}</span><span><b className="text-zinc-200">{s.title}.</b> <span className="text-zinc-400">{s.body}</span></span></li>
          ))}
        </ol>
        <p className="mt-4 text-[11px] text-zinc-500">This server runs in single-user mode: accounts are off and every shoot is visible here. Set <code className="kbd">AUTH_MODE=guest</code> to give each visitor a private workspace.</p>
      </div>
    )
  }
  if (user.is_guest) {
    return (
      <div className={`card p-5 transition ${ring}`} data-testid="workspace-card">
        <div className="flex items-center justify-between">
          <div className="eyebrow">Your workspace</div>
          <span className="flex items-center gap-1.5 rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-300"><Avatar user={user} size={14} /> Guest</span>
        </div>
        <h2 className="mt-2 text-lg font-semibold leading-tight">{tab === 'login' ? 'Welcome back' : tab === 'forgot' ? 'Reset your password' : 'Keep your work on any device'}</h2>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          You are editing in a private guest workspace that lives in this browser. {tab === 'login' ? 'Log in to open the shoots of your account.' : 'Add an email and password to keep these shoots and open them anywhere.'}
        </p>
        {tab !== 'forgot' && (
          <div className="mt-4 grid grid-cols-2 rounded-lg border border-zinc-700 bg-zinc-900 p-0.5 text-xs" role="tablist">
            <button role="tab" aria-selected={tab === 'register'} className={`rounded-md py-1.5 transition ${tab === 'register' ? 'bg-amber-400 font-medium text-zinc-950' : 'text-zinc-300 hover:text-zinc-100'}`} onClick={() => onTab('register')}>Create account</button>
            <button role="tab" aria-selected={tab === 'login'} className={`rounded-md py-1.5 transition ${tab === 'login' ? 'bg-amber-400 font-medium text-zinc-950' : 'text-zinc-300 hover:text-zinc-100'}`} onClick={() => onTab('login')}>Log in</button>
          </div>
        )}
        <div className="mt-4"><AuthForm compact upgrading={tab === 'register'} mode={tab} onMode={onTab} onUser={onUser} /></div>
        <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">
          {tab === 'register' && 'Free, email and password only. Everything in this guest workspace moves into your new account.'}
          {tab === 'login' && 'Logging in switches to that account; the shoots of this guest session stay with the guest session.'}
        </p>
      </div>
    )
  }
  const edits = recent.reduce((n, s) => n + s.edits, 0)
  return (
    <div className={`card p-5 transition ${ring}`} data-testid="workspace-card">
      <div className="eyebrow">Your workspace</div>
      <div className="mt-3 flex items-center gap-3">
        <Avatar user={user} size={40} />
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{user.email}</div>
          <div className="text-xs text-zinc-500">{recent.length} shoot{recent.length === 1 ? '' : 's'} · {edits} edit{edits === 1 ? '' : 's'}{recent[0] ? ` · last active ${ago(recent[0].updated_at)}` : ''}</div>
        </div>
      </div>
      {recent[0]
        ? <button className="btn btn-primary mt-4 w-full justify-center" onClick={async () => onOpen(await api.getShoot(recent[0].id))}><span className="truncate">Continue “{recent[0].name}”</span> <ArrowRight size={14} className="shrink-0" /></button>
        : <p className="mt-4 rounded-lg border border-dashed border-zinc-700 p-3 text-center text-xs text-zinc-400">No shoots yet. Upload photos or open a demo below.</p>}
      <div className="mt-3 flex gap-2">
        <button className="btn btn-ghost flex-1 justify-center" onClick={() => setChanging(c => !c)}><KeyRound size={14} /> Password</button>
        <button className="btn btn-ghost flex-1 justify-center" onClick={onLogout}><LogOut size={14} /> Log out</button>
      </div>
      {changing && <div className="mt-3 rounded-lg border border-zinc-800 p-3"><ChangePassword onDone={() => setChanging(false)} /></div>}
      <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">Shoots, chats and every version are private to this account and saved automatically.</p>
    </div>
  )
}

export default function Home({ onOpen, health, auth, running = [], onUser, onLogout }: Props) {
  const [templates, setTemplates] = useState<Template[]>([])
  const [recent, setRecent] = useState<ShootSummary[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [tab, setTab] = useState<AuthMode>('register')
  const [flash, setFlash] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const demosRef = useRef<HTMLDivElement>(null)

  const load = () => { api.templates().then(setTemplates).catch(() => {}); api.listShoots().then(setRecent).catch(() => setRecent([])) }
  useEffect(load, [auth?.user?.id])

  const openDemo = async (id: string) => {
    setBusy(id); setError(null)
    try { onOpen(await api.createDemo(id)) }
    catch (e: any) { setError(String(e?.message ?? e)) }
    finally { setBusy(null) }
  }
  const upload = async (files: FileList | File[] | null) => {
    const list = Array.from(files ?? []).filter(f => f.type.startsWith('image/'))
    if (!list.length) { setError('Drop JPEG or PNG photos.'); return }
    setBusy('upload'); setError(null)
    try {
      const f = new FormData()
      f.append('name', list.length === 1 ? list[0].name.replace(/\.[^.]+$/, '') : `Upload · ${list.length} photos`)
      list.forEach(x => f.append('files', x))
      onOpen(await api.createShoot(f))
    } catch (e: any) { setError(String(e?.message ?? e)) }
    finally { setBusy(null) }
  }
  const focusCard = (m: AuthMode) => {
    setTab(m)
    cardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setFlash(true); setTimeout(() => setFlash(false), 1600)
  }
  const starter = templates.find(t => t.id === 'sampler') ?? templates[0]
  const user = auth?.user ?? null
  const guest = auth?.auth_mode === 'guest' && !!user?.is_guest
  const recentSorted = [...recent].sort((a, b) => Number(running.includes(b.id)) - Number(running.includes(a.id)))

  return (
    <div className="hero-bg min-h-full">
      {/* top bar */}
      <header className="sticky top-0 z-20 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-2 px-6">
          <a href="#top" className="flex items-center gap-2 text-sm font-semibold tracking-tight"><Sparkles size={18} className="text-amber-300" /> Cocreate</a>
          <nav className="ml-4 hidden items-center gap-0.5 md:flex" aria-label="Sections">
            <a className="nav-link" href="#how">How it works</a><a className="nav-link" href="#start">Start</a><a className="nav-link" href="#demos">Demos</a><a className="nav-link" href="#ask">What you can ask</a>
          </nav>
          <span className="ml-auto" />
          {health && (
            <span title={health.planner ? `${health.planner_model} plans · ${health.edit_model} edits` : ''} className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] lg:inline-flex ${health.planner && health.image_models ? 'bg-emerald-500/10 text-emerald-300' : 'bg-amber-500/15 text-amber-200'}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${health.planner && health.image_models ? 'bg-emerald-400' : 'bg-amber-300'}`} />
              {health.planner ? 'Models online' : health.mock ? 'Mock mode' : 'No model key'}
            </span>
          )}
          {guest && <>
            <button className="btn btn-ghost" onClick={() => focusCard('login')}><LogIn size={14} /> Log in</button>
            <button className="btn btn-primary" onClick={() => focusCard('register')}><UserPlus size={14} /> Sign up free</button>
          </>}
          {user && !guest && <AccountMenu user={user} onUpgrade={() => focusCard('register')} onLogout={onLogout} />}
        </div>
      </header>

      <div id="top" className="mx-auto max-w-6xl px-6 pb-16">
        {/* hero */}
        <section className="grid items-start gap-8 pt-12 lg:grid-cols-[minmax(0,1fr)_400px] lg:pt-16">
          <div className="fade-up">
            <div className="eyebrow">AutoHDR hackathon · Track 3 · conversational image editing</div>
            <h1 className="mt-3 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">Edit a whole real-estate photo shoot<br className="hidden sm:block" /> by <span className="text-amber-300">chatting</span>.</h1>
            <p className="mt-4 max-w-xl text-base leading-relaxed text-zinc-400">
              Describe the change the way you would to a photo editor. The assistant looks at every photo in the shoot, makes the change
              only where it applies, and leaves the rest of each photo untouched. Every step is saved and can be undone.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button className="btn btn-primary btn-lg" disabled={!!busy} onClick={() => fileRef.current?.click()}><Upload size={16} /> {busy === 'upload' ? 'Uploading…' : 'Upload photos'}</button>
              {starter && <button className="btn btn-ghost btn-lg" disabled={!!busy} onClick={() => openDemo(starter.id)}>{busy === starter.id ? 'Preparing…' : <>Try the sampler demo <ArrowRight size={14} /></>}</button>}
              <a href="#how" className="text-sm text-zinc-400 hover:text-amber-300">How it works ↓</a>
              <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={e => { upload(e.target.files); e.target.value = '' }} />
            </div>
            <ul className="mt-7 grid gap-3 text-xs text-zinc-400 sm:grid-cols-3">
              <li className="flex items-start gap-2"><Columns2 size={15} className="mt-0.5 shrink-0 text-amber-300" /><span><b className="text-zinc-200">Only the masked pixels change.</b> The Changes view proves the rest is identical.</span></li>
              <li className="flex items-start gap-2"><ShieldCheck size={15} className="mt-0.5 shrink-0 text-amber-300" /><span><b className="text-zinc-200">QC on every AI edit.</b> A vision check scores fulfilment, collateral and artifacts.</span></li>
              <li className="flex items-start gap-2"><Undo2 size={15} className="mt-0.5 shrink-0 text-amber-300" /><span><b className="text-zinc-200">Every version kept.</b> Undo, revert or regenerate any step.</span></li>
            </ul>
          </div>
          <div ref={cardRef} className="fade-up scroll-mt-24 lg:sticky lg:top-20" style={{ animationDelay: '80ms' }}>
            <WorkspaceCard auth={auth} recent={recent} tab={tab} onTab={setTab} onUser={onUser} onLogout={onLogout} onOpen={onOpen} flash={flash} />
          </div>
        </section>
        {error && <p role="alert" className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-xs text-red-200">{error}</p>}

        {/* returning users */}
        {recent.length > 0 && (
          <section className="mt-14">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-300"><Clock size={16} className="text-amber-300" /> Continue where you left off
              <span className="text-xs font-normal text-zinc-500">· {user && !user.is_guest ? 'saved to your account' : 'saved in this workspace'}</span>
              {recent.length > 6 && <button className="ml-auto text-xs font-normal text-amber-300 hover:underline" onClick={() => setShowAll(v => !v)}>{showAll ? 'Show fewer' : `Show all ${recent.length}`}</button>}
            </h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              {(showAll ? recentSorted : recentSorted.slice(0, 6)).map(s => (
                <div key={s.id} className="card card-hover group relative overflow-hidden">
                  <button onClick={async () => onOpen(await api.getShoot(s.id))} className="block w-full text-left">
                    <div className="relative aspect-[3/2] bg-zinc-800">{s.cover && <img src={s.cover} alt="" className="h-full w-full object-cover" loading="lazy" />}
                      {running.includes(s.id) && <span className="absolute left-1 top-1 flex items-center gap-1 rounded-full bg-amber-400 px-1.5 py-0.5 text-[10px] font-medium text-zinc-950"><Loader2 size={10} className="animate-spin" /> working</span>}
                    </div>
                    <div className="p-2 text-xs">
                      <div className="truncate">{s.name}</div>
                      <div className="mt-0.5 truncate text-[10px] text-zinc-500">{s.photos} photos · {s.edits} edit{s.edits === 1 ? '' : 's'} · {s.messages} msg{s.archived_chats ? ` +${s.archived_chats} chats` : ''} · {ago(s.updated_at)}</div>
                    </div>
                  </button>
                  <button title="Delete this shoot" onClick={async () => { await api.deleteShoot(s.id); load() }}
                    className="absolute right-1 top-1 hidden rounded bg-black/60 p-1 text-zinc-300 hover:text-red-300 group-hover:block"><Trash2 size={12} /></button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* how it works */}
        <section id="how" className="mt-16 scroll-mt-20">
          <div className="eyebrow">How it works</div>
          <h2 className="mt-1 text-xl font-semibold">Three steps, no manual masking</h2>
          <div className="mt-5 grid gap-3 md:grid-cols-3">
            {STEPS.map(s => (
              <div key={s.n} className="card card-hover flex gap-3 p-4">
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-amber-400 text-sm font-semibold text-zinc-950">{s.n}</span>
                <div>
                  <div className="text-sm font-medium">{s.title}</div>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-400">{s.body}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* start */}
        <section id="start" className="mt-16 scroll-mt-20">
          <div className="eyebrow">Start</div>
          <h2 className="mt-1 text-xl font-semibold">Your photos, or a demo built from real customer requests</h2>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <div
              onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
              onDrop={e => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files) }}
              className={`card flex flex-col justify-between border-dashed p-5 transition ${drag ? 'border-amber-400 bg-amber-400/5' : ''}`}>
              <div>
                <div className="flex items-center gap-2 text-sm font-medium"><Upload size={16} className="text-amber-300" /> Your own photos</div>
                <p className="mt-1 text-xs leading-relaxed text-zinc-400">Drop the finished JPEG or PNG photos of one listing here. Each file becomes one photo of a new shoot; you can add more from inside the shoot. Optional: upload a reference image or logo later from the editor.</p>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button className="btn btn-primary" disabled={!!busy} onClick={() => fileRef.current?.click()}><Upload size={16} /> {busy === 'upload' ? 'Uploading…' : 'Choose photos'}</button>
                <span className="text-xs text-zinc-500">or drag them onto this card</span>
              </div>
            </div>
            <div className="card flex flex-col justify-between p-5">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium"><Images size={16} className="text-amber-300" /> No photos handy? Try a demo shoot</div>
                <p className="mt-1 text-xs leading-relaxed text-zinc-400">Each demo is built from AutoHDR's real customer requests. The customer's original request appears as a chip above the chat box, so you can click it and watch the assistant work, or type anything you like.</p>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                {starter && <button className="btn btn-primary" disabled={!!busy} onClick={() => openDemo(starter.id)}>{busy === starter.id ? 'Preparing…' : <>Start here: {starter.title.split(' · ')[0]} <ArrowRight size={14} /></>}</button>}
                <button className="btn btn-ghost" onClick={() => demosRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>All demos</button>
              </div>
            </div>
          </div>
        </section>

        {/* demos */}
        <section id="demos" ref={demosRef} className="mt-16 scroll-mt-20">
          <div className="eyebrow">Demo shoots</div>
          <h2 className="mt-1 text-xl font-semibold">Real photos, real requests</h2>
          <p className="mt-1 text-xs text-zinc-500">Opening one copies the photos into a fresh shoot of your own, so you can experiment freely.</p>
          <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {templates.map(t => (
              <button key={t.id} disabled={!!busy} onClick={() => openDemo(t.id)}
                className={`card card-hover group overflow-hidden text-left disabled:opacity-60 ${t.id === starter?.id ? 'border-amber-400/40' : ''}`}>
                <div className="relative aspect-[3/2] overflow-hidden bg-zinc-800">
                  <img src={t.cover} alt="" className="h-full w-full object-cover transition group-hover:scale-105" loading="lazy" />
                  <span className="absolute right-2 top-2 rounded-full bg-black/60 px-2 py-0.5 text-[11px]">{t.count} photos</span>
                  {t.id === starter?.id && <span className="absolute left-2 top-2 rounded-full bg-amber-400 px-2 py-0.5 text-[11px] font-medium text-zinc-950">Start here</span>}
                  {busy === t.id && <span className="absolute inset-0 grid place-items-center bg-black/50 text-sm">Preparing…</span>}
                </div>
                <div className="p-3">
                  <div className="text-sm font-medium">{t.title}</div>
                  <div className="mt-1 line-clamp-2 text-xs text-zinc-400">{t.blurb}</div>
                  {t.requests[0] && <div className="mt-2 flex items-start gap-1.5 text-[11px] text-zinc-500"><MessageSquareText size={12} className="mt-0.5 shrink-0 text-amber-300/80" /><span className="line-clamp-2 italic">“{t.requests[0].replace(/\s+/g, ' ')}”</span></div>}
                </div>
              </button>
            ))}
            {!templates.length && Array.from({ length: 4 }).map((_, i) => <div key={i} className="card aspect-[3/2] animate-pulse bg-zinc-900" />)}
          </div>
        </section>

        {/* what you can ask */}
        <section id="ask" className="mt-16 scroll-mt-20">
          <div className="eyebrow">What you can ask for</div>
          <h2 className="mt-1 text-xl font-semibold">One request, the right tool per photo</h2>
          <p className="mt-1 text-xs text-zinc-500">Mix several changes in one message; the assistant picks the tool for each photo and tells you what it skipped. Tick <b>Plan first</b> in the chat to approve the plan before anything runs.</p>
          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {CAN_DO.map(c => (
              <div key={c.title} className="card card-hover flex gap-3 p-3">
                <c.icon size={16} className="mt-0.5 shrink-0 text-amber-300" />
                <div><div className="text-sm">{c.title}</div><div className="mt-0.5 text-xs text-zinc-500">{c.eg}</div></div>
              </div>
            ))}
          </div>
        </section>

        <footer className="mt-16 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-zinc-800 pt-5 text-[11px] text-zinc-500">
          <span className="flex items-center gap-1.5"><Sparkles size={12} className="text-amber-300" /> Cocreate · AutoHDR hackathon, Track 3</span>
          <span className="flex items-center gap-1.5"><Columns2 size={12} /> Every edit is pasted back only inside its mask; the <b>Changes</b> view proves the rest is untouched.</span>
          <span>Planning costs about a cent per request and each AI edit a few cents; the running total is shown in the shoot header.</span>
        </footer>
      </div>
    </div>
  )
}
