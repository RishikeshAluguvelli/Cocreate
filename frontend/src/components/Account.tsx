import { useEffect, useRef, useState } from 'react'
import { ChevronDown, KeyRound, LogOut, UserPlus, UserRound } from 'lucide-react'
import { api } from '../api'
import type { User } from '../types'

const initialOf = (u: User) => (u.email ? u.email[0] : 'G').toUpperCase()

/** Round avatar with the account's initial (amber for accounts, grey for guests). */
export function Avatar({ user, size = 28 }: { user: User; size?: number }) {
  return (
    <span className={`grid shrink-0 place-items-center rounded-full font-semibold ${user.is_guest ? 'bg-zinc-700 text-zinc-200' : 'bg-amber-400 text-zinc-950'}`}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.45) }} aria-hidden>{initialOf(user)}</span>
  )
}

/** Inline "change password" form (accounts only). */
export function ChangePassword({ onDone }: { onDone?: () => void }) {
  const [cur, setCur] = useState(''); const [nw, setNw] = useState(''); const [msg, setMsg] = useState<string | null>(null); const [ok, setOk] = useState(false)
  const change = async (e: React.FormEvent) => {
    e.preventDefault(); setMsg(null)
    try { await api.changePassword(cur, nw); setOk(true); setMsg('Password changed. Other devices were signed out.'); setCur(''); setNw(''); onDone?.() }
    catch (err: any) { setOk(false); setMsg(String(err?.message ?? err).replace(/^\d+\s*/, '')) }
  }
  return (
    <form onSubmit={change} className="space-y-2">
      <input type="password" required placeholder="Current password" autoComplete="current-password" value={cur} onChange={e => setCur(e.target.value)} className="input !py-1.5 text-xs" />
      <input type="password" required minLength={8} placeholder="New password (8+ characters)" autoComplete="new-password" value={nw} onChange={e => setNw(e.target.value)} className="input !py-1.5 text-xs" />
      <button type="submit" className="btn btn-primary w-full justify-center !py-1.5 text-xs">Save new password</button>
      {msg && <div className={`text-[11px] ${ok ? 'text-emerald-300' : 'text-red-300'}`}>{msg}</div>}
    </form>
  )
}

/** Account pill with a small menu: change password, create an account (guests), log out. */
export function AccountMenu({ user, onUpgrade, onLogout }: { user: User; onUpgrade: () => void; onLogout: () => void }) {
  const [open, setOpen] = useState(false)
  const [changing, setChanging] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])
  return (
    <div className="relative" ref={ref}>
      <button className="btn btn-ghost !pl-1.5" onClick={() => setOpen(o => !o)} title="Account" aria-haspopup="menu" aria-expanded={open}>
        <Avatar user={user} size={22} /> <span className="max-w-[160px] truncate">{user.is_guest ? 'Guest' : user.email}</span> <ChevronDown size={12} className="text-zinc-500" />
      </button>
      {open && (
        <div role="menu" className="card absolute right-0 top-10 z-30 w-72 p-3 text-xs shadow-2xl">
          <div className="mb-2 flex items-center gap-2 text-zinc-400"><UserRound size={13} />{user.is_guest ? 'Guest workspace: private, but only reachable from this browser session.' : <span className="truncate">Signed in as <b className="text-zinc-200">{user.email}</b></span>}</div>
          {user.is_guest
            ? <button className="btn btn-primary mb-2 w-full justify-center" onClick={() => { setOpen(false); onUpgrade() }}><UserPlus size={14} /> Create an account to keep this work</button>
            : <button className="btn btn-ghost mb-2 w-full justify-center" onClick={() => setChanging(c => !c)}><KeyRound size={14} /> Change password</button>}
          {changing && !user.is_guest && <div className="mb-2"><ChangePassword /></div>}
          <button className="btn btn-ghost w-full justify-center" onClick={onLogout}><LogOut size={14} /> Log out</button>
        </div>
      )}
    </div>
  )
}
