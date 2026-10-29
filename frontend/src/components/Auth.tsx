import { useState } from 'react'
import { ArrowRight, KeyRound, LogIn, Sparkles, UserPlus, UserRound } from 'lucide-react'
import { api } from '../api'
import type { User } from '../types'

export type AuthMode = 'login' | 'register' | 'forgot' | 'reset'

const clean = (e: any) => String(e?.message ?? e).replace(/^\d+\s*/, '').replace(/^\{"detail":"(.*)"\}$/, '$1')

/** The email / password form with its four modes. Used inline on the home page (account card) and on the full-page
 *  Auth screen (login wall, reset links). `upgrading` = a guest adding an email and password to keep their workspace. */
export function AuthForm({ onUser, mode, onMode, resetToken, upgrading, compact }:
  { onUser: (u: User) => void; mode: AuthMode; onMode: (m: AuthMode) => void; resetToken?: string | null; upgrading?: boolean; compact?: boolean }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [password2, setPassword2] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null); setInfo(null)
    try { await fn() } catch (e: any) { setError(clean(e)) } finally { setBusy(false) }
  }
  const go = (m: AuthMode) => { onMode(m); setError(null); setInfo(null) }
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (mode === 'login') run(async () => onUser((await api.login(email, password)).user!))
    else if (mode === 'register') run(async () => {
      if (password !== password2) throw new Error('The two passwords do not match.')
      onUser((await api.register(email, password)).user!)
    })
    else if (mode === 'forgot') run(async () => {
      const r = await api.forgot(email)
      setInfo(r.sent ? 'If that email has an account, a reset link is on its way. It is valid for one hour.'
        : r.reset_link ? `Mail is not configured on this server, so here is your reset link: ${r.reset_link}`
          : 'If that email has an account, a reset link has been generated. Mail is not configured on this server; ask the administrator for the link from the server log.')
    })
    else if (mode === 'reset') run(async () => {
      if (password !== password2) throw new Error('The two passwords do not match.')
      const r = await api.reset(resetToken!, password)
      location.hash = ''
      onUser(r.user!)
    })
  }
  const label = compact ? 'text-[11px]' : 'text-xs'
  return (
    <form onSubmit={submit} className="space-y-3">
      {mode !== 'reset' && (
        <label className={`block ${label} text-zinc-400`}>Email
          <input type="email" required autoComplete="email" placeholder="you@studio.com" value={email} onChange={e => setEmail(e.target.value)} className="input mt-1" />
        </label>
      )}
      {mode !== 'forgot' && (
        <label className={`block ${label} text-zinc-400`}>
          <span className="flex items-center justify-between">{mode === 'reset' ? 'New password' : 'Password'}
            {mode === 'login' && <button type="button" tabIndex={-1} className="text-[11px] text-zinc-500 hover:text-amber-300" onClick={() => go('forgot')}>Forgot it?</button>}
          </span>
          <input type="password" required minLength={8} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} placeholder={mode === 'login' ? '••••••••' : 'At least 8 characters'} value={password} onChange={e => setPassword(e.target.value)} className="input mt-1" />
        </label>
      )}
      {(mode === 'register' || mode === 'reset') && (
        <label className={`block ${label} text-zinc-400`}>Repeat password
          <input type="password" required minLength={8} autoComplete="new-password" value={password2} onChange={e => setPassword2(e.target.value)} className="input mt-1" />
        </label>
      )}
      {error && <p role="alert" className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">{error}</p>}
      {info && <p className="break-all rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">{info}</p>}
      <button type="submit" disabled={busy} className="btn btn-primary w-full justify-center py-2">
        {mode === 'login' ? <><LogIn size={14} /> Log in</> : mode === 'register' ? <><UserPlus size={14} /> {upgrading ? 'Create account & keep my work' : 'Create account'}</> : mode === 'forgot' ? <><KeyRound size={14} /> Send reset link</> : <><KeyRound size={14} /> Set new password</>}
      </button>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-[11px] text-zinc-500">
        {mode === 'login' && <button type="button" className="hover:text-amber-300" onClick={() => go('register')}>New here? Create an account</button>}
        {mode === 'register' && <button type="button" className="hover:text-amber-300" onClick={() => go('login')}>Already have an account? Log in</button>}
        {(mode === 'forgot' || mode === 'reset') && <button type="button" className="hover:text-amber-300" onClick={() => { go('login'); if (mode === 'reset') location.hash = '' }}>Back to log in</button>}
      </div>
    </form>
  )
}

/** Full-page account screen: the login wall in `required` mode and the landing page of a password-reset link. */
export default function Auth({ onUser, resetToken, upgrading, onCancel, initialMode }: { onUser: (u: User) => void; resetToken?: string | null; upgrading?: boolean; onCancel?: () => void; initialMode?: 'register' | 'login' }) {
  const [mode, setMode] = useState<AuthMode>(resetToken ? 'reset' : initialMode ?? (upgrading ? 'register' : 'login'))
  const [busy, setBusy] = useState(false)
  const title = { login: 'Log in', register: upgrading ? 'Create your account' : 'Create an account', forgot: 'Reset your password', reset: 'Choose a new password' }[mode]
  const guest = async () => { setBusy(true); try { onUser((await api.guest()).user!) } finally { setBusy(false) } }
  return (
    <div className="hero-bg min-h-full">
      <div className="mx-auto flex min-h-full max-w-md flex-col justify-center px-6 py-10">
        <div className="mb-6 flex items-center gap-2 text-amber-300"><Sparkles size={18} /><span className="text-xs uppercase tracking-widest">Cocreate</span></div>
        <h1 className="text-2xl font-semibold">{title}</h1>
        <p className="mt-1 text-sm text-zinc-400">
          {mode === 'login' && (upgrading ? 'Log in to an existing account. The shoots in your current guest workspace stay with the guest session.' : 'Your shoots, chats and every edited version are private to your account.')}
          {mode === 'register' && (upgrading ? 'Keep everything you made as a guest: add an email and password to this workspace.' : 'Each account has its own private workspace.')}
          {mode === 'forgot' && 'Enter the email of your account and we will send a one-hour reset link.'}
          {mode === 'reset' && 'Choose a new password. Other devices that were logged in will be signed out.'}
        </p>
        <div className="card mt-5 p-5">
          <AuthForm onUser={onUser} mode={mode} onMode={setMode} resetToken={resetToken} upgrading={upgrading} />
        </div>
        {upgrading && onCancel && <button className="mt-4 self-start text-xs text-zinc-400 hover:text-amber-300" onClick={onCancel}>← Back to my shoots</button>}
        {!upgrading && mode !== 'reset' && (
          <div className="mt-8 border-t border-zinc-800 pt-5">
            <div className="text-xs text-zinc-500">Just want to try it?</div>
            <button disabled={busy} onClick={guest} className="btn btn-ghost mt-2"><UserRound size={14} /> Continue as guest <ArrowRight size={14} /></button>
            <p className="mt-2 text-[11px] text-zinc-500">A guest workspace is private too and lives in this browser's session. You can add an email and password later to keep it.</p>
          </div>
        )}
      </div>
    </div>
  )
}
