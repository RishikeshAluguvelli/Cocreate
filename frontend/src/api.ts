import type { AuthState, Health, Photo, Shoot, ShootSummary, SseEvent, Template } from './types'

/** Called when the server answers 401 (session expired / accounts required): the app shows the login page. */
export let onUnauthorized: () => void = () => {}
export const setOnUnauthorized = (fn: () => void) => { onUnauthorized = fn }

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    if (r.status === 401) onUnauthorized()
    let msg = await r.text()
    try { msg = JSON.parse(msg).detail ?? msg } catch { /* plain text */ }
    throw new Error(`${r.status} ${msg}`)
  }
  return r.json() as Promise<T>
}
const json = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  health: () => j<Health>('/api/health'),
  templates: () => j<Template[]>('/api/demo-shoots'),
  listShoots: () => j<ShootSummary[]>('/api/shoots'),
  createDemo: (id: string) => j<Shoot>(`/api/shoots/demo/${id}`, { method: 'POST' }),
  createShoot: (form: FormData) => j<Shoot>('/api/shoots', { method: 'POST', body: form }),
  addPhotos: (sid: string, files: File[]) => { const f = new FormData(); files.forEach(x => f.append('files', x)); return j<Shoot>(`/api/shoots/${sid}/photos`, { method: 'POST', body: f }) },
  getShoot: (id: string) => j<Shoot>(`/api/shoots/${id}`),
  deleteShoot: (id: string) => j<{ ok: boolean }>(`/api/shoots/${id}`, { method: 'DELETE' }),
  newChat: (sid: string) => j<Shoot>(`/api/shoots/${sid}/chat/new`, { method: 'POST' }),
  revert: (sid: string, pid: string, version_id: string) => j<Photo>(`/api/shoots/${sid}/photos/${pid}/revert`, json({ version_id })),
  manualAdjust: (sid: string, pid: string, vals: Record<string, number>, base_version_id?: string) => j<Photo>(`/api/shoots/${sid}/photos/${pid}/adjust`, json({ ...vals, base_version_id })),
  // accounts
  me: () => j<AuthState>('/api/auth/me'),
  login: (email: string, password: string) => j<AuthState>('/api/auth/login', json({ email, password })),
  register: (email: string, password: string) => j<AuthState>('/api/auth/register', json({ email, password })),
  guest: () => j<AuthState>('/api/auth/guest', { method: 'POST' }),
  logout: () => j<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  changePassword: (current_password: string, new_password: string) => j<{ ok: boolean }>('/api/auth/change-password', json({ current_password, new_password })),
  forgot: (email: string) => j<{ ok: boolean; sent: boolean; reset_link?: string }>('/api/auth/forgot', json({ email })),
  reset: (token: string, new_password: string) => j<AuthState>('/api/auth/reset', json({ token, new_password })),
  uploadLogo: (sid: string, file: File) => { const f = new FormData(); f.append('file', file); return j<{ ok: boolean }>(`/api/shoots/${sid}/logo`, { method: 'POST', body: f }) },
  // colour tool (eyedropper + one-click recolour): deterministic, no planner, sub-second
  colorAt: (sid: string, pid: string, body: PointBody) => j<ColorAt>(`/api/shoots/${sid}/photos/${pid}/color-at`, json(body)),
  regionAt: (sid: string, pid: string, body: PointBody & { mode: RegionMode; points?: number[][]; follow_light?: boolean }) => j<RegionAt>(`/api/shoots/${sid}/photos/${pid}/region-at`, json(body)),
  applyColor: (sid: string, pid: string, body: PointBody & { mode: RegionMode; points?: number[][]; follow_light?: boolean; mask_edit?: MaskEdit | null; hex: string; strength: number; luminance_follow: number }) => j<Photo>(`/api/shoots/${sid}/photos/${pid}/apply-color`, json(body)),
}

export type PickSource = 'head' | 'original' | 'bracket' | 'reference'
export type RegionMode = 'auto' | 'surface' | 'object'
export type PointBody = { x: number; y: number; source?: PickSource; index?: number | null; tolerance?: number }
export type ColorAt = { hex: string; hex_at_click: string; hex_light: string; hex_shade: string; lab: number[]; pixels: number; area_pct: number; region_mode: string; box: number[] | null; photo: string; sampled_from: string; source: PickSource; index: number | null; preview_url: string }
export type RegionAt = { area_pct: number; mode_used: string; box: number[] | null; preview_url: string }
export type MaskEdit = { add: string | null; erase: string | null }   // PNG data URLs drawn in the editor: opaque = painted
export type PickedColor = { hex: string; from: string }
export type PaintMask = { photo_id: string; png: string; area: number }
export type ChatBody = { text: string; selected_photo_id?: string | null; points?: number[][] | null; mask?: PaintMask | null; picked_color?: PickedColor | null; scope: 'shoot' | 'photo'; confirm_first: boolean }

/** POST + parse a text/event-stream response (sse-starlette format). */
export async function* chatStream(sid: string, body: ChatBody): AsyncGenerator<SseEvent> {
  const r = await fetch(`/api/shoots/${sid}/chat`, { ...json(body), headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' } })
  if (!r.ok || !r.body) { if (r.status === 401) onUnauthorized(); throw new Error(`${r.status} ${await r.text()}`) }
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    // sse-starlette terminates lines with \r\n; normalise so the '\n\n' event boundary below matches
    buf = (buf + dec.decode(value, { stream: true })).replace(/\r\n/g, '\n')
    let idx: number
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2)
      let type = 'message'; const data: string[] = []
      for (const line of chunk.split('\n')) {
        if (line.startsWith('event:')) type = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (!data.length) continue
      try { yield { type, data: JSON.parse(data.join('\n')) } } catch { yield { type, data: data.join('\n') } }
    }
  }
}
