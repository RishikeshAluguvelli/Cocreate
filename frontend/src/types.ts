export type QC = { verdict: 'pass' | 'needs_review' | 'skipped'; fulfilled?: number; collateral?: number; artifacts?: number; reason?: string }
export type Version = {
  id: string; index: number; parent_id: string | null; tool: string; summary: string
  params: Record<string, unknown> & { qc?: QC }; url: string; created_at: number; cost_usd: number
  mask_preview_url?: string | null
}
export type Photo = {
  id: string; name: string; width: number; height: number; original_url: string
  head_version_id: string; brackets: string[]; references: string[]
  dataset_request?: string | null; versions: Version[]
}
export type ToolEvent = { type: string; name?: string; input?: any; result?: any; error?: string }
export type Scope = { mode: 'shoot' | 'photo'; photo_name?: string | null; photos?: number }
export type ChatMsg = {
  role: 'user' | 'assistant'; text: string; created_at: number; events: ToolEvent[]
  thinking?: string; scope?: Scope | null; status_log?: string[]; seconds?: number | null
}
export type ArchivedChat = { id: string; created_at: number; ended_at: number; messages: ChatMsg[]; memory_notes?: string }
export type Shoot = {
  id: string; name: string; created_at: number; total_cost_usd: number; has_logo: boolean; logo_url?: string | null
  photos: Photo[]; chat: ChatMsg[]; archived_chats: ArchivedChat[]; memory_notes: string
}
export type ShootSummary = { id: string; name: string; photos: number; cover: string | null; created_at: number; updated_at: number; edits: number; messages: number; archived_chats: number }
export type Template = { id: string; title: string; blurb: string; count: number; cover: string; requests: string[] }
export type AuthMode = 'open' | 'guest' | 'required'
export type Health = { ok: boolean; planner: boolean; planner_provider: string; image_models: boolean; image_backend: string; mock: boolean; planner_model: string; edit_model: string; auth_mode?: AuthMode }
export type User = { id: string; email: string | null; is_guest: boolean }
export type AuthState = { auth_mode: AuthMode; user: User | null }
export type SseEvent = { type: string; data: any }

export const headOf = (p: Photo): Version => p.versions.find(v => v.id === p.head_version_id) ?? p.versions[0]
export const prevOf = (p: Photo): Version | undefined => {
  const h = headOf(p)
  return h.parent_id ? p.versions.find(v => v.id === h.parent_id) : undefined
}
export const TOOL_LABEL: Record<string, string> = {
  inspect_photo: 'Inspected', sample_color: 'Sampled colour', repaint: 'Re-painted', adjust: 'Adjusted', draw_lot_lines: 'Lot lines',
  add_watermark: 'Watermarked', remove_object: 'Removed', generative_edit: 'AI edit', tv_screen: 'TV screen', blur: 'Blurred',
  match_look: 'Matched look', crop_rotate: 'Crop / straighten', undo: 'Undid', revert_to: 'Reverted', manual_adjust: 'Sliders', recolor: 'Recoloured',
}
