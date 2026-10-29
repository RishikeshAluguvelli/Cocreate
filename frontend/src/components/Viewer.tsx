import { useEffect, useMemo, useRef, useState } from 'react'
import { Brush, Columns2, Diff, Download, Eraser, Layers, Loader2, MapPin, MessageSquareText, Pipette, RefreshCw, ShieldAlert, ShieldCheck, SlidersHorizontal, Undo2, X } from 'lucide-react'
import type { ColorAt, PaintMask, PickSource, PickedColor, RegionAt, RegionMode } from '../api'
import type { Photo, Version } from '../types'
import { headOf, TOOL_LABEL } from '../types'
import { api } from '../api'

type Vals = { brightness: number; contrast: number; saturation: number; temperature: number; shadows: number; highlights: number; sharpen: number }
const ZERO: Vals = { brightness: 0, contrast: 0, saturation: 0, temperature: 0, shadows: 0, highlights: 0, sharpen: 0 }
const KEYS = Object.keys(ZERO) as (keyof Vals)[]
const PREVIEW_MAX = 1280
const ADJUST_TOOLS = new Set(['adjust', 'manual_adjust'])
const isGlobalAdjust = (v: Version) => ADJUST_TOOLS.has(v.tool) && !v.params?.region
const clampVal = (k: keyof Vals, x: number) => Math.max(k === 'sharpen' ? 0 : -100, Math.min(100, Math.round(x)))
const sameVals = (a: Vals, b: Vals) => KEYS.every(k => a[k] === b[k])
/** Walk back through consecutive whole-frame adjustments (chat or manual): the image they started from and their summed
 *  slider values. The sliders show these values and re-render from that base, so a chat "brighten 20%" moves the
 *  brightness slider to +20 and dragging it to +30 refines the edit instead of stacking another +30 on top. */
function adjustChain(photo: Photo, head: Version): { base: Version; vals: Vals } {
  const vals = { ...ZERO }
  let v: Version | undefined = head
  while (v && isGlobalAdjust(v)) {
    for (const k of KEYS) vals[k] += Number(v.params?.[k] ?? 0)
    const pid: string | null = v.parent_id
    v = pid ? photo.versions.find(x => x.id === pid) : undefined
  }
  for (const k of KEYS) vals[k] = clampVal(k, vals[k])
  return { base: v ?? photo.versions[0], vals }
}

// ---- same maths as backend/app/tools/brightness.py, run in the browser for live preview ----
const LIN = new Float32Array(256)
for (let i = 0; i < 256; i++) { const x = i / 255; LIN[i] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4) }
const SRGB = new Float32Array(4097)
for (let i = 0; i <= 4096; i++) { const x = i / 4096; SRGB[i] = x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055 }
const toSrgb = (x: number) => x <= 0 ? 0 : x >= 1 ? 1.055 * Math.pow(x, 1 / 2.4) - 0.055 : SRGB[(x * 4096) | 0]

function applyAdjust(src: ImageData, v: Vals): ImageData {
  const s = src.data, n = s.length
  const out = new ImageData(src.width, src.height); const d = out.data
  const exp = Math.pow(2, 1.5 * v.brightness / 100)
  const t = v.temperature / 100, tr = 1 + 0.12 * t, tg = 1 + 0.02 * t, tb = 1 - 0.12 * t
  const sh = v.shadows / 100, hl = v.highlights / 100
  const c = v.contrast / 100, k = c > 0 ? 1 + 0.9 * c : 1 + 0.6 * c
  const sat = 1 + v.saturation / 100
  for (let i = 0; i < n; i += 4) {
    let r = LIN[s[i]] * exp * tr, g = LIN[s[i + 1]] * exp * tg, b = LIN[s[i + 2]] * exp * tb
    if (sh || hl) {
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b
      if (sh) { let w = 1 - y / 0.35; w = w < 0 ? 0 : w > 1 ? 1 : w; const m = 1 + 0.8 * sh * Math.pow(w, 1.5); r *= m; g *= m; b *= m }
      if (hl) { let w = (y - 0.55) / 0.45; w = w < 0 ? 0 : w > 1 ? 1 : w; const m = 1 - 0.6 * hl * Math.pow(w, 1.2); r *= m; g *= m; b *= m }
    }
    let R = toSrgb(r), G = toSrgb(g), B = toSrgb(b)
    if (c) { R = 0.5 + (R - 0.5) * k; G = 0.5 + (G - 0.5) * k; B = 0.5 + (B - 0.5) * k }
    if (v.saturation) { const y = 0.2126 * R + 0.7152 * G + 0.0722 * B; R = y + (R - y) * sat; G = y + (G - y) * sat; B = y + (B - y) * sat }
    d[i] = R <= 0 ? 0 : R >= 1 ? 255 : R * 255 + 0.5; d[i + 1] = G <= 0 ? 0 : G >= 1 ? 255 : G * 255 + 0.5; d[i + 2] = B <= 0 ? 0 : B >= 1 ? 255 : B * 255 + 0.5; d[i + 3] = 255
  }
  if (v.sharpen) {  // 3x3 unsharp mask approximating the backend's gaussian unsharp
    const a = 1.2 * v.sharpen / 100, w = src.width, h = src.height, p = new Uint8ClampedArray(d)
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) for (let ch = 0; ch < 3; ch++) {
      const i = (y * w + x) * 4 + ch
      const blur = (p[i - w * 4 - 4] + 2 * p[i - w * 4] + p[i - w * 4 + 4] + 2 * p[i - 4] + 4 * p[i] + 2 * p[i + 4] + p[i + w * 4 - 4] + 2 * p[i + w * 4] + p[i + w * 4 + 4]) / 16
      const val = p[i] + (p[i] - blur) * a
      d[i] = val < 0 ? 0 : val > 255 ? 255 : val
    }
  }
  return out
}

async function loadImageData(url: string, maxSide: number, size?: { w: number; h: number }): Promise<ImageData> {
  const img = new Image(); img.decoding = 'async'; img.src = url
  await img.decode()
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight))
  const w = size?.w ?? Math.max(1, Math.round(img.naturalWidth * scale)), h = size?.h ?? Math.max(1, Math.round(img.naturalHeight * scale))
  const c = document.createElement('canvas'); c.width = w; c.height = h
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(img, 0, 0, w, h)
  return ctx.getImageData(0, 0, w, h)
}

const errText = (e: any) => String(e?.message ?? e).replace(/^\d+\s*/, '').replace(/^\{"detail":"(.*)"\}$/, '$1')
const HEX_RE = /^#[0-9a-fA-F]{6}$/

type Props = {
  shootId: string; photo: Photo; points: number[][]; pointsMode: boolean; busy?: boolean
  onPointsMode: (on: boolean) => void; onPoints: (pts: number[][]) => void; onPhoto: (p: Photo) => void
  onSend?: (text: string) => void
  paintMode: boolean; onPaintMode: (on: boolean) => void; paint: PaintMask | null; onPaint: (m: PaintMask | null) => void
  colorMode: boolean; onColorMode: (on: boolean) => void; picked: PickedColor | null; onPicked: (c: PickedColor | null) => void
}
const PAINT_RES = 1024
type PickSrc = { source: PickSource; index: number | null }

export default function Viewer({ shootId, photo, points, pointsMode, busy: chatBusy, onPointsMode, onPoints, onPhoto, onSend, paintMode, onPaintMode, paint, onPaint, colorMode, onColorMode, picked, onPicked }: Props) {
  const head = headOf(photo)
  const chain = useMemo(() => adjustChain(photo, head), [photo, head])
  const [compare, setCompare] = useState(true)
  const [split, setSplit] = useState(50)
  const [view, setView] = useState<'image' | 'mask' | 'changes'>('image')
  const [compareTo, setCompareTo] = useState<'original' | 'previous'>('previous')
  const [sliders, setSliders] = useState(false)
  const [vals, setVals] = useState<Vals>(() => chain.vals)
  const [busy, setBusy] = useState(false)
  const [changedPct, setChangedPct] = useState<number | null>(null)
  const imgWrap = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const changesRef = useRef<HTMLCanvasElement>(null)
  const srcData = useRef<{ url: string; data: ImageData } | null>(null)
  const raf = useRef(0)
  // paint-to-fix: strokes are drawn on an offscreen-resolution canvas and exported as a PNG mask with the next message
  const paintRef = useRef<HTMLCanvasElement>(null)
  const painting = useRef(false)
  const lastPt = useRef<[number, number] | null>(null)
  const [brush, setBrush] = useState(4)          // percent of image width
  const [erasing, setErasing] = useState(false)
  const paintW = PAINT_RES, paintH = Math.max(1, Math.round(PAINT_RES * photo.height / photo.width))
  // colour tool: pick a colour from any image of this photo (head / original / bracket / reference), apply it with one click
  const [stage, setStage] = useState<'pick' | 'apply'>('pick')
  const [pickSrc, setPickSrc] = useState<PickSrc>({ source: 'head', index: null })
  const [tolerance, setTolerance] = useState(50)
  const [regionMode, setRegionMode] = useState<RegionMode>('auto')
  const [followLight, setFollowLight] = useState(false)
  const [strength, setStrength] = useState(100)
  const [lumFollow, setLumFollow] = useState(70)
  const [pickInfo, setPickInfo] = useState<ColorAt | null>(null)
  const [region, setRegion] = useState<(RegionAt & { x: number; y: number; points: number[][] }) | null>(null)
  const [colorBusy, setColorBusy] = useState<string | null>(null)
  const [colorErr, setColorErr] = useState<string | null>(null)
  const [hexDraft, setHexDraft] = useState(picked?.hex ?? '')
  useEffect(() => { setHexDraft(picked?.hex ?? '') }, [picked?.hex])
  useEffect(() => { setPickInfo(null); setRegion(null); setPickSrc({ source: 'head', index: null }); setColorErr(null) }, [photo.id])
  useEffect(() => { if (!colorMode) { setRegion(null); setPickInfo(null); setColorErr(null) } }, [colorMode])
  // when the band or mode changes, recompute the previewed region for the same click
  useEffect(() => {
    if (!region) return
    const { x, y, points: pts } = region
    const t = setTimeout(async () => {
      try { const r = await api.regionAt(shootId, photo.id, { x, y, points: pts, tolerance, mode: regionMode, follow_light: followLight }); setRegion({ ...r, x, y, points: pts }) } catch { /* keep the previous preview */ }
    }, 250)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tolerance, regionMode, followLight])
  const srcUrl = (s: PickSrc) => s.source === 'original' ? photo.original_url : s.source === 'bracket' ? photo.brackets[s.index ?? 0] : s.source === 'reference' ? photo.references[s.index ?? 0] : head.url
  const showingSource = colorMode && stage === 'pick' && pickSrc.source !== 'head'
  const pickOverlayVisible = colorMode && stage === 'pick' && !!pickInfo && pickInfo.source === pickSrc.source && (pickInfo.index ?? null) === (pickSrc.index ?? null)

  useEffect(() => {   // a painted mask belongs to one photo: clear the canvas when another photo is shown
    const c = paintRef.current; if (!c) return
    const ctx = c.getContext('2d')!; ctx.clearRect(0, 0, c.width, c.height)
    if (paint && paint.photo_id === photo.id) { const im = new Image(); im.onload = () => ctx.drawImage(im, 0, 0, c.width, c.height); im.src = paint.png }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photo.id, paintMode])
  const paintPos = (e: React.PointerEvent): [number, number] | null => {
    const c = paintRef.current; if (!c) return null
    const r = c.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * c.width, ((e.clientY - r.top) / r.height) * c.height]
  }
  const stroke = (from: [number, number], to: [number, number]) => {
    const c = paintRef.current!; const ctx = c.getContext('2d')!
    ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over'
    ctx.strokeStyle = '#ffffff'; ctx.fillStyle = '#ffffff'; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
    ctx.lineWidth = Math.max(2, (brush / 100) * c.width)
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke()
  }
  const exportPaint = () => {
    const c = paintRef.current!; const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
    let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 127) n++
    if (n === 0) { onPaint(null); return }
    onPaint({ photo_id: photo.id, png: c.toDataURL('image/png'), area: n / (c.width * c.height) })
  }
  const paintDown = (e: React.PointerEvent) => { if (!paintMode) return; e.preventDefault(); (e.target as HTMLElement).setPointerCapture(e.pointerId); painting.current = true; const p = paintPos(e); if (p) { lastPt.current = p; stroke(p, p) } }
  const paintMove = (e: React.PointerEvent) => { if (!paintMode || !painting.current) return; const p = paintPos(e); if (p && lastPt.current) { stroke(lastPt.current, p); lastPt.current = p } }
  const paintUp = () => { if (!painting.current) return; painting.current = false; lastPt.current = null; exportPaint() }
  const clearPaint = () => { const c = paintRef.current; if (c) c.getContext('2d')!.clearRect(0, 0, c.width, c.height); onPaint(null) }

  useEffect(() => {
    setView('image'); setChangedPct(null)
    setVals(chain.vals)                                                   // sliders track what the head version applied
    if (chatBusy && isGlobalAdjust(head)) setSliders(true)                // the assistant used the sliders: show them moving
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [head.id])

  const before: Version | undefined = useMemo(() => {
    if (head.index === 0) return undefined
    if (compareTo === 'original') return photo.versions[0]
    return photo.versions.find(v => v.id === head.parent_id) ?? photo.versions[0]
  }, [photo, head, compareTo])

  const dirty = !sameVals(vals, chain.vals)
  const hasSaved = KEYS.some(k => chain.vals[k] !== 0)
  const allZero = KEYS.every(k => vals[k] === 0)
  const previewing = sliders && dirty

  // live slider preview: decode the base image once, re-run the adjustment maths on every slider move
  useEffect(() => {
    if (!sliders) return
    let cancelled = false
    const url = chain.base.url
    if (srcData.current?.url !== url) {
      srcData.current = null
      loadImageData(url, PREVIEW_MAX).then(data => { if (!cancelled) { srcData.current = { url, data }; schedule() } })
    }
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sliders, chain.base.url])
  const schedule = () => {
    cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(() => {
      const c = canvasRef.current, src = srcData.current
      if (!c || !src) return
      const out = applyAdjust(src.data, vals)
      if (c.width !== out.width || c.height !== out.height) { c.width = out.width; c.height = out.height }
      c.getContext('2d')!.putImageData(out, 0, 0)
    })
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (previewing) schedule() }, [vals, previewing])

  // change heatmap: where does the current version differ from the one it is compared with?
  useEffect(() => {
    if (view !== 'changes' || !before) return
    let cancelled = false
    ;(async () => {
      const a = await loadImageData(before.url, PREVIEW_MAX)
      const b = await loadImageData(head.url, PREVIEW_MAX, { w: a.width, h: a.height })
      if (cancelled || !changesRef.current) return
      const out = new ImageData(a.width, a.height); const d = out.data; let n = 0
      for (let i = 0; i < d.length; i += 4) {
        const diff = Math.max(Math.abs(a.data[i] - b.data[i]), Math.abs(a.data[i + 1] - b.data[i + 1]), Math.abs(a.data[i + 2] - b.data[i + 2]))
        if (diff > 16) { d[i] = 255; d[i + 1] = 40; d[i + 2] = 200; d[i + 3] = Math.min(230, 90 + diff * 2); n++ }
      }
      const c = changesRef.current; c.width = a.width; c.height = a.height
      c.getContext('2d')!.putImageData(out, 0, 0)
      setChangedPct(Math.round(1000 * n / (a.width * a.height)) / 10)
    })()
    return () => { cancelled = true }
  }, [view, before, head.url])

  const onPointer = (e: React.PointerEvent) => {
    if (!dragging.current || !imgWrap.current) return
    const r = imgWrap.current.getBoundingClientRect()
    setSplit(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)))
  }
  const norm = (e: React.MouseEvent): [number, number] | null => {
    if (!imgWrap.current) return null
    const r = imgWrap.current.getBoundingClientRect()
    return [Math.round(((e.clientX - r.left) / r.width) * 1000) / 1000, Math.round(((e.clientY - r.top) / r.height) * 1000) / 1000]
  }
  const onClick = async (e: React.MouseEvent) => {
    const p = norm(e); if (!p) return
    const [x, y] = p
    if (colorMode) {
      if (colorBusy) return
      setColorErr(null)
      if (stage === 'pick') {
        setColorBusy('Reading the colour…')
        try {
          const res = await api.colorAt(shootId, photo.id, { x, y, source: pickSrc.source, index: pickSrc.index, tolerance })
          setPickInfo(res); onPicked({ hex: res.hex, from: `${photo.name} · ${res.sampled_from}` })
        } catch (err) { setColorErr(errText(err)) } finally { setColorBusy(null) }
      } else {
        setColorBusy(region ? 'Adding to the selection…' : 'Finding the surface…')
        const pts = [...(region?.points ?? []), [x, y]]
        try { const res = await api.regionAt(shootId, photo.id, { x, y, points: pts, tolerance, mode: regionMode, follow_light: followLight }); setRegion({ ...res, x, y, points: pts }) }
        catch (err) { setColorErr(errText(err)) } finally { setColorBusy(null) }
      }
      return
    }
    if (!pointsMode) return
    onPoints([...points, [x, y]])
  }
  const applyColor = async () => {
    if (!region || !picked) return
    setColorBusy('Recolouring…'); setColorErr(null)
    try {
      const p = await api.applyColor(shootId, photo.id, { x: region.x, y: region.y, points: region.points, tolerance, mode: regionMode, follow_light: followLight, hex: picked.hex, strength: strength / 100, luminance_follow: lumFollow / 100 })
      onPhoto(p); setRegion(null)
    } catch (err) { setColorErr(errText(err)) } finally { setColorBusy(null) }
  }
  const askAssistant = () => {
    if (!region || !picked || !onSend) return
    onPoints([[region.x, region.y]])
    onSend(`Repaint the surface at the point I marked to ${picked.hex}, keeping its texture and shading; nothing else changes.`)
    setRegion(null); onColorMode(false)
  }
  const setHex = (h: string) => { setHexDraft(h); if (HEX_RE.test(h)) onPicked({ hex: h.toUpperCase(), from: 'typed' }) }
  const enterMode = (m: 'points' | 'paint' | 'color', on: boolean) => {
    onPointsMode(m === 'points' ? on : false); onPaintMode(m === 'paint' ? on : false); onColorMode(m === 'color' ? on : false)
  }
  const revert = async (v: Version) => { setBusy(true); try { onPhoto(await api.revert(shootId, photo.id, v.id)) } finally { setBusy(false) } }
  const applySliders = async () => {
    setBusy(true)
    try { onPhoto(await api.manualAdjust(shootId, photo.id, vals, chain.base.id)) }
    finally { setBusy(false) }
  }
  const qc = head.params?.qc
  const sources: { label: string; src: PickSrc }[] = [
    { label: 'This photo', src: { source: 'head', index: null } },
    ...(head.index > 0 ? [{ label: 'Before edits', src: { source: 'original' as PickSource, index: null } }] : []),
    ...photo.brackets.map((_, i) => ({ label: `Bracket ${i + 1}`, src: { source: 'bracket' as PickSource, index: i } })),
    ...photo.references.map((_, i) => ({ label: `Reference ${i + 1}`, src: { source: 'reference' as PickSource, index: i } })),
  ]
  const sameSrc = (a: PickSrc, b: PickSrc) => a.source === b.source && (a.index ?? null) === (b.index ?? null)
  const mainUrl = showingSource ? srcUrl(pickSrc) : view === 'mask' && head.mask_preview_url ? head.mask_preview_url : head.url
  const compareVisible = !!before && compare && view === 'image' && !showingSource

  return (
    <div className="flex h-full flex-col">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-3 py-2 text-xs">
        <span className="font-medium">{photo.name}</span>
        <span className="text-zinc-500">{photo.width}×{photo.height} · v{head.index}{head.index ? ` · ${TOOL_LABEL[head.tool] ?? head.tool}` : ''}</span>
        {qc && qc.verdict !== 'skipped' && (
          <span title={qc.reason} className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 ${qc.verdict === 'pass' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-amber-500/15 text-amber-200'}`}>
            {qc.verdict === 'pass' ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />} {qc.verdict === 'pass' ? 'QC pass' : 'QC: check'} {qc.fulfilled != null ? `${qc.fulfilled}/5` : ''}
          </span>
        )}
        <span className="ml-auto" />
        <button className={`btn btn-ghost ${compare && before ? 'ring-1 ring-amber-400/60' : ''}`} disabled={!before} onClick={() => setCompare(c => !c)} title="Before / after">
          <Columns2 size={14} /> Compare
        </button>
        {before && compare && (
          <select value={compareTo} onChange={e => setCompareTo(e.target.value as any)} className="rounded-lg bg-zinc-800 px-2 py-1.5">
            <option value="previous">vs previous</option><option value="original">vs original</option>
          </select>
        )}
        <button className={`btn btn-ghost ${view === 'mask' ? 'ring-1 ring-amber-400/60' : ''}`} disabled={!head.mask_preview_url} onClick={() => setView(v => v === 'mask' ? 'image' : 'mask')} title="Show the mask used for this edit">
          <Layers size={14} /> Mask
        </button>
        <button className={`btn btn-ghost ${view === 'changes' ? 'ring-1 ring-amber-400/60' : ''}`} disabled={!before} onClick={() => setView(v => v === 'changes' ? 'image' : 'changes')} title="Highlight every pixel that differs from the compared version (proves the rest is untouched)">
          <Diff size={14} /> Changes{view === 'changes' && changedPct != null ? ` ${changedPct}%` : ''}
        </button>
        <button className={`btn btn-ghost ${pointsMode ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => enterMode('points', !pointsMode)} title="Click on the photo to place points (lot-line corners, an object to remove, a wall to sample)">
          <MapPin size={14} /> Points{points.length ? ` (${points.length})` : ''}
        </button>
        {points.length > 0 && <button className="btn btn-ghost" onClick={() => onPoints([])}><Eraser size={14} /> Clear</button>}
        <button className={`btn btn-ghost ${paintMode ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => enterMode('paint', !paintMode)} title="Paint an area on the photo; it is sent with your next message so the assistant edits exactly there (fix a missed spot, extend a repaint, remove something detection missed)">
          <Brush size={14} /> Paint{paint && paint.photo_id === photo.id ? ` (${(paint.area * 100).toFixed(1)}%)` : ''}
        </button>
        <button className={`btn btn-ghost ${colorMode ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => enterMode('color', !colorMode)} data-testid="color-tool" title="Eyedropper: pick the colour of any wall or object (here, in the original, a bracket or a reference), then click another wall or object to recolour it. Deterministic and instant.">
          <Pipette size={14} /> Colour{picked && <span className="ml-0.5 inline-block h-3 w-3 rounded-sm border border-zinc-600 align-middle" style={{ background: picked.hex }} />}
        </button>
        <button className={`btn btn-ghost ${sliders ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => { setSliders(s => !s); setVals(chain.vals) }}><SlidersHorizontal size={14} /> Sliders{hasSaved && !sliders ? ' •' : ''}</button>
        <button className="btn btn-ghost" disabled={head.index === 0 || busy} onClick={() => before && revert(before)} title="Undo last edit"><Undo2 size={14} /></button>
        {onSend && head.index > 0 && ['generative_edit', 'remove_object', 'tv_screen', 'repaint'].includes(head.tool) && (
          <button className="btn btn-ghost" disabled={busy || chatBusy} title="Ask the assistant to redo this edit for a different, cleaner result"
            onClick={() => onSend(`Regenerate the last edit on ${photo.name} (${head.summary.split(' (')[0]}); aim for a cleaner result and keep everything else identical.`)}>
            <RefreshCw size={14} /> Regenerate
          </button>
        )}
        <a className="btn btn-ghost" href={`/api/shoots/${shootId}/photos/${photo.id}/download`} title="Download this photo"><Download size={14} /></a>
      </div>

      {/* image (+ the colour tool as a side column so it never covers the photo) */}
      <div className="flex min-h-0 flex-1">
      <div className="relative flex min-h-0 flex-1 items-center justify-center bg-zinc-950 p-3">
        <div ref={imgWrap} data-testid="viewer-image" onClick={onClick} onPointerMove={onPointer} onPointerUp={() => (dragging.current = false)} onPointerLeave={() => (dragging.current = false)}
          className={`relative max-h-full select-none overflow-hidden rounded-lg shadow-2xl ${pointsMode || colorMode ? 'cursor-crosshair' : ''}`}
          style={{ aspectRatio: `${photo.width} / ${photo.height}`, maxWidth: '100%' }}>
          {previewing && !showingSource
            ? <canvas ref={canvasRef} className="block h-full w-full" />
            : <img src={mainUrl} alt="" className="block h-full w-full" draggable={false} />}
          {view === 'changes' && before && !showingSource && <canvas ref={changesRef} className="pointer-events-none absolute inset-0 h-full w-full" />}
          {compareVisible && before && (
            <>
              <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
                <img src={before.url} alt="" className="block h-full w-full" draggable={false} />
              </div>
              <div className="absolute inset-y-0 w-0.5 bg-amber-400" style={{ left: `${split}%` }} />
              <div onPointerDown={e => { dragging.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId) }}
                className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize rounded-full border-2 border-amber-400 bg-zinc-900 px-2 py-1 text-[10px]" style={{ left: `${split}%` }}>⇔</div>
              <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px]">before · v{before.index}</span>
              <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px]">{previewing ? 'after · live preview' : `after · v${head.index}`}</span>
            </>
          )}
          {view === 'changes' && !showingSource && <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px]">magenta = pixels changed vs v{before?.index}{changedPct != null ? ` · ${changedPct}% of frame` : ''}</span>}
          {previewing && !compareVisible && !showingSource && <span className="absolute right-2 top-2 rounded bg-amber-400 px-1.5 py-0.5 text-[10px] font-medium text-zinc-950">live preview · not saved</span>}
          {showingSource && <span className="absolute left-2 top-2 rounded bg-sky-400 px-1.5 py-0.5 text-[10px] font-medium text-zinc-950">showing {sources.find(s => sameSrc(s.src, pickSrc))?.label.toLowerCase()} · click to pick a colour</span>}
          {/* points overlay */}
          {points.length > 0 && (
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1000 1000" preserveAspectRatio="none">
              {points.length > 1 && <polyline points={points.map(p => `${p[0] * 1000},${p[1] * 1000}`).join(' ')} fill="rgba(251,191,36,0.12)" stroke="#fbbf24" strokeWidth="3" strokeDasharray="12 8" vectorEffect="non-scaling-stroke" />}
              {points.map((p, i) => <circle key={i} cx={p[0] * 1000} cy={p[1] * 1000} r="7" fill="#fbbf24" stroke="#000" strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
            </svg>
          )}
          {pointsMode && <span className="absolute bottom-2 left-2 rounded bg-amber-400 px-2 py-0.5 text-[11px] font-medium text-zinc-950">Click to add points · they are sent with your next message</span>}
          {/* colour tool overlays: the sampled region (green) or the region about to be recoloured (amber) */}
          {pickOverlayVisible && pickInfo && <img src={pickInfo.preview_url} alt="" className="pointer-events-none absolute inset-0 h-full w-full" draggable={false} />}
          {colorMode && stage === 'apply' && region && <img src={region.preview_url} alt="" className="pointer-events-none absolute inset-0 h-full w-full" draggable={false} data-testid="region-overlay" />}
          {colorMode && stage === 'apply' && region && region.points.map((pt, i) => <span key={i} className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-zinc-950 bg-amber-400" style={{ left: `${pt[0] * 100}%`, top: `${pt[1] * 100}%` }} />)}
          {colorMode && (
            <span className="absolute bottom-2 left-2 flex items-center gap-1.5 rounded bg-amber-400 px-2 py-0.5 text-[11px] font-medium text-zinc-950">
              {colorBusy ? <><Loader2 size={11} className="animate-spin" /> {colorBusy}</> : stage === 'pick' ? 'Click a wall or object to read its colour' : !picked ? 'Pick or type a colour first, then click a wall or object' : region ? 'Click again to add more of the surface · Apply when the highlight is right' : `Click the wall or object to paint ${picked.hex}`}
            </span>
          )}
          {/* paint overlay: visible whenever a painted area exists for this photo, interactive only in paint mode */}
          <canvas ref={paintRef} width={paintW} height={paintH} data-testid="paint-canvas"
            onPointerDown={paintDown} onPointerMove={paintMove} onPointerUp={paintUp} onPointerCancel={paintUp}
            className={`absolute inset-0 h-full w-full ${paintMode ? 'cursor-crosshair touch-none' : 'pointer-events-none'} ${paint && paint.photo_id === photo.id || paintMode ? '' : 'hidden'}`}
            style={{ opacity: 0.55, filter: 'drop-shadow(0 0 0 #f59e0b)', mixBlendMode: 'normal' }} />
          {paintMode && <span className="absolute bottom-2 left-2 rounded bg-amber-400 px-2 py-0.5 text-[11px] font-medium text-zinc-950">Paint over the area to fix · then tell the assistant what to do there</span>}
        </div>

        {paintMode && (
          <div className="card absolute left-4 top-4 z-10 w-60 !bg-zinc-900 p-3 text-xs shadow-2xl">
            <div className="mb-1 flex items-center justify-between font-medium">Paint an area <button onClick={() => onPaintMode(false)}><X size={14} /></button></div>
            <div className="mb-2 text-[10px] text-zinc-500">The white area goes with your next message. Say what to do there: "also repaint this strip", "remove what I painted", "turn this lamp on".</div>
            <label className="block"><span className="flex justify-between text-zinc-400"><span>Brush size</span><span>{brush}%</span></span>
              <input type="range" min={1} max={15} value={brush} onChange={e => setBrush(Number(e.target.value))} className="w-full accent-amber-400" /></label>
            <div className="mt-2 flex gap-2">
              <button className={`btn btn-ghost flex-1 justify-center ${erasing ? 'ring-1 ring-amber-400/60' : ''}`} onClick={() => setErasing(x => !x)}><Eraser size={14} /> {erasing ? 'Erasing' : 'Erase'}</button>
              <button className="btn btn-ghost flex-1 justify-center" disabled={!paint} onClick={clearPaint}>Clear</button>
            </div>
            {paint && paint.photo_id === photo.id && <div className="mt-2 text-[10px] text-amber-200">{(paint.area * 100).toFixed(1)}% of the photo painted · will be sent with your next message</div>}
          </div>
        )}

        {sliders && (
          <div className="card absolute right-4 top-4 z-10 w-64 !bg-zinc-900 p-3 text-xs shadow-2xl">
            <div className="mb-1 flex items-center justify-between font-medium">Adjustments <button onClick={() => { setSliders(false); setVals(chain.vals) }}><X size={14} /></button></div>
            <div className="mb-2 text-[10px] text-zinc-500">
              {hasSaved
                ? <>Showing the values applied in <b className="text-zinc-300">v{head.index}</b>{head.tool === 'adjust' ? ' by the assistant' : ''}. Drag to fine-tune; the result is re-rendered from v{chain.base.index}. </>
                : <>Previews live on the photo{before && compare ? ' (drag the compare handle)' : ''}. </>}
              Nothing is saved until you apply.
            </div>
            {KEYS.map(k => (
              <label key={k} className="mb-1.5 block">
                <span className="flex justify-between text-zinc-400"><span className="capitalize">{k}</span>
                  <span>{vals[k] !== chain.vals[k] && <span className="text-zinc-600">{chain.vals[k]} → </span>}<span className={vals[k] !== 0 ? 'text-amber-200' : ''}>{vals[k]}</span></span></span>
                <input type="range" min={k === 'sharpen' ? 0 : -100} max={100} value={vals[k]} onChange={e => setVals(v => ({ ...v, [k]: Number(e.target.value) }))} onDoubleClick={() => setVals(v => ({ ...v, [k]: 0 }))} className="w-full accent-amber-400" />
              </label>
            ))}
            <div className="mt-2 flex gap-2">
              <button className="btn btn-ghost flex-1 justify-center" disabled={!dirty} onClick={() => setVals(chain.vals)} title="Back to the saved values">Reset</button>
              <button className="btn btn-primary flex-1 justify-center" disabled={busy || !dirty} onClick={applySliders}>{allZero ? `Back to v${chain.base.index}` : `Apply as v${photo.versions.length}`}</button>
            </div>
          </div>
        )}
      </div>

        {colorMode && (
          <aside className="thin-scroll w-72 shrink-0 overflow-y-auto border-l border-zinc-800 bg-zinc-900 p-3 text-xs" data-testid="color-panel">
            <div className="mb-2 flex items-center justify-between font-medium"><span className="flex items-center gap-1.5"><Pipette size={13} className="text-amber-300" /> Colour tool</span><button onClick={() => onColorMode(false)} aria-label="Close"><X size={14} /></button></div>
            <p className="mb-2 text-[11px] leading-relaxed text-zinc-500">Read a colour from any wall or object (also from the original, a bracket or a reference), then paint it onto another with one click. Deterministic, instant, undoable.</p>
            <div className="grid grid-cols-2 rounded-lg border border-zinc-700 bg-zinc-950 p-0.5" role="tablist">
              <button role="tab" aria-selected={stage === 'pick'} className={`rounded-md py-1.5 ${stage === 'pick' ? 'bg-amber-400 font-medium text-zinc-950' : 'text-zinc-300 hover:text-zinc-100'}`} onClick={() => { setStage('pick'); setRegion(null) }}>1 · Pick</button>
              <button role="tab" aria-selected={stage === 'apply'} className={`rounded-md py-1.5 ${stage === 'apply' ? 'bg-amber-400 font-medium text-zinc-950' : 'text-zinc-300 hover:text-zinc-100'}`} onClick={() => setStage('apply')}>2 · Apply</button>
            </div>

            {/* palette: always visible */}
            <div className="mt-3 flex items-center gap-2">
              <label className="relative h-9 w-9 shrink-0 cursor-pointer overflow-hidden rounded-lg border border-zinc-600" title="Open a colour picker" style={{ background: HEX_RE.test(hexDraft) ? hexDraft : 'repeating-conic-gradient(#3f3f46 0 25%, #27272a 0 50%) 0 0/8px 8px' }}>
                <input type="color" value={HEX_RE.test(hexDraft) ? hexDraft : '#808080'} onChange={e => setHex(e.target.value.toUpperCase())} className="absolute inset-0 h-full w-full cursor-pointer opacity-0" />
              </label>
              <div className="min-w-0 flex-1">
                <input value={hexDraft} onChange={e => setHex(e.target.value.startsWith('#') ? e.target.value : '#' + e.target.value)} placeholder="#RRGGBB" spellCheck={false}
                  className={`input !py-1.5 font-mono text-xs ${hexDraft && !HEX_RE.test(hexDraft) ? '!border-red-500/60' : ''}`} data-testid="hex-input" />
                <div className="mt-0.5 truncate text-[10px] text-zinc-500">{picked ? (picked.from === 'typed' ? 'typed' : `from ${picked.from}`) : 'no colour picked yet'}</div>
              </div>
            </div>

            {stage === 'pick' && (
              <div className="mt-3 space-y-2">
                <div className="text-[11px] text-zinc-400">Click any wall or object on the photo. The colour is the median of the whole surface, so shadows and highlights do not skew it.</div>
                {sources.length > 1 && (
                  <div>
                    <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">Pick from</div>
                    <div className="flex flex-wrap gap-1">
                      {sources.map(s => <button key={s.label} className={`chip !px-2 !py-0.5 ${sameSrc(s.src, pickSrc) ? '!border-amber-400 !text-amber-300' : ''}`} onClick={() => setPickSrc(s.src)}>{s.label}</button>)}
                    </div>
                  </div>
                )}
                {pickInfo && (
                  <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
                    <div className="flex items-center gap-2"><span className="h-5 w-5 rounded border border-zinc-600" style={{ background: pickInfo.hex }} /><span className="font-mono">{pickInfo.hex}</span><span className="text-zinc-500">{pickInfo.area_pct}% region</span></div>
                    <div className="mt-2 text-[10px] uppercase tracking-wide text-zinc-500">Variants</div>
                    <div className="mt-1 flex gap-1.5">
                      {[['Lit', pickInfo.hex_light], ['Surface', pickInfo.hex], ['Spot', pickInfo.hex_at_click], ['Shade', pickInfo.hex_shade]].map(([l, h]) => (
                        <button key={l} title={`${l}: ${h}`} onClick={() => setHex(h)} className={`flex flex-1 flex-col items-center gap-1 rounded-md border p-1 ${picked?.hex === h ? 'border-amber-400' : 'border-zinc-800 hover:border-zinc-600'}`}>
                          <span className="h-5 w-full rounded" style={{ background: h }} /><span className="text-[10px] text-zinc-400">{l}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <button className="btn btn-primary w-full justify-center !py-1.5" disabled={!picked} onClick={() => setStage('apply')}>Next: apply it →</button>
              </div>
            )}

            {stage === 'apply' && (
              <div className="mt-3 space-y-2">
                <div className="text-[11px] text-zinc-400">Click the wall or object to recolour. The highlight is what will change: click again to add a part it missed (the lit far end of a floor, a wall split by a doorway), or widen the band.</div>
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">Region</div>
                  <div className="grid grid-cols-3 rounded-lg border border-zinc-700 bg-zinc-950 p-0.5">
                    {(['auto', 'surface', 'object'] as RegionMode[]).map(m => <button key={m} className={`rounded-md py-1 capitalize ${regionMode === m ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`} onClick={() => setRegionMode(m)} title={m === 'surface' ? 'Walls, floors, lawns: colour growth bounded by edges' : m === 'object' ? 'Sofas, cabinets, doors: GrabCut around the click' : 'Surface when the region is large, object when it is small'}>{m}</button>)}
                  </div>
                </div>
                <label className="flex cursor-pointer items-center justify-between gap-2 rounded-lg border border-zinc-800 px-2 py-1.5 text-zinc-300" title="Extend the selection into parts of the surface that fade into strong light (the far end of a hallway floor, a wall beside a window). Follows gradual changes, still stops at edges.">
                  <span>Follow lighting <span className="text-zinc-500">· faded ends</span></span>
                  <input type="checkbox" checked={followLight} onChange={e => setFollowLight(e.target.checked)} className="accent-amber-400" data-testid="follow-light" />
                </label>
                <label className="block"><span className="flex justify-between text-zinc-400"><span>Colour band</span><span>{tolerance}</span></span>
                  <input type="range" min={10} max={100} value={tolerance} onChange={e => setTolerance(Number(e.target.value))} className="w-full accent-amber-400" /></label>
                <label className="block"><span className="flex justify-between text-zinc-400"><span>Colour strength</span><span>{strength}%</span></span>
                  <input type="range" min={10} max={100} value={strength} onChange={e => setStrength(Number(e.target.value))} className="w-full accent-amber-400" /></label>
                <label className="block"><span className="flex justify-between text-zinc-400"><span>Follow colour brightness</span><span>{lumFollow}%</span></span>
                  <input type="range" min={0} max={100} value={lumFollow} onChange={e => setLumFollow(Number(e.target.value))} className="w-full accent-amber-400" /></label>
                {region && <div className="flex items-center justify-between text-[11px] text-amber-200"><span>{region.area_pct}% of the frame selected · {region.mode_used}{region.points.length > 1 ? ` · ${region.points.length} clicks` : ''}</span><button className="text-zinc-400 hover:text-zinc-200" onClick={() => setRegion(null)}>Clear</button></div>}
                <button className="btn btn-primary w-full justify-center" disabled={!region || !picked || !!colorBusy} onClick={applyColor} data-testid="apply-color">
                  {colorBusy ? <Loader2 size={14} className="animate-spin" /> : <>Apply {picked?.hex ?? 'colour'} as v{photo.versions.length}</>}
                </button>
                {onSend && <button className="btn btn-ghost w-full justify-center !py-1.5" disabled={!region || !picked || !!colorBusy || chatBusy} onClick={askAssistant} title="Sends the clicked point and the colour to the assistant: it repaints with detection help and runs the QC check"><MessageSquareText size={13} /> Ask the assistant instead (adds QC)</button>}
              </div>
            )}
            {colorErr && <p role="alert" className="mt-2 rounded-lg border border-red-500/40 bg-red-500/10 px-2 py-1.5 text-[11px] text-red-200">{colorErr}</p>}
          </aside>
        )}
      </div>

      {/* version strip */}
      <div className="thin-scroll flex items-center gap-2 overflow-x-auto border-t border-zinc-800 px-3 py-2">
        {photo.versions.map(v => (
          <button key={v.id} onClick={() => revert(v)} disabled={busy} title={`${v.summary}${v.cost_usd ? ` · $${v.cost_usd.toFixed(2)}` : ''}${v.params?.qc?.reason ? `\nQC: ${v.params.qc.reason}` : ''}`}
            className={`relative shrink-0 overflow-hidden rounded-md border ${v.id === head.id ? 'border-amber-400' : 'border-zinc-800 hover:border-zinc-600'}`}>
            <img src={v.url} alt="" className="h-14 w-[84px] object-cover" loading="lazy" />
            {v.params?.qc?.verdict === 'needs_review' && <ShieldAlert size={11} className="absolute right-1 top-1 text-amber-300 drop-shadow" />}
            <div className="max-w-[84px] truncate bg-zinc-900 px-1 py-0.5 text-[10px] text-zinc-400">v{v.index} {v.index ? (TOOL_LABEL[v.tool] ?? v.tool.replace('_', ' ')) : 'original'}</div>
          </button>
        ))}
        {(photo.brackets.length > 0 || photo.references.length > 0) && (
          <div className="ml-auto flex shrink-0 items-center gap-1 pl-3">
            <span className="mr-1 text-[10px] text-zinc-500">refs</span>
            {photo.brackets.map((b, i) => <button key={b} title={`Original exposure ${i + 1} · click to pick a colour from it`} onClick={() => { enterMode('color', true); setStage('pick'); setPickSrc({ source: 'bracket', index: i }) }}><img src={b} alt="" className="h-10 w-[60px] rounded border border-zinc-800 object-cover opacity-70 hover:opacity-100" loading="lazy" /></button>)}
            {photo.references.map((r, i) => <button key={r} title={`Customer reference ${i + 1} · click to pick a colour from it`} onClick={() => { enterMode('color', true); setStage('pick'); setPickSrc({ source: 'reference', index: i }) }}><img src={r} alt="" className="h-10 w-[60px] rounded border border-amber-500/50 object-cover" loading="lazy" /></button>)}
          </div>
        )}
      </div>
    </div>
  )
}
