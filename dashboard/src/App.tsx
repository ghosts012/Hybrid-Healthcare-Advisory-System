import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
  FileUp,
  History,
  LayoutDashboard,
  Settings,
  ShieldPlus,
} from 'lucide-react'

type NavKey = 'dashboard' | 'history' | 'settings'

type AnalyzeResponse = {
  is_pneumonia: boolean
  severity_score: number
  advisory: string
  status: string
}

type PredictSeverityResponse = {
  filename: string
  clinical_metrics: {
    pneumonia_probability: string
    severity_index: number
    status: string
  }
  rag_advisory: string
  disclaimer: string
}

function getApiBaseUrl(): string {
  const envBase = (import.meta as ImportMeta & { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL
  const resolved = (envBase && envBase.trim()) || 'http://127.0.0.1:8000'
  return resolved.endsWith('/') ? resolved.slice(0, -1) : resolved
}

function clampSeverity(n: number): number {
  if (Number.isNaN(n)) return 0
  return Math.max(0, Math.min(100, n))
}

function formatTodayDate(): string {
  return new Date().toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

function normalizeAdvisoryDate(advisory: string): string {
  const today = formatTodayDate()
  const hasDate = /\*\*Date:\*\*/i.test(advisory)
  if (hasDate) {
    return advisory.replace(/\*\*Date:\*\*.*$/im, `**Date:** ${today}`)
  }
  return `**Date:** ${today}\n\n${advisory}`
}

function normalizeApiResponse(json: unknown): AnalyzeResponse {
  const candidate = json as Partial<PredictSeverityResponse & AnalyzeResponse>

  // Primary shape from FastAPI /predict/severity
  if (
    candidate &&
    typeof candidate === 'object' &&
    candidate.clinical_metrics &&
    typeof candidate.clinical_metrics.severity_index === 'number'
  ) {
    const severity = clampSeverity(candidate.clinical_metrics.severity_index)
    const status = candidate.clinical_metrics.status ?? ''
    return {
      is_pneumonia: severity > 30 || /risk|pneumonia/i.test(status),
      severity_score: severity,
        advisory: normalizeAdvisoryDate(candidate.rag_advisory ?? ''),
      status,
    }
  }

  // Backward-compatible fallback shape
  if (
    candidate &&
    typeof candidate === 'object' &&
    typeof candidate.severity_score === 'number' &&
    typeof candidate.advisory === 'string'
  ) {
    return {
      is_pneumonia: Boolean(candidate.is_pneumonia),
      severity_score: clampSeverity(candidate.severity_score),
      advisory: normalizeAdvisoryDate(candidate.advisory),
      status: candidate.is_pneumonia ? 'Pneumonia suspected' : 'No pneumonia detected',
    }
  }

  throw new Error('Unexpected API response format.')
}

function severityColor(severity: number): {
  ring: string
  badge: string
  text: string
} {
  if (severity > 70) return { ring: 'stroke-rose-500', badge: 'bg-rose-500/15 text-rose-200', text: 'text-rose-200' }
  if (severity >= 30) return { ring: 'stroke-amber-400', badge: 'bg-amber-400/15 text-amber-100', text: 'text-amber-100' }
  return { ring: 'stroke-emerald-500', badge: 'bg-emerald-500/15 text-emerald-200', text: 'text-emerald-200' }
}

function Gauge({ value }: { value: number }) {
  const v = clampSeverity(value)
  const r = 44
  const c = 2 * Math.PI * r
  const dash = (v / 100) * c
  const colors = severityColor(v)

  return (
    <div className="relative grid place-items-center">
      <svg viewBox="0 0 120 120" className="h-36 w-36">
        <circle
          cx="60"
          cy="60"
          r={r}
          className="stroke-slate-700"
          strokeWidth="12"
          fill="none"
        />
        <circle
          cx="60"
          cy="60"
          r={r}
          className={`${colors.ring} drop-shadow`}
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c - dash}`}
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="absolute text-center">
        <div className="text-3xl font-semibold text-slate-100 tabular-nums">{Math.round(v)}</div>
        <div className="text-xs font-medium text-slate-400">Severity Index</div>
      </div>
    </div>
  )
}

function Paper({ children }: { children: React.ReactNode }) {
  return (
    <div className="paper relative overflow-hidden rounded-xl border border-slate-200/10 bg-slate-950/40 shadow-soft">
      <div className="pointer-events-none absolute inset-0 opacity-[0.08]" />
      <div className="relative p-5 sm:p-6">{children}</div>
    </div>
  )
}

export default function App() {
  const [nav, setNav] = useState<NavKey>('dashboard')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement | null>(null)

  const severity = useMemo(() => (result ? clampSeverity(result.severity_score) : 0), [result])
  const severityMeta = useMemo(() => severityColor(severity), [severity])
  const [isNotFound, setIsNotFound] = useState(false)

  useEffect(() => {
    const path = window.location.pathname.toLowerCase()
    if (path === '/' || path === '/dashboard') {
      setNav('dashboard')
      setIsNotFound(false)
      return
    }
    if (path === '/patient-history') {
      setNav('history')
      setIsNotFound(false)
      return
    }
    if (path === '/settings') {
      setNav('settings')
      setIsNotFound(false)
      return
    }
    setIsNotFound(true)
  }, [])

  async function analyze(selected: File) {
    setIsProcessing(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', selected)

      const base = getApiBaseUrl()
      const res = await fetch(`${base}/predict/severity`, { method: 'POST', body: form })
      if (!res.ok) {
        let message = `Request failed (${res.status})`
        try {
          const payload = (await res.json()) as { detail?: string }
          if (typeof payload?.detail === 'string' && payload.detail.trim()) {
            message = payload.detail
          }
        } catch {
          const text = await res.text().catch(() => '')
          if (text) message = text
        }
        throw new Error(message)
      }

      const json = await res.json()
      const normalized = normalizeApiResponse(json)
      setResult(normalized)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unexpected error'
      setError(msg)
    } finally {
      setIsProcessing(false)
    }
  }

  function acceptFile(selected: File | null) {
    if (!selected) return
    if (!/^image\/(png|jpe?g)$/i.test(selected.type)) {
      setError('Please upload a PNG or JPG image.')
      return
    }
    setFile(selected)
    setResult(null)
    setError(null)

    const url = URL.createObjectURL(selected)
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return url
    })
    void analyze(selected)
  }

  const navItems: Array<{ key: NavKey; label: string; icon: React.ReactNode }> = [
    { key: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="h-4 w-4" /> },
    { key: 'history', label: 'Patient History', icon: <History className="h-4 w-4" /> },
    { key: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
  ]

  return (
    <div className="min-h-dvh bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 sm:py-6">
        <div className="grid gap-4 lg:grid-cols-[280px_1fr] lg:gap-6">
          {/* Sidebar */}
          <aside className="rounded-2xl border border-slate-200/10 bg-slate-900/30 shadow-soft">
            <div className="flex items-center gap-3 border-b border-slate-200/10 px-4 py-4">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-500/15 text-indigo-200">
                <ShieldPlus className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold">Hybrid Healthcare</div>
                <div className="truncate text-xs text-slate-400">Advisory System</div>
              </div>
            </div>
            <nav className="p-2">
              {navItems.map((item) => {
                const active = nav === item.key
                return (
                  <button
                    key={item.key}
                    onClick={() => {
                      setNav(item.key)
                      setIsNotFound(false)
                      const path = item.key === 'dashboard' ? '/dashboard' : item.key === 'history' ? '/patient-history' : '/settings'
                      window.history.pushState({}, '', path)
                    }}
                    className={[
                      'flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm transition',
                      active
                        ? 'bg-slate-50/10 text-slate-50'
                        : 'text-slate-300 hover:bg-slate-50/5 hover:text-slate-50',
                    ].join(' ')}
                  >
                    <span className={active ? 'text-indigo-200' : 'text-slate-400'}>{item.icon}</span>
                    <span className="truncate">{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </aside>

          {/* Main */}
          <main className="space-y-6">
            {isNotFound ? (
              <section className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-8 shadow-soft">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Page not found</h1>
                <p className="mt-2 text-sm text-slate-400">
                  The page you tried to open is not available in this dashboard yet.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setNav('dashboard')
                    setIsNotFound(false)
                    window.history.pushState({}, '', '/dashboard')
                  }}
                  className="mt-5 inline-flex items-center justify-center rounded-xl bg-indigo-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-400"
                >
                  Go to Dashboard
                </button>
              </section>
            ) : nav === 'history' ? (
              <section className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-8 shadow-soft">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Patient History</h1>
                <p className="mt-2 text-sm text-slate-400">
                  This module is in progress. Soon you will be able to review prior analyses, compare severity trends, and access archived advisory reports.
                </p>
              </section>
            ) : nav === 'settings' ? (
              <section className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-8 shadow-soft">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Settings</h1>
                <p className="mt-2 text-sm text-slate-400">
                  Configuration tools are in progress. Upcoming options include API configuration, report preferences, and clinician-facing display controls.
                </p>
              </section>
            ) : (
              <>
            <header className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="text-xs font-medium tracking-wide text-slate-400">Medical AI Dashboard</div>
                <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">Chest X-ray Pneumonia Analysis</h1>
                <p className="mt-1 max-w-2xl text-sm text-slate-400">
                  Upload an X-ray (PNG/JPG). The system returns a Severity Index (0–100) and a grounded clinical advisory.
                </p>
              </div>
            </header>

            {/* Upload */}
            <section className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-4 shadow-soft sm:p-5">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  <div className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-500/15 text-indigo-200">
                    <FileUp className="h-5 w-5" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">Upload Chest X-ray</div>
                    <div className="text-xs text-slate-400">Drag & drop or browse. PNG/JPG only.</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/png,image/jpeg"
                    className="hidden"
                    onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
                  />
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="inline-flex items-center justify-center rounded-xl bg-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow-soft transition hover:bg-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-400/60 focus:ring-offset-2 focus:ring-offset-slate-950"
                  >
                    Browse files
                  </button>
                </div>
              </div>

              <div
                className={[
                  'mt-4 rounded-2xl border border-dashed p-5 transition sm:p-6',
                  isDragging ? 'border-indigo-400/70 bg-indigo-500/10' : 'border-slate-200/10 bg-slate-950/20',
                ].join(' ')}
                onDragOver={(e) => {
                  e.preventDefault()
                  setIsDragging(true)
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setIsDragging(false)
                  acceptFile(e.dataTransfer.files?.[0] ?? null)
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
                }}
              >
                <div className="flex flex-col items-center justify-center gap-2 text-center">
                  <div className="text-sm font-medium text-slate-200">
                    {file ? 'File selected' : 'Drop your image here'}
                  </div>
                  <div className="text-xs text-slate-400">
                    {file ? `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB` : 'Your image stays local until you submit to the API.'}
                  </div>
                </div>
              </div>

              {error ? (
                <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-100">
                  {error}
                </div>
              ) : null}
            </section>

            {/* Analysis view */}
            <section className="grid gap-6 lg:grid-cols-2">
              {/* Left: image */}
              <div className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-4 shadow-soft sm:p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold">X-ray Preview</div>
                    <div className="text-xs text-slate-400">Uploaded image with processing overlay.</div>
                  </div>
                  {result ? (
                    <span className={['rounded-full px-3 py-1 text-xs font-semibold', severityMeta.badge].join(' ')}>
                      {result.status || (result.is_pneumonia ? 'Pneumonia suspected' : 'No pneumonia detected')}
                    </span>
                  ) : (
                    <span className="rounded-full bg-slate-50/5 px-3 py-1 text-xs font-medium text-slate-300">
                      Awaiting analysis
                    </span>
                  )}
                </div>

                <div className="relative mt-4 overflow-hidden rounded-2xl border border-slate-200/10 bg-slate-950/30">
                  <div className="aspect-[4/3] w-full">
                    {previewUrl ? (
                      <img src={previewUrl} alt="Uploaded chest X-ray" className="h-full w-full object-contain" />
                    ) : (
                      <div className="grid h-full place-items-center p-6 text-center">
                        <div>
                          <div className="text-sm font-semibold text-slate-200">No image yet</div>
                          <div className="mt-1 text-xs text-slate-400">Upload a PNG/JPG chest X-ray to begin.</div>
                        </div>
                      </div>
                    )}
                  </div>

                  {isProcessing ? (
                    <div className="absolute inset-0 grid place-items-center bg-slate-950/55 backdrop-blur-sm">
                      <div className="flex items-center gap-3 rounded-2xl border border-slate-200/10 bg-slate-950/60 px-4 py-3 shadow-soft">
                        <div className="h-3 w-3 animate-pulse rounded-full bg-indigo-400" />
                        <div className="text-sm font-medium text-slate-100">Processing…</div>
                        <div className="text-xs text-slate-400">Running ONNX + RAG</div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              {/* Right: severity */}
              <div className="rounded-2xl border border-slate-200/10 bg-slate-900/20 p-4 shadow-soft sm:p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-sm font-semibold">Clinical Severity Index</div>
                    <div className="text-xs text-slate-400">0–100 gauge with clinical risk coloring.</div>
                  </div>
                  <span className="rounded-full bg-slate-50/5 px-3 py-1 text-xs font-medium text-slate-300">
                    {severity < 30 ? 'Low risk' : severity <= 70 ? 'Moderate risk' : 'High risk'}
                  </span>
                </div>

                <div className="mt-6 grid place-items-center">
                  <Gauge value={severity} />
                </div>

                <div className="mt-6 grid grid-cols-3 gap-3">
                  <div className="rounded-xl border border-slate-200/10 bg-slate-950/20 p-3">
                    <div className="text-xs text-slate-400">Threshold</div>
                    <div className="mt-1 text-sm font-semibold text-emerald-200">&lt; 30</div>
                  </div>
                  <div className="rounded-xl border border-slate-200/10 bg-slate-950/20 p-3">
                    <div className="text-xs text-slate-400">Threshold</div>
                    <div className="mt-1 text-sm font-semibold text-amber-100">30–70</div>
                  </div>
                  <div className="rounded-xl border border-slate-200/10 bg-slate-950/20 p-3">
                    <div className="text-xs text-slate-400">Threshold</div>
                    <div className="mt-1 text-sm font-semibold text-rose-200">&gt; 70</div>
                  </div>
                </div>

                <div className="mt-4 text-xs text-slate-500">
                  The severity index is a probabilistic score; always correlate with clinical context and radiologist review.
                </div>
              </div>
            </section>

            {/* RAG advisory */}
            <section>
              <Paper>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-100">RAG Clinical Advisory</div>
                  </div>
                </div>

                <div className="prose prose-invert prose-slate mt-4 max-w-none">
                  {result?.advisory ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {result.advisory}
                    </ReactMarkdown>
                  ) : (
                    <div className="rounded-xl border border-slate-200/10 bg-slate-950/20 p-4 text-sm text-slate-300">
                      Upload an image to generate a clinical advisory report.
                    </div>
                  )}
                </div>

                {result?.advisory ? (
                  <div className="mt-4 space-y-1.5 rounded-xl border border-slate-200/10 bg-slate-950/30 px-3 py-2.5 text-xs leading-relaxed text-slate-400">
                    <div>
                      <span className="font-medium text-slate-300">Clinical source:</span>{' '}
                      © World Health Organization 2025.{' '}
                      <a
                        href="https://www.who.int/publications/i/item/9789240103412"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-indigo-300 underline decoration-indigo-300/40 underline-offset-2 transition hover:text-indigo-200"
                      >
                        Guideline on management of pneumonia and diarrhoea in children up to 10 years of age
                      </a>
                      . Available under{' '}
                      <a
                        href="https://creativecommons.org/licenses/by-nc-sa/3.0/igo/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-indigo-300 underline decoration-indigo-300/40 underline-offset-2 transition hover:text-indigo-200"
                      >
                        CC BY-NC-SA 3.0 IGO
                      </a>
                      .
                    </div>
                    <div>
                      Source URL:{' '}
                      <a
                        href="https://www.who.int/publications/i/item/9789240103412"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="break-all text-indigo-300 underline decoration-indigo-300/40 underline-offset-2 transition hover:text-indigo-200"
                      >
                        https://www.who.int/publications/i/item/9789240103412
                      </a>
                      {' '}(accessed 15 March 2026). Local reference:{' '}
                      <code className="text-slate-300">medical_reference_who.pdf</code>
                    </div>
                  </div>
                ) : null}

                <div className="mt-4 text-xs text-slate-500">
                  Disclaimer: This dashboard provides AI-generated advisory content for clinical support only and is not a diagnosis.
                </div>
              </Paper>
            </section>
            </>
            )}
          </main>
        </div>
      </div>
    </div>
  )
}

