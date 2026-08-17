import { useEffect, useMemo, useState } from 'react'
import ReactCrop, { type PercentCrop } from 'react-image-crop'
import type { FramePreview, JobConfig, Subtitle, WorkerEvent, WorkerState } from './types'

const DEFAULT_CROP: PercentCrop = {
  unit: '%',
  x: 5,
  y: 68,
  width: 90,
  height: 27,
}

const DEFAULT_WORKER: WorkerState = {
  status: 'starting',
  message: 'Starting Python worker…',
}

function getAppApi() {
  const api = window.go?.main?.App
  if (!api) throw new Error('The Go application is not connected to the interface')
  return api
}

function readableError(error: unknown) {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return 'An unknown error occurred'
}

function formatClock(seconds: number) {
  if (!Number.isFinite(seconds)) return '00:00'
  const rounded = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const secs = rounded % 60
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

function fileName(path: string) {
  return path.split(/[\\/]/).pop() || path
}

function App() {
  const [worker, setWorker] = useState<WorkerState>(DEFAULT_WORKER)
  const [videoPath, setVideoPath] = useState('')
  const [preview, setPreview] = useState<FramePreview | null>(null)
  const [previewRatio, setPreviewRatio] = useState(0.68)
  const [crop, setCrop] = useState<PercentCrop>(DEFAULT_CROP)
  const [scanStep, setScanStep] = useState(1)
  const [boundaryStep, setBoundaryStep] = useState(0.1)
  const [maxSubtitleSeconds, setMaxSubtitleSeconds] = useState(10)
  const [minConfidence, setMinConfidence] = useState(0.4)
  const [running, setRunning] = useState(false)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [progress, setProgress] = useState(0)
  const [subtitles, setSubtitles] = useState<Subtitle[]>([])
  const [notice, setNotice] = useState('Choose a video to get started')
  const [error, setError] = useState('')

  useEffect(() => {
    getAppApi().GetWorkerState().then(setWorker).catch((err) => {
      setWorker({ status: 'error', message: readableError(err) })
    })

    const runtime = window.runtime
    if (!runtime) return

    const unsubscribers = [
      runtime.EventsOn('worker:status', (data) => setWorker(data as WorkerState)),
      runtime.EventsOn('job:started', () => {
        setRunning(true)
        setNotice('Scanning video and recognizing subtitles…')
      }),
      runtime.EventsOn('job:progress', (data) => {
        const event = data as WorkerEvent
        setProgress(event.percent ?? 0)
        if (event.current_sub) {
          setSubtitles((current) => {
            const exists = current.some(
              (item) => item.start === event.current_sub?.start && item.end === event.current_sub?.end,
            )
            return exists ? current : [...current, event.current_sub as Subtitle]
          })
        }
      }),
      runtime.EventsOn('job:done', (data) => {
        const event = data as WorkerEvent
        setRunning(false)
        setProgress(100)
        setSubtitles(event.data ?? [])
        setNotice(`Complete — found ${event.total_subtitles ?? 0} subtitle segments`)
      }),
      runtime.EventsOn('job:cancelled', () => {
        setRunning(false)
        setNotice('Processing stopped')
      }),
      runtime.EventsOn('job:error', (data) => {
        const event = data as WorkerEvent
        setRunning(false)
        setError(event.message || 'Unable to process video')
        setNotice('Processing failed')
      }),
    ]

    return () => unsubscribers.forEach((unsubscribe) => unsubscribe?.())
  }, [])

  const cropLabel = useMemo(() => {
    const x1 = Math.round(crop.x)
    const y1 = Math.round(crop.y)
    const x2 = Math.round(crop.x + crop.width)
    const y2 = Math.round(crop.y + crop.height)
    return `${x1}% · ${y1}%  →  ${x2}% · ${y2}%`
  }, [crop])

  async function loadPreview(path: string, ratio: number) {
    setLoadingPreview(true)
    setError('')
    try {
      const frame = await getAppApi().GetVideoPreview(path, ratio)
      setPreview(frame)
      setNotice('Drag the frame to select the subtitle area')
    } catch (err) {
      setError(readableError(err))
    } finally {
      setLoadingPreview(false)
    }
  }

  async function chooseVideo() {
    setError('')
    try {
      const path = await getAppApi().SelectVideo()
      if (!path) return
      setVideoPath(path)
      setCrop(DEFAULT_CROP)
      setSubtitles([])
      setProgress(0)
      await loadPreview(path, previewRatio)
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function refreshPreview() {
    if (videoPath) await loadPreview(videoPath, previewRatio)
  }

  async function startJob() {
    if (!videoPath || !preview) return
    setError('')
    setSubtitles([])
    setProgress(0)
    const config: JobConfig = {
      videoPath,
      crop: {
        xMin: crop.x / 100,
        xMax: (crop.x + crop.width) / 100,
        yMin: crop.y / 100,
        yMax: (crop.y + crop.height) / 100,
      },
      scanStep,
      boundaryStep,
      maxSubtitleSeconds,
      minConfidence,
      debugTiming: false,
    }
    try {
      setRunning(true)
      setNotice('Sending configuration to the OCR worker…')
      await getAppApi().StartJob(config)
    } catch (err) {
      setRunning(false)
      setError(readableError(err))
    }
  }

  async function cancelJob() {
    try {
      setNotice('Stopping after the current OCR operation…')
      await getAppApi().CancelJob()
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function saveSRT() {
    try {
      const path = await getAppApi().SaveSRT()
      if (path) setNotice(`Saved ${fileName(path)}`)
    } catch (err) {
      setError(readableError(err))
    }
  }

  const workerReady = worker.status === 'ready'
  const canStart = workerReady && Boolean(videoPath && preview && crop.width > 0 && crop.height > 0)

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span>CC</span>
          </div>
          <div>
            <h1>Subtitle Studio</h1>
            <p>Extract subtitles from video with OCR</p>
          </div>
        </div>

        <div className={`worker-pill worker-${worker.status}`}>
          <span className="status-dot" />
          <div>
            <strong>{workerReady ? 'OCR ready' : worker.status === 'error' ? 'OCR error' : 'Loading OCR'}</strong>
            <span>{worker.message}</span>
          </div>
        </div>
      </header>

      <section className="workspace-grid">
        <div className="preview-panel panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">01 · DETECTION AREA</span>
              <h2>Select the subtitle area</h2>
            </div>
            <button className="secondary-button" onClick={chooseVideo} disabled={!workerReady || running}>
              <FolderIcon />
              {videoPath ? 'Change video' : 'Choose video'}
            </button>
          </div>

          <div className={`preview-stage ${!preview ? 'preview-empty' : ''}`}>
            {preview ? (
              <ReactCrop
                crop={crop}
                onChange={(_, percentCrop) => setCrop(percentCrop)}
                minWidth={40}
                minHeight={24}
                keepSelection
                ruleOfThirds
                disabled={running}
              >
                <img src={preview.image} alt="Frame used to select the subtitle area" />
              </ReactCrop>
            ) : (
              <div className="empty-state">
                <div className="empty-icon"><FrameIcon /></div>
                <h3>{workerReady ? 'Add your first video' : 'Preparing OCR'}</h3>
                <p>
                  {workerReady
                    ? 'Choose a frame, then drag the selection around the area where subtitles appear.'
                    : 'The model loads once, then every video shares the same worker.'}
                </p>
                <button className="primary-button" onClick={chooseVideo} disabled={!workerReady}>
                  <FolderIcon /> Choose video
                </button>
              </div>
            )}
            {loadingPreview && <div className="preview-loading"><span className="spinner" /> Loading frame…</div>}
          </div>

          {preview && (
            <div className="frame-controls">
              <div className="video-meta">
                <strong title={videoPath}>{fileName(videoPath)}</strong>
                <span>{preview.width} × {preview.height} · {formatClock(preview.duration)}</span>
              </div>
              <div className="timeline-control">
                <div className="timeline-labels">
                  <span>Reference frame</span>
                  <strong>{formatClock(previewRatio * preview.duration)}</strong>
                </div>
                <div className="timeline-row">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={previewRatio}
                    onChange={(event) => setPreviewRatio(Number(event.target.value))}
                    disabled={running}
                  />
                  <button className="icon-button" onClick={refreshPreview} disabled={loadingPreview || running} title="Load the frame at this position">
                    <RefreshIcon />
                  </button>
                </div>
              </div>
              <div className="crop-readout">
                <span>Crop coordinates</span>
                <strong>{cropLabel}</strong>
              </div>
            </div>
          )}
        </div>

        <aside className="settings-panel panel">
          <div className="panel-heading compact">
            <div>
              <span className="eyebrow">02 · SETTINGS</span>
              <h2>Fine-tune the scan</h2>
            </div>
          </div>

          <Setting
            label="Scan interval"
            hint="Higher is faster but may miss short subtitles"
            value={`${scanStep.toFixed(2)}s`}
          >
            <input type="range" min="0.25" max="3" step="0.25" value={scanStep} onChange={(e) => setScanStep(Number(e.target.value))} disabled={running} />
          </Setting>

          <Setting
            label="Timing precision"
            hint="Step size used to find each subtitle end"
            value={`${boundaryStep.toFixed(2)}s`}
          >
            <input type="range" min="0.05" max="0.5" step="0.05" value={boundaryStep} onChange={(e) => setBoundaryStep(Number(e.target.value))} disabled={running} />
          </Setting>

          <Setting
            label="OCR confidence"
            hint="Increase to filter weak or noisy recognition"
            value={`${Math.round(minConfidence * 100)}%`}
          >
            <input type="range" min="0.2" max="0.8" step="0.05" value={minConfidence} onChange={(e) => setMinConfidence(Number(e.target.value))} disabled={running} />
          </Setting>

          <Setting
            label="Maximum subtitle length"
            hint="Maximum duration to search for one stable subtitle"
            value={`${maxSubtitleSeconds}s`}
          >
            <input type="range" min="3" max="30" step="1" value={maxSubtitleSeconds} onChange={(e) => setMaxSubtitleSeconds(Number(e.target.value))} disabled={running} />
          </Setting>

          <div className="action-zone">
            <div className="progress-copy">
              <div>
                <span>{running ? 'PROCESSING' : progress === 100 ? 'COMPLETE' : 'READY'}</span>
                <strong>{progress}%</strong>
              </div>
              <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
              <p>{notice}</p>
            </div>

            {running ? (
              <button className="danger-button full-button" onClick={cancelJob}>
                <StopIcon /> Stop processing
              </button>
            ) : (
              <button className="primary-button full-button" onClick={startJob} disabled={!canStart}>
                <PlayIcon /> Start extraction
              </button>
            )}

            <button className="secondary-button full-button" onClick={saveSRT} disabled={running || subtitles.length === 0}>
              <DownloadIcon /> Save SRT file
            </button>
          </div>
        </aside>
      </section>

      {error && (
        <div className="error-banner" role="alert">
          <span>!</span>
          <p>{error}</p>
          <button onClick={() => setError('')} aria-label="Dismiss notification">×</button>
        </div>
      )}

      <section className="results-panel panel">
        <div className="results-heading">
          <div>
            <span className="eyebrow">03 · RESULTS</span>
            <h2>Recognized subtitles</h2>
          </div>
          <span className="result-count">{subtitles.length} segments</span>
        </div>

        {subtitles.length > 0 ? (
          <div className="subtitle-list">
            {subtitles.map((subtitle, index) => (
              <article className="subtitle-row" key={`${subtitle.start}-${subtitle.end}-${index}`}>
                <span className="subtitle-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="subtitle-time">{formatClock(subtitle.start)} → {formatClock(subtitle.end)}</span>
                <p>{subtitle.text}</p>
              </article>
            ))}
          </div>
        ) : (
          <div className="results-empty">
            <WaveIcon />
            <p>Subtitles will appear here while OCR is running.</p>
          </div>
        )}
      </section>
    </main>
  )
}

function Setting({ label, hint, value, children }: { label: string; hint: string; value: string; children: React.ReactNode }) {
  return (
    <div className="setting-block">
      <div className="setting-title">
        <div><strong>{label}</strong><span>{hint}</span></div>
        <output>{value}</output>
      </div>
      {children}
    </div>
  )
}

const FolderIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5v-9Z" /></svg>
const FrameIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3v4H3M19 3v4h2M5 21v-4H3M19 21v-4h2M7 7h10v10H7z" /></svg>
const RefreshIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5M6.1 8.5A7 7 0 0 1 18 7l2 5M4 12l2 5a7 7 0 0 0 11.9-1.5" /></svg>
const PlayIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7V5Z" /></svg>
const StopIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7h10v10H7z" /></svg>
const DownloadIcon = () => <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M5 20h14" /></svg>
const WaveIcon = () => <svg viewBox="0 0 48 24" aria-hidden="true"><path d="M2 12h4l3-8 5 16 5-12 5 8 4-5 5 4 4-3h9" /></svg>

export default App
