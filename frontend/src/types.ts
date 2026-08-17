export type WorkerState = {
  status: 'starting' | 'loading' | 'ready' | 'error'
  message: string
}

export type Subtitle = {
  start: number
  end: number
  text: string
}

export type FramePreview = {
  image: string
  width: number
  height: number
  duration: number
  timestamp: number
}

export type WorkerEvent = {
  type: string
  message?: string
  job_id?: string
  percent?: number
  total_subtitles?: number
  current_sub?: Subtitle
  data?: Subtitle[]
}

export type JobConfig = {
  videoPath: string
  crop: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }
  scanStep: number
  boundaryStep: number
  maxSubtitleSeconds: number
  minConfidence: number
  debugTiming: boolean
}
