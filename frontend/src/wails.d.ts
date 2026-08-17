import type { FramePreview, JobConfig, WorkerState } from './types'

declare global {
  interface Window {
    go?: {
      main: {
        App: {
          GetWorkerState(): Promise<WorkerState>
          SelectVideo(): Promise<string>
          GetVideoPreview(videoPath: string, ratio: number): Promise<FramePreview>
          StartJob(config: JobConfig): Promise<string>
          CancelJob(): Promise<void>
          SaveSRT(): Promise<string>
        }
      }
    }
    runtime?: {
      EventsOn(eventName: string, callback: (...data: unknown[]) => void): () => void
      EventsOff(eventName: string, ...additionalEventNames: string[]): void
    }
  }
}

export {}
