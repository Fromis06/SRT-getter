package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	wailsruntime "github.com/wailsapp/wails/v2/pkg/runtime"
)

type CropBox struct {
	XMin float64 `json:"xMin"`
	XMax float64 `json:"xMax"`
	YMin float64 `json:"yMin"`
	YMax float64 `json:"yMax"`
}

type JobConfig struct {
	VideoPath          string  `json:"videoPath"`
	Crop               CropBox `json:"crop"`
	ScanStep           float64 `json:"scanStep"`
	BoundaryStep       float64 `json:"boundaryStep"`
	MaxSubtitleSeconds float64 `json:"maxSubtitleSeconds"`
	MinConfidence      float64 `json:"minConfidence"`
	DebugTiming        bool    `json:"debugTiming"`
}

type Subtitle struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type FramePreview struct {
	Image     string  `json:"image"`
	Width     int     `json:"width"`
	Height    int     `json:"height"`
	Duration  float64 `json:"duration"`
	Timestamp float64 `json:"timestamp"`
}

type WorkerState struct {
	Status  string `json:"status"`
	Message string `json:"message"`
}

type App struct {
	ctx    context.Context
	worker *PythonWorker

	mu              sync.RWMutex
	activeJobID     string
	lastVideoPath   string
	lastSubtitles   []Subtitle
	lastWorkerState WorkerState
}

func NewApp() *App {
	return &App{
		lastWorkerState: WorkerState{
			Status:  "starting",
			Message: "Starting Python worker…",
		},
	}
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.worker = NewPythonWorker(a.handleWorkerEvent)
	if err := a.worker.Start(); err != nil {
		a.setWorkerState("error", err.Error())
	}
}

func (a *App) shutdown(context.Context) {
	if a.worker != nil {
		a.worker.Shutdown()
	}
}

func (a *App) handleWorkerEvent(event WorkerMessage) {
	switch event.Type {
	case "loading":
		a.setWorkerState("loading", event.Message)
	case "ready":
		a.setWorkerState("ready", event.Message)
	case "fatal":
		a.setWorkerState("error", event.Message)
	case "started":
		wailsruntime.EventsEmit(a.ctx, "job:started", event)
	case "progress":
		wailsruntime.EventsEmit(a.ctx, "job:progress", event)
	case "done":
		a.mu.Lock()
		a.activeJobID = ""
		a.lastSubtitles = append([]Subtitle(nil), event.Data...)
		a.mu.Unlock()
		wailsruntime.EventsEmit(a.ctx, "job:done", event)
	case "cancelled":
		a.mu.Lock()
		a.activeJobID = ""
		a.mu.Unlock()
		wailsruntime.EventsEmit(a.ctx, "job:cancelled", event)
	case "error":
		a.mu.Lock()
		a.activeJobID = ""
		a.mu.Unlock()
		wailsruntime.EventsEmit(a.ctx, "job:error", event)
	case "protocol_error":
		wailsruntime.EventsEmit(a.ctx, "worker:log", event)
	}
}

func (a *App) setWorkerState(status, message string) {
	state := WorkerState{Status: status, Message: message}
	a.mu.Lock()
	a.lastWorkerState = state
	a.mu.Unlock()
	if a.ctx != nil {
		wailsruntime.EventsEmit(a.ctx, "worker:status", state)
	}
}

func (a *App) GetWorkerState() WorkerState {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.lastWorkerState
}

func (a *App) SelectVideo() (string, error) {
	return wailsruntime.OpenFileDialog(a.ctx, wailsruntime.OpenDialogOptions{
		Title: "Choose a video to extract subtitles",
		Filters: []wailsruntime.FileFilter{
			{DisplayName: "Video", Pattern: "*.mp4;*.mkv;*.avi;*.mov;*.webm;*.m4v"},
			{DisplayName: "All files", Pattern: "*.*"},
		},
	})
}

func (a *App) GetVideoPreview(videoPath string, ratio float64) (FramePreview, error) {
	if a.worker == nil {
		return FramePreview{}, errors.New("Python worker has not started")
	}
	if strings.TrimSpace(videoPath) == "" {
		return FramePreview{}, errors.New("no video selected")
	}
	if ratio < 0 || ratio > 1 {
		return FramePreview{}, errors.New("invalid preview position")
	}
	return a.worker.Preview(videoPath, ratio)
}

func (a *App) StartJob(config JobConfig) (string, error) {
	if a.worker == nil {
		return "", errors.New("Python worker has not started")
	}
	if err := validateJobConfig(config); err != nil {
		return "", err
	}

	a.mu.Lock()
	if a.activeJobID != "" {
		a.mu.Unlock()
		return "", errors.New("another video is already being processed")
	}
	jobID := fmt.Sprintf("job-%d", time.Now().UnixNano())
	a.activeJobID = jobID
	a.lastVideoPath = config.VideoPath
	a.lastSubtitles = nil
	a.mu.Unlock()

	if err := a.worker.StartJob(jobID, config); err != nil {
		a.mu.Lock()
		a.activeJobID = ""
		a.mu.Unlock()
		return "", err
	}
	return jobID, nil
}

func validateJobConfig(config JobConfig) error {
	info, err := os.Stat(config.VideoPath)
	if err != nil || info.IsDir() {
		return errors.New("video file not found")
	}
	crop := config.Crop
	if !(crop.XMin >= 0 && crop.XMin < crop.XMax && crop.XMax <= 1 &&
		crop.YMin >= 0 && crop.YMin < crop.YMax && crop.YMax <= 1) {
		return errors.New("invalid crop area")
	}
	if config.ScanStep < 0.1 || config.ScanStep > 10 {
		return errors.New("scan interval must be between 0.1 and 10 seconds")
	}
	if config.BoundaryStep < 0.02 || config.BoundaryStep > 1 {
		return errors.New("boundary precision must be between 0.02 and 1 second")
	}
	if config.MaxSubtitleSeconds < 1 || config.MaxSubtitleSeconds > 60 {
		return errors.New("maximum subtitle duration must be between 1 and 60 seconds")
	}
	if config.MinConfidence < 0.1 || config.MinConfidence > 0.95 {
		return errors.New("OCR confidence must be between 0.1 and 0.95")
	}
	return nil
}

func (a *App) CancelJob() error {
	if a.worker == nil {
		return errors.New("Python worker has not started")
	}
	a.mu.RLock()
	jobID := a.activeJobID
	a.mu.RUnlock()
	if jobID == "" {
		return errors.New("no job is running")
	}
	return a.worker.Cancel(jobID)
}

func (a *App) SaveSRT() (string, error) {
	a.mu.RLock()
	subtitles := append([]Subtitle(nil), a.lastSubtitles...)
	videoPath := a.lastVideoPath
	a.mu.RUnlock()
	if len(subtitles) == 0 {
		return "", errors.New("no subtitles available to save")
	}

	defaultName := "subtitles.srt"
	if videoPath != "" {
		base := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
		defaultName = base + ".srt"
	}
	path, err := wailsruntime.SaveFileDialog(a.ctx, wailsruntime.SaveDialogOptions{
		Title:           "Save SRT subtitles",
		DefaultFilename: defaultName,
		Filters: []wailsruntime.FileFilter{
			{DisplayName: "SubRip subtitle", Pattern: "*.srt"},
		},
	})
	if err != nil || path == "" {
		return path, err
	}
	if !strings.EqualFold(filepath.Ext(path), ".srt") {
		path += ".srt"
	}
	if err := os.WriteFile(path, []byte(formatSRT(subtitles)), 0o644); err != nil {
		return "", err
	}
	return path, nil
}
