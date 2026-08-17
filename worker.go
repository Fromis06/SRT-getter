package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

type WorkerMessage struct {
	Type           string     `json:"type"`
	Message        string     `json:"message,omitempty"`
	JobID          string     `json:"job_id,omitempty"`
	RequestID      string     `json:"request_id,omitempty"`
	Percent        int        `json:"percent,omitempty"`
	TotalSubtitles int        `json:"total_subtitles,omitempty"`
	CurrentSub     *Subtitle  `json:"current_sub,omitempty"`
	Data           []Subtitle `json:"data,omitempty"`
	Image          string     `json:"image,omitempty"`
	Width          int        `json:"width,omitempty"`
	Height         int        `json:"height,omitempty"`
	Duration       float64    `json:"duration,omitempty"`
	Timestamp      float64    `json:"timestamp,omitempty"`
}

type PythonWorker struct {
	onEvent func(WorkerMessage)

	mu      sync.RWMutex
	writeMu sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	ready   bool
	closed  bool
	done    chan struct{}
	pending map[string]chan WorkerMessage
}

func NewPythonWorker(onEvent func(WorkerMessage)) *PythonWorker {
	return &PythonWorker{
		onEvent: onEvent,
		done:    make(chan struct{}),
		pending: make(map[string]chan WorkerMessage),
	}
}

func (w *PythonWorker) Start() error {
	pythonPath, workerPath, err := resolveWorkerPaths()
	if err != nil {
		return err
	}

	cmd := exec.Command(pythonPath, "-u", workerPath)
	cmd.Dir = filepath.Dir(workerPath)
	cmd.Env = append(os.Environ(), "PYTHONIOENCODING=utf-8", "PYTHONUTF8=1")
	if runtime.GOOS == "windows" {
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("cannot create Python command channel: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("cannot read Python output: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("cannot read Python logs: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cannot start Python worker: %w", err)
	}

	w.mu.Lock()
	w.cmd = cmd
	w.stdin = stdin
	w.mu.Unlock()

	go w.readMessages(stdout)
	go w.readLogs(stderr)
	go func() {
		err := cmd.Wait()
		w.mu.Lock()
		wasClosed := w.closed
		w.ready = false
		w.closed = true
		w.mu.Unlock()
		if err != nil && !wasClosed && w.onEvent != nil {
			w.onEvent(WorkerMessage{Type: "fatal", Message: "Python worker stopped: " + err.Error()})
		}
		close(w.done)
	}()
	return nil
}

func (w *PythonWorker) readMessages(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 32*1024*1024)
	for scanner.Scan() {
		var message WorkerMessage
		if err := json.Unmarshal(scanner.Bytes(), &message); err != nil {
			if w.onEvent != nil {
				w.onEvent(WorkerMessage{Type: "protocol_error", Message: "Python returned invalid data"})
			}
			continue
		}

		if message.Type == "ready" {
			w.mu.Lock()
			w.ready = true
			w.mu.Unlock()
		}
		if message.Type == "fatal" {
			w.mu.Lock()
			w.ready = false
			w.mu.Unlock()
		}

		if message.RequestID != "" && (message.Type == "preview" || message.Type == "preview_error") {
			w.mu.Lock()
			channel := w.pending[message.RequestID]
			delete(w.pending, message.RequestID)
			w.mu.Unlock()
			if channel != nil {
				channel <- message
				close(channel)
			}
			continue
		}

		if w.onEvent != nil {
			w.onEvent(message)
		}
	}
	if err := scanner.Err(); err != nil && w.onEvent != nil {
		w.onEvent(WorkerMessage{Type: "protocol_error", Message: "Lost connection to Python worker: " + err.Error()})
	}
}

func (w *PythonWorker) readLogs(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 16*1024), 2*1024*1024)
	for scanner.Scan() {
		fmt.Printf("[python] %s\n", scanner.Text())
	}
}

func (w *PythonWorker) send(command any) error {
	w.mu.RLock()
	stdin := w.stdin
	closed := w.closed
	w.mu.RUnlock()
	if stdin == nil || closed {
		return errors.New("Python worker is not running")
	}

	payload, err := json.Marshal(command)
	if err != nil {
		return err
	}
	w.writeMu.Lock()
	defer w.writeMu.Unlock()
	_, err = stdin.Write(append(payload, '\n'))
	if err != nil {
		return fmt.Errorf("cannot send command to Python: %w", err)
	}
	return nil
}

func (w *PythonWorker) ensureReady() error {
	w.mu.RLock()
	ready := w.ready
	closed := w.closed
	w.mu.RUnlock()
	if closed {
		return errors.New("Python worker has stopped")
	}
	if !ready {
		return errors.New("OCR model is not ready")
	}
	return nil
}

func (w *PythonWorker) Preview(videoPath string, ratio float64) (FramePreview, error) {
	if err := w.ensureReady(); err != nil {
		return FramePreview{}, err
	}
	absoluteVideoPath, err := filepath.Abs(videoPath)
	if err != nil {
		return FramePreview{}, fmt.Errorf("cannot resolve video path: %w", err)
	}
	requestID := fmt.Sprintf("preview-%d", time.Now().UnixNano())
	response := make(chan WorkerMessage, 1)
	w.mu.Lock()
	w.pending[requestID] = response
	w.mu.Unlock()

	err = w.send(map[string]any{
		"type":       "preview",
		"request_id": requestID,
		"video":      absoluteVideoPath,
		"ratio":      ratio,
	})
	if err != nil {
		w.mu.Lock()
		delete(w.pending, requestID)
		w.mu.Unlock()
		return FramePreview{}, err
	}

	select {
	case message := <-response:
		if message.Type == "preview_error" {
			return FramePreview{}, errors.New(message.Message)
		}
		return FramePreview{
			Image: message.Image, Width: message.Width, Height: message.Height,
			Duration: message.Duration, Timestamp: message.Timestamp,
		}, nil
	case <-time.After(30 * time.Second):
		w.mu.Lock()
		delete(w.pending, requestID)
		w.mu.Unlock()
		return FramePreview{}, errors.New("timed out waiting for preview frame")
	}
}

func (w *PythonWorker) StartJob(jobID string, config JobConfig) error {
	if err := w.ensureReady(); err != nil {
		return err
	}
	absoluteVideoPath, err := filepath.Abs(config.VideoPath)
	if err != nil {
		return fmt.Errorf("cannot resolve video path: %w", err)
	}
	return w.send(map[string]any{
		"type":                  "start",
		"job_id":                jobID,
		"video":                 absoluteVideoPath,
		"crop":                  []float64{config.Crop.XMin, config.Crop.XMax, config.Crop.YMin, config.Crop.YMax},
		"step":                  config.ScanStep,
		"boundary_step":         config.BoundaryStep,
		"max_subtitle_duration": config.MaxSubtitleSeconds,
		"min_confidence":        config.MinConfidence,
		"debug_timing":          config.DebugTiming,
	})
}

func (w *PythonWorker) Cancel(jobID string) error {
	return w.send(map[string]any{"type": "cancel", "job_id": jobID})
}

func (w *PythonWorker) Shutdown() {
	w.mu.RLock()
	if w.closed {
		w.mu.RUnlock()
		return
	}
	cmd := w.cmd
	w.mu.RUnlock()

	_ = w.send(map[string]any{"type": "shutdown"})
	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()
	select {
	case <-w.done:
	case <-time.After(2 * time.Second):
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}
}

func resolveWorkerPaths() (string, string, error) {
	roots := candidateProjectRoots()

	pythonCandidates := make([]string, 0, len(roots)*2+1)
	if configured := strings.TrimSpace(os.Getenv("SUBTITLE_PYTHON")); configured != "" {
		pythonCandidates = append(pythonCandidates, configured)
	}
	for _, root := range roots {
		if runtime.GOOS == "windows" {
			pythonCandidates = append(pythonCandidates, filepath.Join(root, ".venv", "Scripts", "python.exe"))
		} else {
			pythonCandidates = append(pythonCandidates, filepath.Join(root, ".venv", "bin", "python"))
		}
	}
	pythonPath := firstExistingFile(pythonCandidates)
	if pythonPath == "" {
		return "", "", errors.New("Python was not found in .venv; set SUBTITLE_PYTHON to use another interpreter")
	}

	workerCandidates := make([]string, 0, len(roots))
	for _, root := range roots {
		workerCandidates = append(workerCandidates, filepath.Join(root, "engine", "worker.py"))
	}
	workerPath := firstExistingFile(workerCandidates)
	if workerPath == "" {
		return "", "", errors.New("engine/worker.py was not found")
	}
	return pythonPath, workerPath, nil
}

func candidateProjectRoots() []string {
	seen := map[string]bool{}
	var roots []string
	add := func(path string) {
		path, err := filepath.Abs(path)
		if err == nil && !seen[path] {
			seen[path] = true
			roots = append(roots, path)
		}
	}

	if current, err := os.Getwd(); err == nil {
		add(current)
	}
	if executable, err := os.Executable(); err == nil {
		dir := filepath.Dir(executable)
		add(dir)
		add(filepath.Dir(dir))
		add(filepath.Dir(filepath.Dir(dir)))
	}
	if _, source, _, ok := runtime.Caller(0); ok {
		add(filepath.Dir(source))
	}
	return roots
}

func firstExistingFile(candidates []string) string {
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}
