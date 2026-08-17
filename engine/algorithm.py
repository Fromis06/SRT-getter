import cv2
import sys
import time
import concurrent.futures
from models.ocr_engine import OCR_BATCH_SIZE, SubtitleOCR
from math_engine import MathEngine


class ProcessingCancelled(Exception):
    """Raised when the active job is cancelled."""


class SubtitleExtractor:
    def __init__(
        self,
        video_path,
        crop_box,
        step_sec=1.0,
        boundary_step=0.1,
        max_subtitle_duration=10.0,
        min_confidence=0.40,
        debug_timing=False,
        ocr=None,
        cancel_event=None,
    ):
        self.video_path = video_path
        self.crop_box = crop_box
        self.step_sec = step_sec
        self.boundary_step = boundary_step
        self.max_subtitle_duration = max_subtitle_duration
        self.min_confidence = min_confidence
        self.ocr = ocr or SubtitleOCR()
        self.cancel_event = cancel_event
        self.math = MathEngine()
        self.debug_timing = debug_timing
        self._timing = {"position": 0.0, "has_text": 0.0, "find_t_end": 0.0, "read_text": 0.0}
        self._timing_count = {"position": 0, "has_text": 0, "find_t_end": 0, "read_text": 0}

    def _check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ProcessingCancelled("Job was cancelled")

    def _log_stage(self, stage, elapsed):
        self._timing[stage] += elapsed
        self._timing_count[stage] += 1

    def _print_timing_summary(self):
        total = sum(self._timing.values())
        print("=== Processing timing summary (seconds) ===", file=sys.stderr)
        for stage, secs in self._timing.items():
            n = self._timing_count[stage]
            avg_ms = (secs / n * 1000) if n else 0
            print(f"  {stage:12s}: {secs:8.2f}s total | {n:5d} calls | {avg_ms:7.2f}ms average", file=sys.stderr)
        print(f"  {'TOTAL':12s}: {total:8.2f}s", file=sys.stderr)

    def _crop(self, frame):
        h, w = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.crop_box
        return frame[int(h*y_min):int(h*y_max), int(w*x_min):int(w*x_max)]

    def _find_t_end(self, cap, anchor_crop, start_time):
        """Find the subtitle end using sequential frame grabs."""
        t0 = time.perf_counter()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step_frames = max(1, int(fps * self.boundary_step))
        
        anchor_signature = self.math.compute_anchor_signature(anchor_crop)

        curr_time = start_time
        max_time = start_time + self.max_subtitle_duration
        
        while curr_time < max_time:
            self._check_cancelled()
            for _ in range(step_frames - 1):
                cap.grab()
                
            ret, frame = cap.read()
            if not ret: 
                break
                
            curr_time += self.boundary_step
            test_crop = self._crop(frame)
            
            if not self.math.is_subtitle_stable_precomputed(anchor_signature, test_crop):
                if self.debug_timing:
                    self._log_stage("find_t_end", time.perf_counter() - t0)
                return curr_time - self.boundary_step
        
        if self.debug_timing:
            self._log_stage("find_t_end", time.perf_counter() - t0)
        return curr_time

    def process(self, progress_callback=None):
        self._check_cancelled()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise Exception("Cannot open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
        
        curr_sec = 0.0
        last_decoded_sec = None
        results = []
        pending_ocr = []
        active_batch = None

        SEEK_THRESHOLD_SEC = 2.0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ocr_executor:
            def collect_active_batch():
                nonlocal active_batch
                if active_batch is None:
                    return

                texts = active_batch["future"].result()
                self._check_cancelled()
                if self.debug_timing:
                    self._log_stage("read_text", time.perf_counter() - active_batch["submitted_at"])

                for candidate, text in zip(active_batch["candidates"], texts):
                    if not text:
                        continue
                    sub = {
                        "start": round(candidate["start"], 2),
                        "end": round(candidate["end"], 2),
                        "text": text,
                    }
                    results.append(sub)
                    if progress_callback:
                        progress_callback(
                            min(100, (candidate["end"] / total_sec) * 100),
                            sub,
                        )
                active_batch = None

            def submit_pending_batch():
                nonlocal active_batch, pending_ocr
                collect_active_batch()
                if not pending_ocr:
                    return
                candidates = pending_ocr
                pending_ocr = []
                active_batch = {
                    "candidates": candidates,
                    "submitted_at": time.perf_counter(),
                    "future": ocr_executor.submit(
                        self.ocr.read_text_batch,
                        [candidate["crop"] for candidate in candidates],
                        self.min_confidence,
                    ),
                }

            while curr_sec < total_sec:
                self._check_cancelled()
                t_pos0 = time.perf_counter()
                if last_decoded_sec is None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000.0)
                else:
                    gap = curr_sec - last_decoded_sec
                    if 0 <= gap <= SEEK_THRESHOLD_SEC:
                        skip_frames = max(0, int(round(gap * fps)) - 1)
                        for _ in range(skip_frames):
                            cap.grab()
                    else:
                        cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000.0)

                ret, frame = cap.read()
                if self.debug_timing:
                    self._log_stage("position", time.perf_counter() - t_pos0)
                if not ret: break
                last_decoded_sec = curr_sec

                crop_img = self._crop(frame)

                t_ht0 = time.perf_counter()
                has_text = self.ocr.has_text(crop_img)
                if self.debug_timing:
                    self._log_stage("has_text", time.perf_counter() - t_ht0)

                if has_text:
                    t_start = curr_sec
                    t_end = self._find_t_end(cap, crop_img, curr_sec)
                    last_decoded_sec = t_end

                    pending_ocr.append(
                        {"start": t_start, "end": t_end, "crop": crop_img.copy()}
                    )
                    if len(pending_ocr) >= OCR_BATCH_SIZE:
                        submit_pending_batch()

                    curr_sec = t_end + 0.2
                    if progress_callback:
                        progress_callback(min(100, (curr_sec / total_sec) * 100), None)
                    continue

                curr_sec += self.step_sec
                if progress_callback:
                    progress_callback(min(100, (curr_sec / total_sec) * 100), None)

            submit_pending_batch()
            collect_active_batch()

        cap.release()
        results.sort(key=lambda sub: sub["start"])
        if self.debug_timing:
            self._print_timing_summary()
        return results
