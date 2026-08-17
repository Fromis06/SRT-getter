"""Persistent JSON-lines worker used by the Go desktop application.

Stdout is reserved for protocol messages. Diagnostics always go to stderr.
"""

import base64
import json
import os
import sys
import threading
import traceback
import uuid

import cv2

from algorithm import ProcessingCancelled, SubtitleExtractor
from models.ocr_engine import SubtitleOCR


_write_lock = threading.Lock()
_job_lock = threading.Lock()
_current_job = None


def emit(message_type, **payload):
    message = {"type": message_type, **payload}
    with _write_lock:
        print(json.dumps(message, ensure_ascii=False), flush=True)


def validate_crop(crop):
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError("crop must contain [x_min, x_max, y_min, y_max]")
    values = tuple(float(value) for value in crop)
    x_min, x_max, y_min, y_max = values
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        raise ValueError("crop coordinates must be normalized between 0 and 1")
    return values


def create_preview(video_path, ratio):
    if not os.path.isfile(video_path):
        raise FileNotFoundError("Video file does not exist")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if frame_count > 0 else 0.0
        ratio = max(0.0, min(1.0, float(ratio)))
        timestamp = ratio * max(0.0, duration - (1.0 / fps))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Cannot decode preview frame")

        height, width = frame.shape[:2]
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("Cannot encode preview frame")

        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
            "width": width,
            "height": height,
            "duration": round(duration, 3),
            "timestamp": round(timestamp, 3),
        }
    finally:
        cap.release()


def run_job(command, ocr, cancel_event):
    global _current_job
    job_id = command.get("job_id") or str(uuid.uuid4())

    try:
        extractor = SubtitleExtractor(
            video_path=command["video"],
            crop_box=validate_crop(command["crop"]),
            step_sec=float(command.get("step", 1.0)),
            boundary_step=float(command.get("boundary_step", 0.1)),
            max_subtitle_duration=float(command.get("max_subtitle_duration", 10.0)),
            min_confidence=float(command.get("min_confidence", 0.40)),
            debug_timing=bool(command.get("debug_timing", False)),
            ocr=ocr,
            cancel_event=cancel_event,
        )

        emit("started", job_id=job_id)
        last_percent = 0

        def on_progress(percent, current_sub):
            nonlocal last_percent
            last_percent = max(last_percent, min(100, int(percent)))
            emit(
                "progress",
                job_id=job_id,
                percent=last_percent,
                current_sub=current_sub,
            )

        results = extractor.process(progress_callback=on_progress)
        emit(
            "done",
            job_id=job_id,
            percent=100,
            total_subtitles=len(results),
            data=results,
        )
    except ProcessingCancelled:
        emit("cancelled", job_id=job_id)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", job_id=job_id, message=str(exc))
    finally:
        with _job_lock:
            _current_job = None


def main():
    global _current_job

    try:
        emit("loading", message="Loading OCR model into GPU…")
        ocr = SubtitleOCR()
        emit("ready", message="OCR model is ready")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("fatal", message=str(exc))
        return 1

    for raw_line in sys.stdin:
        try:
            command = json.loads(raw_line)
            command_type = command.get("type")

            if command_type == "preview":
                with _job_lock:
                    busy = _current_job is not None
                if busy:
                    emit(
                        "preview_error",
                        request_id=command.get("request_id"),
                        message="Worker is busy processing a video",
                    )
                    continue
                try:
                    preview = create_preview(command["video"], command.get("ratio", 0.5))
                    emit("preview", request_id=command.get("request_id"), **preview)
                except Exception as exc:
                    emit(
                        "preview_error",
                        request_id=command.get("request_id"),
                        message=str(exc),
                    )

            elif command_type == "start":
                with _job_lock:
                    if _current_job is not None:
                        emit(
                            "error",
                            job_id=command.get("job_id"),
                            message="Another job is already running",
                        )
                        continue
                    cancel_event = threading.Event()
                    thread = threading.Thread(
                        target=run_job,
                        args=(command, ocr, cancel_event),
                        daemon=True,
                    )
                    _current_job = {
                        "job_id": command.get("job_id"),
                        "cancel": cancel_event,
                        "thread": thread,
                    }
                    thread.start()

            elif command_type == "cancel":
                with _job_lock:
                    current = _current_job
                    if current is not None and (
                        not command.get("job_id")
                        or command.get("job_id") == current["job_id"]
                    ):
                        current["cancel"].set()

            elif command_type == "shutdown":
                with _job_lock:
                    current = _current_job
                    if current is not None:
                        current["cancel"].set()
                return 0
            else:
                emit("protocol_error", message=f"Unknown command: {command_type}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("protocol_error", message=str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
