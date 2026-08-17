import cv2
import numpy as np
import easyocr
import torch

# EasyOCR uses one Latin-script recognizer for Vietnamese and English.
OCR_LANGUAGES = ["vi", "en"]
GPU_DEVICE = "cuda:0"
OCR_BATCH_SIZE = 8


class SubtitleOCR:
    """EasyOCR wrapper with lightweight OpenCV filtering."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in PyTorch. Install a CUDA-enabled PyTorch build before running GPU OCR."
            )
        self.engine = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=GPU_DEVICE,
            verbose=False,
        )

    def has_text(self, crop_img: np.ndarray) -> bool:
        """Reject crops that clearly contain no text before invoking OCR."""
        if crop_img is None or crop_img.size == 0:
            return False

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if height < 8 or width < 16:
            return False

        bright = cv2.inRange(gray, 180, 255)
        edges = cv2.Canny(gray, 80, 180)
        mask = cv2.bitwise_or(bright, edges)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
        )

        active_per_row = np.count_nonzero(mask, axis=1)
        row_threshold = max(8, int(width * 0.018))
        active_rows = np.count_nonzero(active_per_row >= row_threshold)
        required_rows = max(2, min(8, int(height * 0.06)))
        return active_rows >= required_rows

    @staticmethod
    def _prepare_for_recognition(crop_img: np.ndarray) -> np.ndarray:
        """Improve local contrast and resolution before recognition."""
        h, w = crop_img.shape[:2]
        large_img = cv2.resize(crop_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(large_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _assemble_reading_order(results, min_confidence=0.40) -> str:
        """Sort recognized text boxes from top to bottom and left to right."""
        candidates = []
        for box, text, score in results:
            if not text or score < min_confidence:
                continue
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            candidates.append(
                {
                    "text": text.strip(),
                    "left": min(xs),
                    "center_y": (min(ys) + max(ys)) / 2,
                    "height": max(1, max(ys) - min(ys)),
                }
            )

        candidates.sort(key=lambda item: item["center_y"])
        row_tolerance = max(
            12,
            (float(np.median([item["height"] for item in candidates])) * 0.8)
            if candidates
            else 12,
        )
        rows = []
        for item in candidates:
            if not rows or item["center_y"] - rows[-1]["center_y"] > row_tolerance:
                rows.append({"center_y": item["center_y"], "items": [item]})
            else:
                rows[-1]["items"].append(item)

        texts = []
        for row in rows:
            row["items"].sort(key=lambda item: item["left"])
            texts.extend(item["text"] for item in row["items"])
        return " ".join(texts).strip()

    def read_text_batch(
        self,
        crop_images: list[np.ndarray],
        min_confidence: float = 0.40,
    ) -> list[str]:
        """Recognize a batch of same-sized crops in one GPU call."""
        if not crop_images:
            return []
        try:
            images = [self._prepare_for_recognition(image) for image in crop_images]
            min_confidence = max(0.10, min(0.95, float(min_confidence)))
            results_per_image = self.engine.readtext_batched(
                images,
                batch_size=min(OCR_BATCH_SIZE, len(images)),
                detail=1,
                paragraph=False,
                text_threshold=min_confidence,
                low_text=max(0.10, min_confidence * 0.625),
                link_threshold=0.30,
            )
            return [
                self._assemble_reading_order(result, min_confidence)
                for result in results_per_image
            ]
        except Exception:
            return [""] * len(crop_images)

    def read_text(self, crop_img: np.ndarray, min_confidence: float = 0.40) -> str:
        """Recognize a single crop."""
        return self.read_text_batch([crop_img], min_confidence)[0]
