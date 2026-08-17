import cv2
import numpy as np

class MathEngine:
    @staticmethod
    def compute_anchor_signature(anchor_img: np.ndarray):
        """Build the reusable grayscale and edge signature for an anchor frame."""
        g_anchor = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g_anchor, 100, 200)
        anchor_edge_count = cv2.countNonZero(edges)
        return g_anchor, edges, anchor_edge_count

    @staticmethod
    def is_subtitle_stable_precomputed(anchor_signature, test_img: np.ndarray, diff_threshold=20) -> bool:
        """Compare a frame with a precomputed anchor signature."""
        g_anchor, edges, anchor_edge_count = anchor_signature

        if anchor_edge_count < 10:
            return False

        g_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g_anchor, g_test)
        _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)

        stable_text = cv2.bitwise_and(edges, static_mask)
        stable_count = cv2.countNonZero(stable_text)

        return stable_count > (anchor_edge_count * 0.4)

    @staticmethod
    def is_subtitle_stable(anchor_img: np.ndarray, test_img: np.ndarray, diff_threshold=20) -> bool:
        """Compare two frames without a precomputed signature."""
        anchor_signature = MathEngine.compute_anchor_signature(anchor_img)
        return MathEngine.is_subtitle_stable_precomputed(anchor_signature, test_img, diff_threshold)
