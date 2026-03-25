"""Face embedding via InsightFace buffalo_l on a YOLO26x crop or full frame.

InsightFace: https://www.insightface.ai/
Model: buffalo_l — best open-source face recognition model.
Install: pip install '.[receptionist]'  (insightface + onnxruntime)

Pipeline
--------
1. YOLO26x detects the face/person bounding box in the camera frame.
2. ``embed_frame_with_bbox`` crops that region (+ 15% margin).
3. InsightFace runs its own internal face detector on the crop, then computes
   a 512-d L2-normalised embedding used for cosine similarity matching.
"""

from __future__ import annotations
import logging

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Compute normalized face embeddings using InsightFace recognition on a face crop."""

    def __init__(self, model_name: str = "buffalo_l", device_id: int = -1) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "InsightFace is required for receptionist face embeddings. "
                "Install with: pip install '.[receptionist]'",
            ) from e

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name=model_name, providers=providers)
        ctx = 0 if device_id >= 0 else -1
        self._app.prepare(ctx_id=ctx, det_size=(640, 640))
        logger.info("InsightFace model %s ready (ctx_id=%s)", model_name, ctx)

    def embed_crop(self, bgr_crop: NDArray[np.uint8]) -> np.ndarray | None:
        """Run detection + recognition on a crop; return first/largest face embedding."""
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        try:
            faces = self._app.get(bgr_crop)
        except Exception as e:
            logger.warning("InsightFace get() failed: %s", e)
            return None
        if not faces:
            return None
        # Largest by bbox area
        best = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        emb = np.asarray(best.embedding, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 1e-12:
            emb = emb / n
        return emb

    def embed_frame_with_bbox(
        self,
        frame_bgr: NDArray[np.uint8],
        xyxy: NDArray[np.float32] | None,
        margin: float = 0.15,
    ) -> np.ndarray | None:
        """Crop frame using xyxy (YOLO-style); optional relative margin expansion."""
        if xyxy is None:
            return self.embed_crop(frame_bgr)
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(x) for x in xyxy]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        mx = bw * margin
        my = bh * margin
        x1 = int(max(0, x1 - mx))
        y1 = int(max(0, y1 - my))
        x2 = int(min(w - 1, x2 + mx))
        y2 = int(min(h - 1, y2 + my))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        return self.embed_crop(crop)


def try_face_embedder(model_name: str) -> FaceEmbedder | None:
    try:
        return FaceEmbedder(model_name=model_name)
    except ImportError:
        return None
    except Exception as e:
        logger.error("Failed to initialize FaceEmbedder: %s", e)
        return None
