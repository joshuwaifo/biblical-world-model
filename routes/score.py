"""
POST /score  — oracle scoring for any content stream.

Uses the trained BibleGAT (same weights, no retraining) to score:
  alignment  [0,1]  — reconstruction likelihood: how consistent is this
                       with the Biblical world model?
  novelty    [0,1]  — flipped logit: how outside the known manifold?
  coordinates       — 8-dim position in Biblical space
  nearest_concepts  — closest Bible verses at this position

Accepts any modality: text body, base64-encoded image/audio, raw bytes.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import embedder, oracle

router = APIRouter()


class ScoreRequest(BaseModel):
    content: str                        # text, or base64-encoded bytes
    modality: str = "auto"             # "text" | "image" | "audio" | "bytes" | "auto"
    k_neighbors: int = 5


@router.post("/score")
def score_content(req: ScoreRequest):
    """
    Score any content against the Biblical world model.

    Returns alignment (truth/consistency with the learned manifold),
    novelty (how far outside the known information space this is), and
    the Biblical coordinate position.
    """
    try:
        if req.modality in ("image", "audio", "video", "bytes"):
            import base64
            raw = base64.b64decode(req.content)
            vec = embedder.encode_bytes(raw, req.modality)
        else:
            vec = embedder.encode(req.content, req.modality)
    except Exception as e:
        raise HTTPException(400, f"Encoding failed: {e}")

    try:
        result = oracle.score(vec, k_neighbors=req.k_neighbors)
        result["coordinates"] = [round(float(x), 5) for x in vec]
        return result
    except Exception as e:
        raise HTTPException(500, f"Oracle scoring failed: {e}")
