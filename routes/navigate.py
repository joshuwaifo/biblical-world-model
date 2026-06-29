"""
POST /navigate  — oracle-guided search through any information space.

Accepts any content (text, base64-encoded image/audio/bytes) as the query.
Uses the Biblical world model geometry + oracle (GNN attention, alignment
scoring, novelty detection) to guide tool invocations toward the goal.
"""

import base64
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import embedder, navigator

router = APIRouter()


class NavigateRequest(BaseModel):
    query: str                              # text query or base64-encoded bytes
    modality: str = "text"                 # "text" | "image" | "audio" | "bytes"
    goal: str = ""                         # what we're looking for (text)
    max_steps: int = 5
    convergence_threshold: float = 0.85
    tool_preference: Optional[list[str]] = None   # e.g. ["web_search", "ollama"]


@router.post("/navigate")
def navigate(req: NavigateRequest):
    """
    Navigate through the information space toward a goal.

    The Biblical world model geometry guides which tools to invoke and
    how to interpret results.  All retrieved content is scored for
    alignment (truth consistency) and novelty by the oracle.
    """
    try:
        if req.modality in ("image", "audio", "video", "bytes"):
            raw = base64.b64decode(req.query)
            query_content = raw
        else:
            query_content = req.query

        result = navigator.navigate(
            query=query_content,
            query_modality=req.modality,
            goal=req.goal or req.query,
            max_steps=req.max_steps,
            convergence_threshold=req.convergence_threshold,
            tool_preference=req.tool_preference,
        )

        return {
            # ── Coordinates ───────────────────────────────────────────────
            "query_coordinates":    result.query_coordinates,
            "goal_coordinates":     result.goal_coordinates,
            "final_coordinates":    result.final_coordinates,
            "final_cosine_to_goal": result.final_cosine_to_goal,
            "converged":            result.converged,
            # ── Truth layer ───────────────────────────────────────────────
            "truth_layer_summary":   result.truth_layer_summary,
            # ── Factoid layer ─────────────────────────────────────────────
            "factoid_layer_summary": result.factoid_layer_summary,
            "synthesis":             result.synthesis,
            # ── RL diagnostics ────────────────────────────────────────────
            "rl_diagnostics":        result.rl_diagnostics,
            # ── Step trace ────────────────────────────────────────────────
            "steps": [
                {
                    "step":                    s.step_num,
                    "biblical_context":        s.biblical_context,
                    "tool":                    s.tool_used,
                    "tool_input":              s.tool_input,
                    "output_preview":          s.tool_output[:400],
                    "coordinates":             s.coordinates,
                    "cosine_to_goal":          s.cosine_to_goal,
                    "alignment":               s.alignment,
                    "novelty":                 s.novelty,
                    "action_confidence":       s.action_confidence,
                    "elapsed_s":               s.elapsed_s,
                    "outside_prior":           s.outside_prior,
                    "manifold_interpretation": s.manifold_interpretation,
                    "td_error":                s.td_error,
                    "library_hit":             s.library_hit,
                }
                for s in result.steps
            ],
            "retrieved_content": result.retrieved_content,
        }

    except Exception as e:
        raise HTTPException(500, str(e))
