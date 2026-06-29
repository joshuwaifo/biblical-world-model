from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import renderer

router = APIRouter()


class RenderRequest(BaseModel):
    reference: Optional[str] = None   # "Isaiah 53:5"
    query: Optional[str] = None       # free-text: "servant bearing sin"
    top_k: int = 10


@router.post("/render")
def render(req: RenderRequest):
    if req.reference and req.query:
        raise HTTPException(400, "Provide either 'reference' or 'query', not both.")
    if not req.reference and not req.query:
        raise HTTPException(400, "Provide 'reference' or 'query'.")
    try:
        if req.reference:
            return renderer.render_by_reference(req.reference, top_k=req.top_k)
        return renderer.render_by_query(req.query, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(404, str(e))
