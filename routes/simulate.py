from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import simulator

router = APIRouter()


class SimulateRequest(BaseModel):
    period_books: list[str]          # e.g. ["Exod", "Num", "Deut"]
    focus_entity: Optional[str] = None   # e.g. "Moses"
    top_k: int = 10


@router.post("/simulate")
def simulate(req: SimulateRequest):
    if not req.period_books:
        raise HTTPException(400, "'period_books' must be a non-empty list.")
    try:
        return simulator.simulate(req.period_books, req.focus_entity, req.top_k)
    except ValueError as e:
        raise HTTPException(404, str(e))
