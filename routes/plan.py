from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import planner

router = APIRouter()


class PlanRequest(BaseModel):
    source: str   # "Genesis 12:3"
    target: str   # "Galatians 3:16"


@router.post("/plan")
def plan(req: PlanRequest):
    try:
        return planner.plan(req.source, req.target)
    except ValueError as e:
        raise HTTPException(404, str(e))
