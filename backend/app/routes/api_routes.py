from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.auth import require_api_key
from backend.app.database import get_db
from backend.app.schemas import RecognitionEventIngest, RecognitionEventIngestResponse
from backend.app.services import StaleRecognitionEventError, ingest_recognition_event

router = APIRouter(prefix="/api", tags=["API"])


@router.post("/recognition-event", response_model=RecognitionEventIngestResponse)
def recognition_event_ingest(
    payload: RecognitionEventIngest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    if idempotency_key is not None and idempotency_key != str(payload.event_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key must match payload event_id",
        )
    try:
        return ingest_recognition_event(db, payload)
    except StaleRecognitionEventError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
