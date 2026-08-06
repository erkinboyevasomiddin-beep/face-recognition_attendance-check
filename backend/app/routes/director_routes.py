from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from backend.app.auth import require_director
from backend.app.database import get_db
from backend.app.models import SchoolClass, User, WorkerCategory
from backend.app.schemas import AttendanceStatus, ManualCorrectionCreate, PersonType
from backend.app.security import validate_csrf_token
from backend.app.services import (
    apply_manual_correction,
    current_local_date,
    get_class_detail,
    get_director_overview,
    get_recent_recognition_events,
    get_students_overview,
    get_worker_categories_overview,
    get_worker_category_detail,
)

router = APIRouter(prefix="/director", tags=["Director"])


def templates(request: Request):
    return request.app.state.templates


@router.get("", response_class=HTMLResponse)
def director_dashboard(
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    overview = get_director_overview(db, current_local_date())
    return templates(request).TemplateResponse(
        request=request,
        name="director_dashboard.html",
        context={
            "page_title": "Direktor paneli",
            "current_user": current_user,
            "active_section": "overview",
            "overview": overview,
        },
    )


@router.get("/students", response_class=HTMLResponse)
def students_page(
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    today = current_local_date()
    return templates(request).TemplateResponse(
        request=request,
        name="students.html",
        context={
            "page_title": "O'quvchilar",
            "current_user": current_user,
            "active_section": "students",
            "today": today,
            "grade_groups": get_students_overview(db, today),
            "overview": get_director_overview(db, today),
        },
    )


@router.get("/students/class/{class_id}", response_class=HTMLResponse)
def class_detail_page(
    class_id: int,
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    detail = get_class_detail(db, class_id, current_local_date())
    if not detail:
        raise HTTPException(status_code=404, detail="Sinf topilmadi.")
    school_class = detail["school_class"]
    assert isinstance(school_class, SchoolClass)

    return templates(request).TemplateResponse(
        request=request,
        name="class_detail.html",
        context={
            "page_title": school_class.name,
            "current_user": current_user,
            "active_section": "students",
            "detail": detail,
        },
    )


@router.get("/workers", response_class=HTMLResponse)
def workers_page(
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    today = current_local_date()
    return templates(request).TemplateResponse(
        request=request,
        name="workers.html",
        context={
            "page_title": "Xodimlar",
            "current_user": current_user,
            "active_section": "workers",
            "today": today,
            "categories": get_worker_categories_overview(db, today),
            "overview": get_director_overview(db, today),
        },
    )


@router.get("/events", response_class=HTMLResponse)
def events_page(
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    events = get_recent_recognition_events(db, limit=100)
    return templates(request).TemplateResponse(
        request=request,
        name="events.html",
        context={
            "page_title": "Tanish hodisalari",
            "current_user": current_user,
            "active_section": "events",
            "events": events,
            "summary": {
                "total": len(events),
                "matched": sum(1 for event in events if event["matched"]),
                "unmatched": sum(1 for event in events if not event["matched"]),
            },
        },
    )


@router.get("/workers/category/{category_id}", response_class=HTMLResponse)
def worker_category_page(
    category_id: int,
    request: Request,
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    detail = get_worker_category_detail(db, category_id, current_local_date())
    if not detail:
        raise HTTPException(status_code=404, detail="Xodim toifasi topilmadi.")
    category = detail["category"]
    assert isinstance(category, WorkerCategory)

    return templates(request).TemplateResponse(
        request=request,
        name="worker_category_detail.html",
        context={
            "page_title": category.name,
            "current_user": current_user,
            "active_section": "workers",
            "detail": detail,
        },
    )


@router.post("/attendance/correction")
def correct_attendance(
    request: Request,
    person_type: PersonType = Form(...),
    person_id: str = Form(..., min_length=2, max_length=64),
    attendance_date: date = Form(...),
    attendance_status: AttendanceStatus = Form(...),
    reason: str = Form(..., min_length=3, max_length=500),
    csrf_token: str = Form(...),
    current_user: User = Depends(require_director),
    db: Session = Depends(get_db),
):
    validate_csrf_token(request, csrf_token)
    try:
        apply_manual_correction(
            db,
            ManualCorrectionCreate(
                person_type=person_type,
                person_id=person_id,
                attendance_date=attendance_date,
                status=attendance_status,
                reason=reason,
            ),
            current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}
