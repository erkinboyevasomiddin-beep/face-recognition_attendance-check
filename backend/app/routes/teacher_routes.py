from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from backend.app.auth import require_teacher
from backend.app.models import User

router = APIRouter(prefix="/teacher", tags=["Teacher"])


def templates(request: Request):
    return request.app.state.templates


@router.get("", response_class=HTMLResponse)
def teacher_dashboard(request: Request, current_user: User = Depends(require_teacher)):
    return templates(request).TemplateResponse(
        request=request,
        name="teacher_dashboard.html",
        context={
            "page_title": "O'qituvchi paneli",
            "current_user": current_user,
            "active_section": "teacher",
        },
    )
