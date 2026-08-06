from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from backend.app.auth import (
    authenticate_user,
    get_dashboard_path,
    login_user,
    logout_user,
    optional_user,
)
from backend.app.database import get_db
from backend.app.models import User
from backend.app.security import (
    get_csrf_token,
    login_rate_key,
    login_rate_limiter,
    validate_csrf_token,
)

router = APIRouter()


def templates(request: Request):
    return request.app.state.templates


@router.get("/", include_in_schema=False)
def root(current_user: User | None = Depends(optional_user)):
    destination = "/login" if not current_user else get_dashboard_path(current_user.role)
    return RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, current_user: User | None = Depends(optional_user)):
    if current_user:
        return RedirectResponse(get_dashboard_path(current_user.role), status_code=303)
    get_csrf_token(request)
    return templates(request).TemplateResponse(
        request=request,
        name="login.html",
        context={"page_title": "Kirish", "error": None},
    )


@router.post("/login", response_class=HTMLResponse)
def login_action(
    request: Request,
    username: str = Form(..., min_length=1, max_length=50),
    password: str = Form(..., min_length=1, max_length=256),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    validate_csrf_token(request, csrf_token)
    key = login_rate_key(request, username)
    login_rate_limiter.check(key)
    user = authenticate_user(db, username=username, password=password)
    if not user:
        login_rate_limiter.record_failure(key)
        return templates(request).TemplateResponse(
            request=request,
            name="login.html",
            context={"page_title": "Kirish", "error": "Login ma'lumotlari noto'g'ri."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    login_rate_limiter.clear(key)
    login_user(request, user)
    return RedirectResponse(get_dashboard_path(user.role), status_code=303)


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form(...)):
    validate_csrf_token(request, csrf_token)
    logout_user(request)
    return RedirectResponse("/login", status_code=303)
