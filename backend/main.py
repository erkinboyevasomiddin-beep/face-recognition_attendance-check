from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.app.auth import RedirectException
from backend.app.database import SessionLocal, configure_database, init_db
from backend.app.routes.api_routes import router as api_router
from backend.app.routes.auth_routes import router as auth_router
from backend.app.routes.director_routes import router as director_router
from backend.app.routes.teacher_routes import router as teacher_router
from backend.app.schemas import AttendanceStatus, UserRole
from backend.app.security import get_csrf_token
from backend.app.settings import AttendanceSettings, get_settings
from backend.seed_data import seed_synthetic_data_if_requested

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATUS_LABELS = {
    AttendanceStatus.PRESENT.value: "Keldi",
    AttendanceStatus.LATE.value: "Kech qoldi",
    AttendanceStatus.ABSENT.value: "Kelmadi",
}
ROLE_LABELS = {UserRole.DIRECTOR.value: "Direktor", UserRole.TEACHER.value: "O'qituvchi"}


def format_ui_date(value: date | None) -> str:
    return value.strftime("%d.%m.%Y") if value else ""


def format_ui_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is not None:
        value = value.astimezone(get_settings().local_zone)
    return value.strftime("%d.%m.%Y %H:%M:%S")


def create_app(settings: AttendanceSettings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_database(resolved.database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db()
        if resolved.auto_seed_demo:
            with SessionLocal() as session:
                seed_synthetic_data_if_requested(session)
        yield

    application = FastAPI(
        title="School Attendance Prototype",
        debug=resolved.debug,
        lifespan=lifespan,
    )
    session_secret = (
        resolved.session_secret.get_secret_value()
        if resolved.session_secret
        else secrets.token_urlsafe(48)
    )
    if resolved.session_secret is None:
        LOGGER.warning(
            "No ATTENDANCE_SESSION_SECRET configured; using an ephemeral development secret. "
            "Sessions will reset on restart."
        )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=resolved.allowed_hosts)
    application.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        session_cookie=resolved.session_cookie_name,
        max_age=resolved.session_max_age_seconds,
        same_site=resolved.session_cookie_same_site,
        https_only=resolved.session_cookie_secure,
    )

    static_dir = BASE_DIR / "app" / "static"
    template_dir = BASE_DIR / "app" / "templates"
    application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(template_dir))
    templates.env.globals.update(
        status_label=lambda value: STATUS_LABELS.get(value, value),
        role_label=lambda value: ROLE_LABELS.get(value, value),
        format_ui_date=format_ui_date,
        format_ui_datetime=format_ui_datetime,
        csrf_token=get_csrf_token,
        recognition_status_label=lambda value: {
            "ENTRY": "Kirish",
            "EXIT": "Chiqish",
            "MANUAL_CORRECTION": "Qo'lda tuzatish",
        }.get(value, value),
    )
    application.state.templates = templates

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        if resolved.app_env == "production":
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        if request.url.path in {"/login", "/logout"}:
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.exception_handler(RedirectException)
    async def handle_redirect_exception(_: Request, exc: RedirectException):
        return RedirectResponse(url=exc.destination, status_code=303)

    @application.get("/healthz", include_in_schema=False)
    def health_check() -> dict[str, str]:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}

    application.include_router(auth_router)
    application.include_router(api_router)
    application.include_router(director_router)
    application.include_router(teacher_router)
    return application


app = create_app()
