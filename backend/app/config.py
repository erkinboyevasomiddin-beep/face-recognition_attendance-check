"""Backward-compatible accessors backed by validated settings."""

from backend.app.settings import get_settings

_settings = get_settings()
ATTENDANCE_ON_TIME_HOUR = _settings.attendance_on_time_hour
ATTENDANCE_ON_TIME_MINUTE = _settings.attendance_on_time_minute
