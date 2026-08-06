"""Compatibility facade for the recognition command modules.

New code should accept :class:`RecognitionSettings` explicitly where practical. The
constants below keep the original scripts readable while sourcing every configurable
value from validated environment settings.
"""

from recognition.app.settings import get_settings

_settings = get_settings()

BASE_DIR = _settings.data_dir.parent
DATA_DIR = _settings.data_dir
KNOWN_FACES_DIR = _settings.known_faces_dir
EMBEDDINGS_DIR = _settings.embeddings_dir
DATABASE_PATH = _settings.database_path
MODELS_DIR = _settings.models_dir
INSIGHTFACE_ROOT = _settings.models_dir.parent
MODEL_NAME = _settings.model_name
PREFERRED_PROVIDERS = _settings.preferred_providers
RECOGNITION_THRESHOLD = _settings.match_threshold
DETECTION_SIZE = _settings.detection_size
MIN_FACE_SIZE = _settings.min_face_size
VIDEO_SOURCE = _settings.video_source
VIDEO_SOURCE_TYPE = _settings.video_source_type
VIDEO_SOURCE_NAME = _settings.video_source_name
CAMERA_MODE = _settings.camera_mode
WEBCAM_CAMERA_INDEX = _settings.camera_id
WEBCAM_INDEX = WEBCAM_CAMERA_INDEX
WEBCAM_FRAME_WIDTH = _settings.webcam_frame_width
WEBCAM_FRAME_HEIGHT = _settings.webcam_frame_height
WEBCAM_PROCESS_EVERY_N_FRAMES = _settings.webcam_process_every_n_frames
WEBCAM_RECOGNITION_SCALE = _settings.webcam_recognition_scale
WEBCAM_DET_SIZE = _settings.webcam_detection_size
WEBCAM_OVERLAY_MODE = _settings.webcam_overlay_mode
WEBCAM_RESULT_TTL_SECONDS = _settings.webcam_result_ttl_seconds
WEBCAM_MAX_QUEUE_SIZE = _settings.webcam_max_queue_size
WEBCAM_ENABLE_THREADED_CAPTURE = _settings.webcam_enable_threaded_capture
WEBCAM_START_FULLSCREEN = _settings.webcam_start_fullscreen
MAX_ACTIVE_FACES = _settings.max_active_faces
TRACK_MATCH_DISTANCE_THRESHOLD = _settings.track_match_distance_threshold
TRACK_MAX_MISSING_FRAMES = _settings.track_max_missing_frames
TRACK_MAX_AGE_SECONDS = _settings.track_max_age_seconds
CAMERA_AUTO_RECONNECT = _settings.camera_auto_reconnect
CAMERA_RECONNECT_INTERVAL_SECONDS = _settings.camera_reconnect_interval_seconds
CAMERA_MAX_READ_FAILURES_BEFORE_RECONNECT = _settings.camera_max_read_failures_before_reconnect
CAMERA_READ_FAILURE_RETRY_DELAY_SECONDS = _settings.camera_read_failure_retry_delay_seconds
FRAME_RESIZE_SCALE = _settings.frame_resize_scale
FRAME_SKIP = _settings.frame_skip
TEMPORAL_SMOOTHING = _settings.temporal_smoothing
SMOOTHING_WINDOW = _settings.smoothing_window
RECOGNITION_DISPLAY_MIN_SIMILARITY = _settings.display_threshold
RECOGNITION_CONFIRM_MIN_SIMILARITY = _settings.confirm_threshold
RECOGNITION_REQUIRED_CONSECUTIVE_HITS = _settings.required_consecutive_hits
RECOGNITION_MAX_MISSES_BEFORE_RESET = _settings.max_misses_before_reset
RECOGNITION_STABILITY_TIME_WINDOW = _settings.stability_time_window
RECOGNITION_POSITION_TOLERANCE = _settings.recognition_position_tolerance
API_ENABLED = _settings.api_enabled
API_URL = _settings.api_url
API_TIMEOUT = _settings.api_timeout_seconds
API_SEND_ONLY_RECOGNIZED = True
API_MIN_SIMILARITY = _settings.api_min_similarity
API_COOLDOWN_SECONDS = _settings.api_cooldown_seconds
API_KEY = _settings.api_key.get_secret_value() if _settings.api_key else ""
API_MAX_RETRIES = _settings.api_max_retries
SNAPSHOT_ENABLED = _settings.snapshot_enabled
SNAPSHOT_DIR = _settings.snapshot_dir
SNAPSHOT_IMAGE_EXTENSION = ".jpg"
SNAPSHOT_JPEG_QUALITY = _settings.snapshot_jpeg_quality
SNAPSHOT_FACE_PADDING = _settings.snapshot_face_padding
SNAPSHOT_COOLDOWN_SECONDS = _settings.snapshot_cooldown_seconds
LOG_LEVEL = _settings.log_level
QUIET_MODE = _settings.quiet_mode

# Presentation-only constants remain code defaults.
UNKNOWN_LABEL = "Unknown"
WINDOW_NAME = "Local Face Recognition"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
RECOGNITION_SITE_SEND_ONLY_CONFIRMED = True
ASYNC_API_SEND = True
ASYNC_SNAPSHOT_SAVE = True
WEBCAM_SHOW_FPS = True
WEBCAM_SHOW_DISPLAY_FPS = True
WEBCAM_SHOW_RECOGNITION_FPS = False
WEBCAM_SHOW_GLOBAL_PANEL = False
WEBCAM_SHOW_FACE_LABELS = True
WEBCAM_SHOW_PER_FACE_STATE = False
WEBCAM_SHOW_FACE_COUNT = False
WEBCAM_SHOW_HELP_OVERLAY = False
WEBCAM_SHOW_SOURCE = False
WEBCAM_SHOW_STATUS = True
WEBCAM_ENABLE_RESET_HOTKEY = True
WEBCAM_ENABLE_TRACKING = True


def ensure_project_directories() -> None:
    for path in (DATA_DIR, KNOWN_FACES_DIR, EMBEDDINGS_DIR, MODELS_DIR, SNAPSHOT_DIR):
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
