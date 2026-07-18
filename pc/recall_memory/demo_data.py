"""Sample cross-source data for the demo: one meeting (dual-mic), a GitHub
file, a PDF section, and a whiteboard OCR block — all touching JWT auth so
cross-source retrieval has something to bridge."""
from __future__ import annotations

import time

_NOW = None


def now() -> float:
    global _NOW
    if _NOW is None:
        _NOW = time.time()
    return _NOW


def sample_meeting(device: str = "arduino", confidence: float = 0.92) -> dict:
    start = now() - 3600
    return {
        "meeting_id": "mtg-auth-sync",
        "title": "Auth design sync",
        "capture_device": device,
        "device_id": f"{device}-01",
        "capture_confidence": confidence,
        "start_time": start,
        "asr_provider": "whisper_local",
        "utterances": [
            {"speaker": "Priya", "t_start": 0, "t_end": 9, "asr_confidence": confidence,
             "text": "Okay, main topic today is authentication for the backend API."},
            {"speaker": "Rahul", "t_start": 10, "t_end": 24, "asr_confidence": confidence,
             "text": "I compared session cookies against JWT tokens this week. "
                     "Sessions need sticky state on the hub."},
            {"speaker": "Priya", "t_start": 25, "t_end": 38, "asr_confidence": confidence,
             "text": "We decided to use JWT authentication with refresh tokens "
                     "for the backend API."},
            {"speaker": "Ananya", "t_start": 39, "t_end": 52, "asr_confidence": confidence,
             "text": "Action item: Rahul will implement the JWT middleware in the "
                     "backend repo by Friday."},
            {"speaker": "Rahul", "t_start": 53, "t_end": 66, "asr_confidence": confidence,
             "text": "Fine. We'll use the PyJWT library, HS256 to start, and rotate "
                     "to RS256 once the key service lands."},
            {"speaker": "Ananya", "t_start": 67, "t_end": 80, "asr_confidence": confidence,
             "text": "Also we decided to use PostgreSQL for the token store instead "
                     "of Redis, since we already run Postgres."},
        ],
    }


def sample_meeting_phone() -> dict:
    """Phone fallback capture of the same meeting: overlapping middle section at
    lower ASR confidence + one tail utterance the Arduino missed."""
    m = sample_meeting(device="mobile", confidence=0.71)
    m["capture_device"] = "mobile"
    m["device_id"] = "mobile-01"
    m["utterances"] = m["utterances"][2:4] + [
        {"speaker": "Priya", "t_start": 81, "t_end": 95, "asr_confidence": 0.71,
         "text": "Last thing, let's revisit rate limiting next week after the "
                 "JWT middleware is merged."},
    ]
    return m


SAMPLE_GITHUB = {
    "repo": "recall-backend",
    "file_path": "src/auth/jwt_middleware.py",
    "content": '''"""JWT authentication middleware for the Recall backend API."""
import time
import jwt
from fastapi import Request, HTTPException

SECRET_KEY = "env:RECALL_JWT_SECRET"
ALGORITHM = "HS256"
ACCESS_TTL = 900          # 15 min access tokens
REFRESH_TTL = 1209600     # 14 day refresh tokens


def create_access_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + ACCESS_TTL, "typ": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + REFRESH_TTL, "typ": "refresh"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def jwt_middleware(request: Request, call_next):
    """Validate the Bearer token on every API request."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        claims = jwt.decode(auth[7:], SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "token expired")
    request.state.user_id = claims["sub"]
    return await call_next(request)
''',
}

SAMPLE_PDF = {
    "title": "Recall Architecture Doc — Security",
    "content": (
        "Section 4: Authentication and Authorization.\n\n"
        "The Recall backend API authenticates every request with JSON Web "
        "Tokens (JWT). Access tokens are short-lived (15 minutes) and signed "
        "with HS256; refresh tokens live for 14 days and are stored hashed in "
        "PostgreSQL. The hub validates tokens in a FastAPI middleware before "
        "any memory route executes.\n\n"
        "Token rotation: when the key service ships, signing moves to RS256 "
        "asymmetric keys so edge devices can verify tokens without holding "
        "the signing secret. Revocation is handled by a deny-list table in "
        "PostgreSQL checked on refresh.\n\n"
        "Threat model: tokens never leave the local network; the Arduino and "
        "phone talk to the hub over WebSocket with the same bearer scheme."),
    "source_meta": {"page": 4, "document_title": "Recall Architecture Doc",
                    "author": "Recall Team", "heading": "Security"},
}

SAMPLE_OCR = {
    "title": "Whiteboard: auth flow",
    "content": ("Auth flow sketch: client -> login -> JWT access + refresh -> "
                "API gateway validates -> backend. Refresh path hits Postgres "
                "token store. TODO rate limiting."),
    "source_meta": {"ocr_confidence": 0.88, "image_width": 1280,
                    "image_height": 960, "detected_objects": ["whiteboard"]},
}


def seed(ingestor) -> dict:
    """Ingest the full sample set; returns the memory ids."""
    ids = {
        "meeting_arduino": ingestor.ingest_meeting(sample_meeting()),
        "meeting_phone": ingestor.ingest_meeting(sample_meeting_phone()),
        "github": ingestor.ingest_github_file(**SAMPLE_GITHUB),
        "pdf": ingestor.ingest_document("pdf", SAMPLE_PDF["title"],
                                        SAMPLE_PDF["content"],
                                        SAMPLE_PDF["source_meta"]),
        "ocr": ingestor.ingest_document("image", SAMPLE_OCR["title"],
                                        SAMPLE_OCR["content"],
                                        SAMPLE_OCR["source_meta"]),
    }
    return ids
