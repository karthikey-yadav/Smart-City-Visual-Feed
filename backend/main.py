"""
main.py - FastAPI backend for the Smart City Visual Feed Quality Gatekeeper.

Endpoints:
    POST /analyze         Upload an image, run quality analysis, persist + return result
    GET  /results/{id}    Retrieve a single past analysis by id
    GET  /results         List past analyses (history)
    GET  /health          Health/status check

Run locally with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import json
import time
import sqlite3
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from feature_extractor import extract_features

# ---------------------------------------------------------------------------
# Configuration (env vars with sensible local defaults)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "quality_model_best.pt")
DB_PATH = os.environ.get("DB_PATH", "quality_results.db")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
HOST_DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP_INV = {0: "ACCEPTABLE", 1: "DEGRADED", 2: "DEFECTIVE"}

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
device = torch.device(HOST_DEVICE)

_model = models.mobilenet_v2(weights=None)
_model.classifier[1] = nn.Linear(_model.last_channel, 3)

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'. Set MODEL_PATH env var "
        f"or place quality_model_best.pt next to main.py."
    )

_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
_model = _model.to(device)
_model.eval()

_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"[startup] Model loaded from '{MODEL_PATH}' on device '{device}'.")


# ---------------------------------------------------------------------------
# Fusion logic (classical CV + CNN) - carried over from evaluation notebook
# ---------------------------------------------------------------------------
def fuse_prediction(cnn_label: str, cv_features: dict) -> str:
    """
    Combines the CNN's prediction with classical CV flags to correct
    for domain-shift overconfidence (a CNN trained on curated reference
    images tends to over-predict severity on real-world camera photos).

    0 CV flags triggered  -> ACCEPTABLE
    1 CV flag triggered   -> cap at DEGRADED (never let CNN push to DEFECTIVE alone)
    2+ CV flags triggered -> trust the CNN's label
    """
    flags = [
        cv_features["is_blurry"],
        cv_features["is_underexposed"],
        cv_features["is_overexposed"],
        cv_features["is_noisy"],
        cv_features["is_likely_corrupt"],
    ]
    n_flags = sum(flags)

    if n_flags == 0:
        return "ACCEPTABLE"
    elif n_flags == 1:
        return "DEGRADED" if cnn_label == "DEFECTIVE" else cnn_label
    else:
        return cnn_label


def label_to_score(label: str, cv_features: dict) -> int:
    """
    Maps the final quality_label to a 0-100 quality_score, nudged by
    how many CV issues were found (more issues -> lower score within
    the label's band) so the score isn't just a 3-value step function.
    """
    n_flags = sum([
        cv_features["is_blurry"], cv_features["is_underexposed"],
        cv_features["is_overexposed"], cv_features["is_noisy"],
        cv_features["is_likely_corrupt"],
    ])
    bands = {"ACCEPTABLE": (85, 100), "DEGRADED": (50, 84), "DEFECTIVE": (0, 49)}
    lo, hi = bands[label]
    # more flags within a band -> push toward the low end of that band
    frac = min(n_flags / 3, 1.0)
    return int(round(hi - frac * (hi - lo)))


def build_issues_list(cv_features: dict) -> list:
    issues = []
    if cv_features["is_blurry"]:
        issues.append({"type": "blur", "severity": _severity_from(cv_features["blur_score"], invert=True, lo=30, hi=100),
                        "confidence": round(min(1.0, max(0.5, 1 - cv_features["blur_score"] / 100)), 2)})
    if cv_features["is_underexposed"]:
        issues.append({"type": "underexposure", "severity": _severity_from(cv_features["brightness_mean"], invert=True, lo=20, hi=60),
                        "confidence": round(min(1.0, max(0.5, 1 - cv_features["brightness_mean"] / 60)), 2)})
    if cv_features["is_overexposed"]:
        issues.append({"type": "overexposure", "severity": _severity_from(cv_features["brightness_mean"], lo=200, hi=255),
                        "confidence": round(min(1.0, max(0.5, cv_features["bright_clip_fraction"] * 2)), 2)})
    if cv_features["is_noisy"]:
        issues.append({"type": "noise", "severity": _severity_from(cv_features["noise_sigma"], lo=11, hi=30),
                        "confidence": round(min(1.0, max(0.5, cv_features["noise_sigma"] / 30)), 2)})
    if cv_features["is_likely_corrupt"]:
        issues.append({"type": "corruption", "severity": _severity_from(cv_features["blockiness_score"], lo=1.2, hi=3.0),
                        "confidence": round(min(1.0, max(0.5, cv_features["blockiness_score"] / 3)), 2)})
    return issues


def _severity_from(value, lo, hi, invert=False):
    frac = (value - lo) / (hi - lo) if hi != lo else 0
    frac = max(0.0, min(1.0, frac))
    if invert:
        frac = 1 - frac
    if frac < 0.34:
        return "low"
    elif frac < 0.67:
        return "medium"
    return "high"


def run_inference(image_bytes: bytes) -> dict:
    # Decode for CV features (OpenCV wants BGR ndarray)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr_img is None:
        raise ValueError("Could not decode image data")

    cv_feats = extract_features(bgr_img)

    # CNN inference (PIL wants RGB)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = _eval_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = _model(img_t)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        cnn_label = LABEL_MAP_INV[int(np.argmax(probs))]
        cnn_confidence = float(np.max(probs))

    final_label = fuse_prediction(cnn_label, cv_feats)
    score = label_to_score(final_label, cv_feats)
    issues = build_issues_list(cv_feats)

    return {
        "quality_score": score,
        "quality_label": final_label,
        "issues": issues,
        "cnn_raw_label": cnn_label,
        "cnn_confidence": round(cnn_confidence, 3),
        "features": cv_feats,
    }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT,
            quality_score INTEGER,
            quality_label TEXT,
            result_json TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Smart City Visual Feed Quality Gatekeeper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(obj):
    """Recursively convert numpy scalar types to native Python types
    so json.dumps never chokes on numpy.bool_/float32/int64 etc."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class AnalysisRecord(BaseModel):
    id: int
    filename: str
    created_at: str
    quality_score: int
    quality_label: str
    issues: list


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp", "image/webp"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f}MB > {MAX_UPLOAD_MB}MB limit)")

    try:
        result = run_inference(contents)
    except (ValueError, UnidentifiedImageError):
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    created_at = datetime.now(timezone.utc).isoformat()
    safe_result = _json_safe(result)
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO analyses (filename, created_at, quality_score, quality_label, result_json) VALUES (?, ?, ?, ?, ?)",
        (file.filename, created_at, safe_result["quality_score"], safe_result["quality_label"], json.dumps(safe_result)),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()

    return {
        "id": record_id,
        "filename": file.filename,
        "created_at": created_at,
        "quality_score": safe_result["quality_score"],
        "quality_label": safe_result["quality_label"],
        "issues": safe_result["issues"],
        "details": {
            "cnn_raw_label": safe_result["cnn_raw_label"],
            "cnn_confidence": safe_result["cnn_confidence"],
            "image_features": safe_result["features"],
        },
    }


@app.get("/results/{result_id}")
def get_result(result_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (result_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No analysis found with id {result_id}")
    return json.loads(row["result_json"]) | {
        "id": row["id"], "filename": row["filename"], "created_at": row["created_at"]
    }


@app.get("/results")
def list_results(limit: int = 50, offset: int = 0):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, created_at, quality_score, quality_label FROM analyses "
        "ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
