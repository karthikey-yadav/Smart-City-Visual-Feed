# Smart City Visual Feed Quality Gatekeeper

An image quality and defect detection system designed to catch blurry, noisy, or corrupted camera footage before it reaches downstream analysis pipelines.

## The idea

Smart cities run on cameras — traffic junctions, CCTV, citizen-reported pothole/garbage photos, drone infrastructure inspections. All of that footage feeds into downstream AI systems (vehicle counting, waste detection, defect detection). Those systems fail silently when the *input* is bad: fog, a dirty lens, a badly-exposed night shot, sensor noise, or a corrupted upload from a citizen's phone. Garbage in, garbage out — the AI making the real decision never knows it was fed junk.

This application sits in front of those pipelines as a quality gate. A camera or citizen uploads an image; the system scores it (blur, exposure, noise, corruption, overall defect likelihood); if it's acceptable it can be forwarded to the real downstream pipeline, and if it's degraded or defective it's flagged for recapture or maintenance — logged for a dashboard instead of silently corrupting whatever analysis runs next.

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────┐
│  Frontend   │ ───────────────▶│   Backend    │
│ (HTML/JS,   │◀─────────────── │  (FastAPI)   │
│  nginx)     │      JSON       └──────┬───────┘
└─────────────┘                        │
                          ┌─────────────┴─────────────┐
                          │                            │
                 ┌────────▼────────┐         ┌─────────▼────────┐
                 │ Classical CV     │         │  CNN (MobileNetV2 │
                 │ feature          │         │  transfer         │
                 │ extractor        │         │  learning)        │
                 │ (OpenCV)         │         │  (PyTorch)        │
                 └────────┬────────┘         └─────────┬────────┘
                          │                             │
                          └───────────┬─────────────────┘
                                      ▼
                            Fusion decision logic
                                      ▼
                              SQLite persistence
```

## Quick start (Docker — recommended)

Requires Docker Desktop.

```bash
git clone <this-repo>
cd SCRC
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Analysis results persist in a Docker volume (`backend_data`), so they survive container restarts. To reset, run `docker compose down -v`.

## Quick start (local, without Docker)

```bash
cd backend
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` directly in a browser (it calls `http://localhost:8000` by default).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `quality_model_best.pt` | Path to the trained CNN weights |
| `DB_PATH` | `quality_results.db` | SQLite database file location |
| `DEVICE` | `cpu` (auto-detects `cuda` if available) | Inference device |
| `MAX_UPLOAD_MB` | `10` | Max accepted upload size |

## Dataset and training

**Training/validation data:** [KADID-10k](http://database.mmsp-kn.de/kadid-10k-database.html) — 81 pristine reference images × 25 distortion types × 5 severity levels (10,125 distorted images total). KADID's 25 distortion types were mapped onto our 5 target categories:

| Our category | KADID distortion types used |
|---|---|
| blur | Gaussian blur, lens blur, motion blur |
| noise | White noise, color-component noise, impulse noise, multiplicative noise |
| underexposure | Darken |
| overexposure | Brighten |
| corruption | JPEG compression, JPEG2000 compression, pixelate, color block |

After filtering to these 5 categories: **5,265 usable images**. Quality labels (ACCEPTABLE / DEGRADED / DEFECTIVE) were derived by thresholding each image's DMOS (differential mean opinion score, human-rated quality on a 1–5 scale where higher = better quality): ≥4.0 → ACCEPTABLE, 2.5–4.0 → DEGRADED, <2.5 → DEFECTIVE.

Split **by reference image** (not by individual distorted image) into train/val/test so the same underlying scene never appears in more than one split:
- Train: 3,640 images
- Validation: 780 images
- Test: 845 images

**Demo/generalization set:** 40 real traffic-camera images (Kaggle "Traffic Detection Project" dataset), synthetically degraded with our own script (`degrade_images.py`) covering blur, under/over-exposure, noise, and corruption at 3 severity levels, plus untouched clean images — used specifically to test generalization to a different visual domain than the training set, and to produce the sample images required for submission (see `data/demo_degraded/`).

## Model

**Architecture:** MobileNetV2 (ImageNet-pretrained), classification head replaced with a 3-class linear layer (ACCEPTABLE / DEGRADED / DEFECTIVE). Chosen as a lightweight transfer-learning approach appropriate for a 48-hour timeline and CPU-friendly inference in production.

**Training:** 8 epochs, Adam optimizer (lr=1e-4), cross-entropy loss, best checkpoint selected by validation accuracy (not final epoch — see Evaluation below for why).

**Hybrid design:** the CNN alone is not the final decision-maker. Its output is combined with 5 classical, physically-grounded CV signals (blur via Laplacian variance, exposure via brightness/clipping, noise via high-frequency residual MAD, corruption via JPEG-blockiness detection) through a fusion rule (see `main.py::fuse_prediction`). This is what the assessment brief calls a hybrid approach — image-quality features combined with a learned model — and it exists for a concrete, tested reason (see Evaluation).

## API

### `POST /analyze`
Upload an image for analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@sample.jpg"
```

Response:
```json
{
  "id": 1,
  "filename": "sample.jpg",
  "created_at": "2026-08-29T09:15:28.393735+00:00",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {"type": "noise", "severity": "low", "confidence": 0.71}
  ],
  "details": {
    "cnn_raw_label": "ACCEPTABLE",
    "cnn_confidence": 0.87,
    "image_features": { "blur_score": 210.4, "brightness_mean": 128.3, "...": "..." }
  }
}
```

Returns `400` for unreadable/invalid images, `415` for unsupported content types, `413` for oversized files.

### `GET /results/{id}`
Retrieve one past analysis by ID. Returns `404` if not found.

### `GET /results?limit=50&offset=0`
List past analyses, most recent first.

### `GET /health`
Service status check — returns model-loaded state and device.

## Database

SQLite, auto-created on first run at `DB_PATH`. Single table:

```sql
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    created_at TEXT,
    quality_score INTEGER,
    quality_label TEXT,
    result_json TEXT   -- full analysis result, including CV features
);
```

No migration step needed — the table is created automatically at startup if it doesn't exist.

## Evaluation

### Test set (KADID-10k, 845 unseen images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| ACCEPTABLE | 0.742 | 0.401 | 0.520 |
| DEGRADED | 0.534 | 0.597 | 0.563 |
| DEFECTIVE | 0.719 | 0.868 | 0.786 |
| **Overall accuracy** | | | **0.647** |

Confusion matrix (rows = true, columns = predicted; order ACCEPTABLE/DEGRADED/DEFECTIVE):
```
[[ 89 117  16]
 [ 31 182  92]
 [  0  42 276]]
```

**Key finding:** the model almost never confuses the two extremes — only 16 of 845 truly-ACCEPTABLE-or-DEFECTIVE images were predicted as the opposite extreme. Nearly all errors are between *adjacent* classes, meaning the model learned the ordinal structure of quality even though it was trained as plain 3-way classification. DEFECTIVE recall (0.868) is the strongest — arguably the most important capability for a gatekeeper system, since catching the worst images matters more than perfectly ranking borderline ones.

**Training/overfitting note:** training accuracy reached 0.85+ by epoch 8, while validation accuracy plateaued around 0.65–0.70 from epoch 3 onward and validation loss began increasing — a standard overfitting signature. The best checkpoint (by validation accuracy, epoch 7) was used rather than the final epoch's weights.

### Generalization test (40 real traffic-camera images, different domain than training)

| | Accuracy |
|---|---|
| CNN alone | 35.0% (14/40) |
| **CNN + classical CV fusion** | **80.0% (32/40)** |

**Finding:** the CNN, trained entirely on KADID-10k's curated reference photography, showed a clear domain-shift problem when applied to real traffic-camera imagery — it flagged *every single clean traffic image* as DEGRADED or DEFECTIVE, apparently because the visual style (compression, resolution, lighting) of camera footage differs from curated reference photos even when technically undegraded.

Fusing the CNN's prediction with 5 classical CV flags (blur/exposure/noise/corruption) corrected this: when zero CV issues are physically detected, the image is called ACCEPTABLE regardless of the CNN's raw label; when exactly one issue is detected, DEFECTIVE calls are capped down to DEGRADED unless a second corroborating signal is present. This more than doubled cross-domain accuracy (35% → 80%) and is the practical justification for the hybrid architecture, not just a checkbox requirement.

### Known limitations / failure cases

- **Mild-severity exposure misses:** underexposure/overexposure at the mildest synthetic severity level sometimes falls below the classical CV thresholds and isn't flagged — expected, since "mild" degradation is inherently close to the acceptable boundary.
- **Corruption/overexposure overlap:** the JPEG-blockiness corruption detector occasionally fires on severely overexposed (blown-out) images, since large flat clipped regions share some structural similarity with block-corrupted regions. Documented, not corrected, given time constraints — a secondary "is this region actually saturated vs. blocky" check would resolve it.
- **ACCEPTABLE recall (0.401) is the model's weakest spot** on the KADID test set — images near the DMOS threshold boundary are inherently ambiguous since the ground-truth label itself comes from thresholding a continuous human quality score.

## Explainability

Every analysis response includes:
- **Per-issue confidence scores** (0–1) and severity levels (low/medium/high), derived directly from how far each interpretable CV statistic (blur score, brightness, noise sigma, blockiness) sits from its calibrated threshold — not a black-box score.
- **Raw CNN confidence and label** alongside the final fused decision, so a reviewer can see when/why fusion overrode the CNN.
- **Full interpretable feature dump** (`image_features` in the API response) — blur score, brightness mean, dark/bright clip fractions, noise sigma, contrast, saturation, blockiness score — giving a complete, human-readable basis for every decision rather than an opaque number.

## Project structure

```
SCRC/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # FastAPI app, fusion logic, DB
│   ├── feature_extractor.py     # classical CV feature module
│   └── quality_model_best.pt    # trained CNN weights
├── frontend/
│   ├── Dockerfile
│   └── index.html               # single-file UI
├── data/
│   └── demo_degraded/           # sample images + ground-truth labels.json
├── degrade_images.py            # synthetic degradation generator
├── organize_kadid.py            # KADID-10k dataset preparation
└── README.md
```

## Sample images

`data/demo_degraded/` contains 40 smart-city (traffic camera) images with controlled synthetic degradations across all 5 categories plus untouched clean examples, along with `labels.json` giving the ground-truth degradation type, severity, and quality label for each — used for the generalization evaluation above and provided here as the required sample image set.
