# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Start the server
uvicorn app.main:app --reload

# Install dependencies
pip install -r requirements.txt

# Test the AI pipeline in isolation (no FastAPI/Supabase required)
python test_model.py

# Interactive API docs (after server is running)
# http://localhost:8000/docs
```

`test_model.py` runs an interactive CLI that lets you test preprocessing only (fast, no GPU needed) or the full SD3 pipeline. Place test images in `test_images/` first.

## Architecture Overview

This is a backend API for an AI-powered camouflage pattern generator that applies textures to 3D models.

### Tech Stack
- **Framework**: FastAPI with SQLAlchemy ORM
- **Database**: Supabase PostgreSQL (SSL required)
- **Storage**: Supabase Storage (3 buckets)
- **Authentication**: Supabase JWT tokens via Bearer
- **3D Processing**: Trimesh + xatlas for GLB manipulation and UV mapping
- **AI**: Stable Diffusion 3 img2img via diffusers

### Three Core Workflows

**1. Pattern Generation** (`/api/collections`)
- Upload 1–9 environment images → stored in `camouflage-base-images`
- Collection created with associated base images
- AI generates camouflage pattern via SD3 img2img; falls back silently if AI unavailable
- Pattern stored in `camouflage-patterns`; URL saved to collection record
- Retry/regenerate via `POST /api/collections/{id}/generate?apply_segmentation=true|false`

**2. 3D Model Texturing** (`/api/apply-uv`, `/api/apply-pattern`)
- `POST /api/apply-uv` (`app/routers/images.py`): Upload GLB → xatlas generates UV maps for meshes missing them → returns fixed GLB
- `POST /api/apply-pattern` (`app/routers/trimesh_router.py`): Upload UV-mapped GLB + pattern image → returns textured GLB as download

Both endpoints stream the result back directly; they do not persist to storage.

**3. Authentication Flow**
- `get_current_user` in `app/auth.py` validates Supabase JWT and returns `UserInfo(id: UUID, email: str)`
- Protected endpoints use `Depends(get_current_user)`
- Items endpoints (`/api/items`) have no auth enforcement (intended as admin-only, not yet protected)

### Database Schema

SQLAlchemy models in `app/models.py` — no `User` model, user identity comes from Supabase auth:
- `Collection` (user_id UUID, title, pattern_image_url) → `BaseImage` (1-to-many, cascade delete)
- `Collection` → `AppliedPattern` (cascade delete)
- `AppliedPattern` (user_id UUID, collection_id, item_id, applied_model_url, thumbnail_url)
- `Item` (item_type, item_3d_model_url, thumbnail_url) — catalog of 3D models

### Storage Structure

Three Supabase buckets; all uploads use `SUPABASE_SERVICE_ROLE_KEY` to bypass RLS:
- **camouflage-base-images**: `user_{id}/collection_{id}/{uuid}.{ext}`
- **camouflage-patterns**: `user_{id}/pattern_collection_{id}.jpg`
- **camouflage-applied-models**: `user_{id}/applied_{id}.glb` and `applied_{id}_thumb.jpg`

Upload/delete helpers are in `app/storage.py`. Pattern upload does a remove-then-upload (not upsert) to handle the case where a pattern already exists for that collection.

### AI Pipeline (`ai_model/`)

`PatternService` singleton (`app/services/pattern_service.py`) loads at startup and is reused for every request. Pipeline stages:
1. **Preprocessing** (`preprocessing.py`): KMeans color extraction + block-shuffle composite (destroys spatial structure, preserves color/texture)
2. **Generation** (`pipeline.py`): `CamouflagePipeline` wraps SD3 img2img with optional LoRA weights
3. **Postprocessing** (`postprocessing.py`): Optional median denoising + SLIC superpixel segmentation mapped to extracted color palette via CIEDE2000 distance

The `apply_segmentation` flag (default `True` in collections router, default `False` in `PatternService.generate_pattern`) controls whether SLIC runs. Raw SD3 output (no SLIC) often produces better digital camo patterns.

### Server Lifecycle

`app/main.py` uses `lifespan` context manager. At startup: creates DB tables, then attempts to load `PatternService`. If AI load fails, the server starts anyway — all non-AI endpoints remain functional. Check `GET /` for `ai_model` status.

## Environment Configuration

Required `.env` variables (see `env.example`):
- `DATABASE_URL`: Must include `sslmode=require`
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `HF_TOKEN`: HuggingFace access token — required to download SD3 (~4–5 GB on first run); accept the SD3 license at huggingface.co first
- `BUCKET_BASE_IMAGES`, `BUCKET_PATTERNS`, `BUCKET_APPLIED_MODELS`: bucket names
- `SECRET_KEY`: Application secret key
- `LORA_WEIGHTS_PATH`: (Optional) Path to LoRA `.safetensors` for fine-tuned SD3
- `AI_DEVICE`: (Optional) Force `cuda`, `mps`, or `cpu`; auto-detects otherwise

## Gotchas

- **Test endpoint**: `POST /api/collections/test-create` in `collections.py` bypasses auth with a hardcoded UUID. Remove before any production deployment.
- **Bucket name default mismatch**: `storage.py` defaults `BUCKET_BASE_IMAGES` to `camouflage-images`, but `env.example` uses `camouflage-base-images`. Set explicitly in `.env`.
- **`app/trimesh/` directory**: Contains prototype/test scripts (`trimesh_test.py`, `trimesh_apply.py`). Not imported by the app — the production trimesh endpoint is `app/routers/trimesh_router.py`.
- **DB pool_pre_ping**: Enabled in `database.py` to reconnect stale connections that time out during long AI generation runs (~1–2 min on MPS).
- **Items CRUD is unprotected**: Create/delete item endpoints have no auth dependency — intended for admin use but not enforced in code.
