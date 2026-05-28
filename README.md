# CamoCam Backend

FastAPI-based backend for the CamoCam AI-powered camouflage pattern generator. Handles pattern generation and 3D model texturing using Stable Diffusion 3 and advanced image processing.

## Features

- **AI Pattern Generation**: Generate unique camouflage patterns from environment images using Stable Diffusion 3
- **3D Texturing**: Apply patterns to GLB models with UV mapping support
- **RESTful API**: Comprehensive FastAPI with automatic interactive docs
- **Secure Authentication**: Supabase JWT token validation
- **Cloud Storage**: Supabase PostgreSQL + object storage

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI 0.104+ |
| **Server** | Uvicorn with async workers |
| **Database** | PostgreSQL (Supabase) with SQLAlchemy ORM |
| **Storage** | Supabase Storage (3 buckets) |
| **AI Model** | Stable Diffusion 3 (diffusers) |
| **Image Processing** | PIL, scikit-image, numpy |
| **3D Processing** | Trimesh, xatlas, Open3D |
| **Authentication** | Supabase JWT |

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Disk Space**: ~10 GB for models (SD3 ~5GB + others)

### External Services

- **Supabase Account**: For PostgreSQL, Storage, and Authentication
- **HuggingFace Token**: For downloading SD3 model

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/TaigahG/camocam-be.git
cd camocam-be
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example env file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# ──────────────────────────────────────────────
# Database & Cloud (Supabase)
# ──────────────────────────────────────────────
DATABASE_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres?sslmode=require
SUPABASE_URL=https://PROJECT.supabase.co
SUPABASE_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY

# ──────────────────────────────────────────────
# Storage Buckets
# ──────────────────────────────────────────────
BUCKET_BASE_IMAGES=camouflage-base-images
BUCKET_PATTERNS=camouflage-patterns
BUCKET_APPLIED_MODELS=camouflage-applied-models

# ──────────────────────────────────────────────
# AI Model Settings
# ──────────────────────────────────────────────
# Get HF token from: https://huggingface.co/settings/tokens
# Accept SD3 license: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
HF_TOKEN=hf_YOUR_TOKEN_HERE

# Device selection (auto-detects if not set)
# Options: cuda, mps, cpu
# AI_DEVICE=mps

# Optional: Path to LoRA weights for fine-tuned models
# LORA_WEIGHTS_PATH=models/sd3_lora_camo.safetensors

# ──────────────────────────────────────────────
# App Settings
# ──────────────────────────────────────────────
SECRET_KEY=your_secret_key_here
```

### 5. Run Development Server

```bash
# With auto-reload on file changes
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`

### 7. Access API Documentation

Open your browser to:
`http://localhost:8000/docs` (Swagger UI)

## Project Structure

```
app/
├── main.py                 # FastAPI app initialization & lifespan
├── auth.py                 # Supabase JWT validation
├── database.py             # SQLAlchemy setup & session
├── models.py               # Database models (Collection, BaseImage, AppliedPattern, Item)
├── crud.py                 # Database CRUD operations
├── schemas.py              # Pydantic request/response schemas
├── storage.py              # Supabase Storage upload/delete helpers
├── routers/                # API route handlers
│   ├── collections.py      # Pattern collection management
│   ├── images.py           # UV mapping for GLB models
│   ├── trimesh_router.py   # 3D model texturing
│   ├── applied_patterns.py # Applied pattern records
│   ├── items.py            # 3D model catalog
│   └── users.py            # User management
└── services/               # Business logic
    └── pattern_service.py  # AI pattern generation pipeline

ai_model/                   # AI/ML modules
├── config.py              # Model configuration & hyperparameters
├── pipeline.py            # SD3 img2img pipeline
├── preprocessing.py       # Color extraction, image compositing
└── postprocessing.py      # Denoising, SLIC segmentation

test_model.py              # Interactive testing script
check_gpu.py              # GPU availability checker
```

## API Endpoints

### Collections (Pattern Management)

```
POST   /api/collections              # Create new collection
GET    /api/collections/me           # Get user's collections
GET    /api/collections/{id}         # Get specific collection
POST   /api/collections/{id}/generate # Regenerate pattern
DELETE /api/collections/{id}         # Delete collection
```

### Pattern Application

```
POST   /api/apply-pattern            # Apply pattern to GLB (download)
POST   /api/apply-pattern-and-save   # Apply pattern & save to storage
POST   /api/apply-uv                 # Generate UV maps for GLB
```

### Items (3D Models)

```
GET    /api/items                    # List available 3D models
GET    /api/items/{id}               # Get specific item
```

### Applied Patterns

```
GET    /api/applied-patterns/        # Get user's applied patterns
GET    /api/applied-patterns/{id}    # Get specific applied pattern
```

## Database Schema

### Collections
```sql
collections (
  collection_id SERIAL PRIMARY KEY,
  user_id UUID NOT NULL,
  title VARCHAR NOT NULL,
  pattern_image_url VARCHAR,
  created_at TIMESTAMP
)
```

### BaseImages
```sql
base_images (
  image_id SERIAL PRIMARY KEY,
  collection_id INTEGER FOREIGN KEY,
  image_url VARCHAR NOT NULL,
  upload_order INTEGER,
  uploaded_at TIMESTAMP
)
```

### AppliedPatterns
```sql
applied_patterns (
  applied_id SERIAL PRIMARY KEY,
  user_id UUID NOT NULL,
  collection_id INTEGER FOREIGN KEY,
  item_id INTEGER FOREIGN KEY,
  applied_model_url VARCHAR,
  thumbnail_url VARCHAR,
  created_at TIMESTAMP
)
```

### Items
```sql
items (
  item_id SERIAL PRIMARY KEY,
  item_type VARCHAR,
  item_3d_model_url VARCHAR,
  thumbnail_url VARCHAR
)
```

## AI Pipeline

### 1. Pattern Generation (`/api/collections`)

```
1. User uploads 1-9 environment images
   ↓
2. Preprocessing: Extract dominant colors via KMeans + block-shuffle composite
   ↓
3. SD3 img2img generation: Transform composite into camo pattern
   ↓
4. Postprocessing (optional): Median denoising + SLIC superpixel segmentation
   ↓
5. Pattern saved to storage, URL stored in database
```


### 2. Model Texturing (`/api/apply-pattern`)

```
1. User uploads GLB model + pattern image
   ↓
2. Load model with Trimesh
   ↓
3. For each mesh without UV coordinates: Generate UVs with xatlas
   ↓
4. Apply pattern texture to all UV-mapped meshes
   ↓
5. Export as new GLB file
```

## Related

- [CamoCam Frontend](../CamoCam) - Flutter app
- [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
