"""
app/main.py

Loads the AI model at startup via a lifespan context manager and cleans
up at shutdown.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from . import models
from .routers import users, collections, images, trimesh_router, applied_patterns


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    try:
        from .services.pattern_service import pattern_service
        pattern_service.startup()
    except Exception as e:
        print(f"[WARNING] AI model failed to load: {e}")
        print("[WARNING] Pattern generation will be unavailable.")
        print("[WARNING] All other API endpoints will work normally.")

    yield

    print("[Server] Shutting down...")

app = FastAPI(
    title="Camouflage Pattern Generator API",
    description="API for generating and managing AI-powered camouflage patterns with Supabase",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(users.router, prefix="/api")
app.include_router(collections.router, prefix="/api")

app.include_router(images.router, prefix="/api")
app.include_router(trimesh_router.router, prefix="/api")
app.include_router(applied_patterns.router, prefix="/api")

@app.get("/")
def root():
    """Root endpoint - API information"""
    try:
        from .services.pattern_service import pattern_service
        ai_status = "ready" if pattern_service.is_ready else "unavailable"
    except Exception:
        ai_status = "not loaded"

    return {
        "message": "Camouflage Pattern Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "database": "Supabase PostgreSQL",
        "storage": "Supabase Storage",
        "ai_model": ai_status,
        "status": "operational"
    }
