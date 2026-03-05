from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from sqlalchemy import inspect
from . import models 
from .routers import users, collections, items, images, trimesh_router

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Camouflage Pattern Generator API",
    description="API for generating and managing AI-powered camouflage patterns with Supabase",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(items.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(trimesh_router.router, prefix="/api")
@app.get("/")
def root():
    """Root endpoint - API information"""
    return {
        "message": "Camouflage Pattern Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "database": "Supabase PostgreSQL",
        "storage": "Supabase Storage",
        "status": "operational"
    }
