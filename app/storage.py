from supabase import create_client, Client
from dotenv import load_dotenv
import os
from typing import Optional
import uuid

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Service role key bypasses RLS — required for server-side storage uploads
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# 3 Buckets
BUCKET_BASE_IMAGES = os.getenv("BUCKET_BASE_IMAGES", "camouflage-images")
BUCKET_PATTERNS = os.getenv("BUCKET_PATTERNS", "camouflage-patterns")
BUCKET_APPLIED_MODELS = os.getenv("BUCKET_APPLIED_MODELS", "camouflage-applied-models")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

# Use service role key for storage if available, otherwise fall back to anon key
_storage_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, _storage_key)


def upload_base_image(file_content: bytes, filename: str, user_id: uuid.UUID, collection_id: int) -> str:
    """
    Upload a base/environment image to Supabase Storage
    
    Args:
        file_content: Image file bytes
        filename: Original filename (e.g., "photo.jpg")
        user_id: User ID for folder structure
        collection_id: Collection ID for folder structure
    
    Returns:
        Public URL of uploaded image
    """
    file_extension = filename.split(".")[-1] if "." in filename else "jpg"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    
    storage_path = f"user_{user_id}/collection_{collection_id}/{unique_filename}"
    
    try:
        supabase.storage.from_(BUCKET_BASE_IMAGES).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": f"image/{file_extension}"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(BUCKET_BASE_IMAGES).get_public_url(storage_path)
        
        print(f"Base image uploaded: {storage_path}")
        return public_url
        
    except Exception as e:
        print(f"Error uploading base image: {e}")
        raise


def upload_pattern(file_content: bytes, collection_id: int, user_id: uuid.UUID) -> str:
    """
    Upload AI-generated pattern to Supabase Storage
    
    Args:
        file_content: Pattern image bytes
        collection_id: Collection ID this pattern belongs to
        user_id: User ID for folder structure
    
    Returns:
        Public URL of uploaded pattern
    """
    filename = f"pattern_collection_{collection_id}.jpg"
    
    storage_path = f"user_{user_id}/{filename}"
    
    try:
        supabase.storage.from_(BUCKET_PATTERNS).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": "image/jpeg", "upsert": "true"}  # Overwrite if exists
        )
        
        public_url = supabase.storage.from_(BUCKET_PATTERNS).get_public_url(storage_path)
        
        print(f"Pattern uploaded: {storage_path}")
        return public_url
        
    except Exception as e:
        print(f" Error uploading pattern: {e}")
        raise


def upload_applied_model(
    file_content: bytes, 
    filename: str, 
    user_id: uuid.UUID,
    applied_id: int,
    file_type: str = "model"  
) -> str:
    """
    Upload 3D model with applied pattern or its thumbnail
    
    Args:
        file_content: File bytes (GLB model or thumbnail image)
        filename: Original filename
        user_id: User ID for folder structure
        applied_id: Applied pattern ID
        file_type: "model" or "thumbnail"
    
    Returns:
        Public URL of uploaded file
    """
    # Determine file extension and content type
    file_extension = filename.split(".")[-1] if "." in filename else "glb"
    
    if file_type == "thumbnail":
        content_type = "image/jpeg"
        unique_filename = f"applied_{applied_id}_thumb.jpg"
    else:
        content_type = "model/gltf-binary" if file_extension == "glb" else "model/gltf+json"
        unique_filename = f"applied_{applied_id}.{file_extension}"
    
    storage_path = f"user_{user_id}/{unique_filename}"
    
    try:
        supabase.storage.from_(BUCKET_APPLIED_MODELS).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": content_type}
        )
        
        public_url = supabase.storage.from_(BUCKET_APPLIED_MODELS).get_public_url(storage_path)
        
        print(f"Applied {file_type} uploaded: {storage_path}")
        return public_url
        
    except Exception as e:
        print(f"Error uploading applied {file_type}: {e}")
        raise


def delete_base_images(user_id: uuid.UUID, collection_id: int) -> bool:
    """
    Delete all base images for a collection
    
    Args:
        user_id: User ID
        collection_id: Collection ID
    
    Returns:
        True if successful
    """
    folder_path = f"user_{user_id}/collection_{collection_id}"
    
    try:
        files = supabase.storage.from_(BUCKET_BASE_IMAGES).list(folder_path)
        
        if files:
            file_paths = [f"{folder_path}/{file['name']}" for file in files]
            supabase.storage.from_(BUCKET_BASE_IMAGES).remove(file_paths)
            print(f"Deleted {len(file_paths)} base images from {folder_path}")
        
        return True
        
    except Exception as e:
        print(f"Error deleting base images: {e}")
        return False


def delete_pattern(user_id: uuid.UUID, collection_id: int) -> bool:
    """
    Delete AI-generated pattern for a collection
    
    Args:
        user_id: User ID
        collection_id: Collection ID
    
    Returns:
        True if successful
    """
    storage_path = f"user_{user_id}/pattern_collection_{collection_id}.jpg"
    
    try:
        supabase.storage.from_(BUCKET_PATTERNS).remove([storage_path])
        print(f"Deleted pattern: {storage_path}")
        return True
        
    except Exception as e:
        print(f"Error deleting pattern: {e}")
        return False


def delete_applied_model(user_id: uuid.UUID, applied_id: int) -> bool:
    """
    Delete applied 3D model and its thumbnail
    
    Args:
        user_id: User ID
        applied_id: Applied pattern ID
    
    Returns:
        True if successful
    """
    folder_path = f"user_{user_id}"
    
    try:
        # List all files in user folder
        files = supabase.storage.from_(BUCKET_APPLIED_MODELS).list(folder_path)
        
        # Find files matching this applied_id
        files_to_delete = [
            f"{folder_path}/{file['name']}" 
            for file in files 
            if file['name'].startswith(f"applied_{applied_id}")
        ]
        
        if files_to_delete:
            supabase.storage.from_(BUCKET_APPLIED_MODELS).remove(files_to_delete)
            print(f"Deleted {len(files_to_delete)} files for applied_{applied_id}")
        
        return True
        
    except Exception as e:
        print(f"Error deleting applied model: {e}")
        return False


def delete_user_storage(user_id: int) -> bool:
    folder_path = f"user_{user_id}"
    success = True
    
    # Delete from base images bucket
    try:
        files = supabase.storage.from_(BUCKET_BASE_IMAGES).list(folder_path)
        if files:
            file_paths = [f"{folder_path}/{file['name']}" for file in files]
            supabase.storage.from_(BUCKET_BASE_IMAGES).remove(file_paths)
            print(f"Deleted base images for user_{user_id}")
    except Exception as e:
        print(f"Error deleting base images: {e}")
        success = False
    
    # Delete from patterns bucket
    try:
        files = supabase.storage.from_(BUCKET_PATTERNS).list(folder_path)
        if files:
            file_paths = [f"{folder_path}/{file['name']}" for file in files]
            supabase.storage.from_(BUCKET_PATTERNS).remove(file_paths)
            print(f"Deleted patterns for user_{user_id}")
    except Exception as e:
        print(f"Error deleting patterns: {e}")
        success = False
    
    # Delete from applied models bucket
    try:
        files = supabase.storage.from_(BUCKET_APPLIED_MODELS).list(folder_path)
        if files:
            file_paths = [f"{folder_path}/{file['name']}" for file in files]
            supabase.storage.from_(BUCKET_APPLIED_MODELS).remove(file_paths)
            print(f"Deleted applied models for user_{user_id}")
    except Exception as e:
        print(f"Error deleting applied models: {e}")
        success = False
    
    return success