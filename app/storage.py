from supabase import create_client, Client
import os
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "camouflage-images")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def upload_image(file: bytes, filename: str, folder: str = "base_images") -> Optional[str]:
    try:
        file_extension = filename.split(".")[-1]
        unique_filename = f"{folder}/{uuid.uuid4()}.{file_extension}"

        response = supabase.storage.from_(SUPABASE_BUCKET).upload(unique_filename, file, {"content-type": f"image/{file_extension}"})
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
        
        return public_url.public_url
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None
    
async def delete_image(file_path: str) -> bool:
    try:
        path = file_path.split(f"/{SUPABASE_BUCKET}/")[-1]
        response = supabase.storage.from_(SUPABASE_BUCKET).remove([path])
        return True
    except Exception as e:
        print(f"Error deleting image: {e}")
        return False