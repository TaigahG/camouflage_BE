import trimesh
from PIL import Image

# Load your UV-fixed model
scene = trimesh.load("T34-85-UV.glb")

# Load texture image
texture = Image.open("pattern_1.jpg")

# Apply texture to ALL meshes
for name, mesh in scene.geometry.items():
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        print(f"{name}: still NO UV → skipped")
        continue

    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        image=texture
    )

print("Texture applied to all UV meshes")

# Export textured model
scene.export("textured_model.glb")