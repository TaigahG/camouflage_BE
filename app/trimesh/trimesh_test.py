import trimesh
import xatlas
import numpy as np

scene = trimesh.load("assets/T34-85.glb")   # can be obj/glb

new_geometry = {}

for name, mesh in scene.geometry.items():

    has_uv = (
        hasattr(mesh.visual, "uv") and
        mesh.visual.uv is not None and
        len(mesh.visual.uv) > 0
    )

    if has_uv:
        print(f"{name}: UV EXISTS")
        new_geometry[name] = mesh
        continue

    print(f"{name}: NO UV → Generating...")

    v = mesh.vertices.astype(np.float32)
    f = mesh.faces.astype(np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(v, f)
    atlas.generate()

    vmapping, indices, uvs = atlas[0]

    new_mesh = trimesh.Trimesh(
        vertices=v[vmapping],
        faces=indices,
        process=False
    )

    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    new_geometry[name] = new_mesh


# Create new scene with ALL meshes fixed
fixed_scene = trimesh.Scene(new_geometry)

fixed_scene.export("T34-85-UV.glb")

print("DONE — All meshes now have UV")