"""
Diagnostic: isolate the BRep generation stage (fit_cad_faces / fit_and_intersection)
from all upstream segmentation noise, by feeding it a clean synthetic mesh with
exact face labels. If this still produces garbage geometry, the bug is in BRep
generation itself, not in upstream over-segmentation.

Shape: a plain cylinder (2 planar caps + 1 curved side) -- simplest case that
exercises Plane + Cylinder together, which the box-only test doesn't cover.
"""
import os
import numpy as np
import trimesh

from test_real_images import fit_cad_faces, save_step_file

mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=48)
# NOTE: intentionally left low-density (no subdivide) -- this is the exact case that
# used to trigger the RANSAC min_points=30 GUARD fallback on all 3 components. Testing
# whether get_segmentation_result.py's retry-with-lower-min_points fix recovers it.
face_normals = mesh.face_normals
face_centers = mesh.vertices[mesh.faces].mean(axis=1)

labels = np.zeros(len(mesh.faces), dtype=np.int64)
z = face_normals[:, 2]
labels[z > 0.9] = 1   # top cap -> Plane
labels[z < -0.9] = 2  # bottom cap -> Plane
# everything else (z ~ 0) stays label 0 -> Cylinder side

components_labels = [1, 0, 0]  # label -> primitive type code (0=Plane,1=Cylinder,2=Cone,3=Sphere,4=Torus)
choose_count = 3
new_trimm_face_labels = labels

print("label counts:", {int(l): int((labels == l).sum()) for l in np.unique(labels)})


class Cfg:
    pass


cfg = Cfg()
cfg.output_path = "/mnt/sdb/TMEMJ/CADDreamer-jjy/neus/temp_mid_outputs/"
cfg.config_dir = os.path.join(cfg.output_path, "diag_cylinder")
cfg.review = True
os.makedirs(cfg.config_dir, exist_ok=True)

faces = fit_cad_faces(choose_count, face_centers, labels, mesh, components_labels, new_trimm_face_labels, None, cfg)
out_path = os.path.join(cfg.output_path, "diag_cylinder_test.step")
save_step_file(faces, out_path)
print("DONE ->", out_path)
