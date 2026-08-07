import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import trimesh
import test_real_images as tri

MESH_PATH = "our_inputs/claude_obj/tube.obj"
OUT_DIR = "neus/temp_mid_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

mesh = trimesh.load(MESH_PATH, process=False)
print("mesh:", len(mesh.vertices), "verts,", len(mesh.faces), "faces")

# 하드엣지(이음매)마다 버텍스가 중복돼 있어서 face_adjacency가 반토막 나 있었다.
# 좌표가 같은 버텍스를 하나로 합쳐서 위상적으로 연결되게 만든다.
mesh.merge_vertices(merge_tex=False, merge_norm=True)
print("병합 후:", len(mesh.vertices), "verts,", len(mesh.faces), "faces,", len(mesh.face_adjacency), "adjacency")

# pyransac의 epsilon/bitmap_epsilon 기본값은 NEUS가 늘 만들어내는 반지름~1 스케일
# 메시를 전제로 한다. 우리 합성 튜브(반지름 15, 길이 100)는 그보다 훨씬 커서
# 그대로 넣으면 RANSAC이 후보를 전혀 못 찾는다(silent failure) — 단위 구에
# 맞춰 정규화해서 넣는다.
scale = np.abs(mesh.vertices).max()
mesh.vertices = mesh.vertices / scale
print(f"정규화: /{scale} 적용, 새 bbox 최대값 = {np.abs(mesh.vertices).max()}")

centroids = mesh.vertices[mesh.faces].mean(axis=1)
r = np.linalg.norm(centroids[:, :2], axis=1)
z = centroids[:, 2]

new_labels = np.full(len(mesh.faces), -1, dtype=np.int64)
new_labels[(r > 14.5/scale) & (np.abs(z) < 49/scale)] = 0   # outer cylinder
new_labels[(r < 13.0/scale) & (np.abs(z) < 49/scale)] = 1   # inner cylinder
new_labels[z >= 49/scale] = 2                                # top annular cap
new_labels[z <= -49/scale] = 3                               # bottom annular cap
print("label counts:", {int(l): int((new_labels == l).sum()) for l in np.unique(new_labels)})
assert (new_labels != -1).all(), "라벨 안 붙은 면이 있음"

components_labels = [1, 1, 0, 0]  # pyransac type: 0=plane,1=cylinder
choose_count = len(set(new_labels.tolist()))
trimm_face_centers = centroids
new_trimm = mesh
new_trimm_face_labels = new_labels
bind_parallel_comps = None

class Cfg:
    pass
cfg = Cfg()
cfg.output_path = os.path.abspath(OUT_DIR)
cfg.config_dir = os.path.abspath(os.path.dirname(MESH_PATH))
cfg.review = True

faces = tri.fit_cad_faces(
    choose_count, trimm_face_centers, new_labels, new_trimm,
    components_labels, new_trimm_face_labels, bind_parallel_comps, cfg
)
out_step = os.path.join(cfg.output_path, "tube_synthetic_test.step")
tri.save_step_file(faces, out_step)
print("STEP saved to:", out_step)
