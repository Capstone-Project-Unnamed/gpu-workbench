"""
Stage-by-stage dashboard for a single --idx run of run_pipeline.sh.

Renders every intermediate artifact the pipeline already produces (input
normal map, MVDiffusion multiview outputs, per-pixel primitive classification,
NEUS mesh, raw per-view primitive instances, graph-cut result, final
segmentation, final BRep .step) into one PNG grid, so you can see exactly
where a run starts going wrong without opening each file by hand.

Usage:
    python visualize_pipeline.py --idx 7
    python visualize_pipeline.py --idx 7 --out my_dashboard.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
USED_VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']


def load_image(path):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("RGB"))


def montage(paths, cols=3):
    imgs = [load_image(p) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    h, w = imgs[0].shape[:2]
    imgs = [np.array(Image.fromarray(im).resize((w, h))) if im.shape[:2] != (h, w) else im for im in imgs]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return canvas


def show_image(ax, img, title):
    ax.axis('off')
    ax.set_title(title, fontsize=9)
    if img is None:
        ax.text(0.5, 0.5, "N/A\n(not generated yet)", ha='center', va='center', fontsize=9, color='gray')
        return
    ax.imshow(img)


def show_mesh(ax, mesh_path, title, max_faces=25000):
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    if mesh_path is None or not os.path.exists(mesh_path):
        ax.text2D(0.5, 0.5, "N/A\n(not generated yet)", ha='center', va='center', fontsize=9, color='gray', transform=ax.transAxes)
        return
    mesh = trimesh.load(mesh_path, process=False)
    if len(mesh.faces) > max_faces:
        idx = np.random.choice(len(mesh.faces), max_faces, replace=False)
        idx.sort()
        faces = mesh.faces[idx]
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            colors = mesh.visual.face_colors[idx][:, :3] / 255.0
        else:
            colors = 'lightgray'
    else:
        faces = mesh.faces
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            colors = mesh.visual.face_colors[:, :3] / 255.0
        else:
            colors = 'lightgray'
    tri_verts = mesh.vertices[faces]
    coll = Poly3DCollection(tri_verts, facecolor=colors, edgecolor='none')
    ax.add_collection3d(coll)
    vmin, vmax = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    center = (vmin + vmax) / 2
    radius = max((vmax - vmin).max() / 2, 1e-6)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=25, azim=45)
    ax.text2D(0.02, 0.02, f"{len(mesh.faces)} faces", transform=ax.transAxes, fontsize=7, color='gray')


def tessellate_step(step_path, tol=0.1):
    if not os.path.exists(step_path):
        return None
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()

    BRepMesh_IncrementalMesh(shape, tol)
    verts, tris = [], []
    offset = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = face.Location()
        tri_data = BRep_Tool.Triangulation(face, loc)
        if tri_data is not None:
            trsf = loc.Transformation()
            n = tri_data.NbNodes()
            for i in range(1, n + 1):
                p = tri_data.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            nt = tri_data.NbTriangles()
            for i in range(1, nt + 1):
                a, b, c = tri_data.Triangle(i).Get()
                tris.append((a - 1 + offset, b - 1 + offset, c - 1 + offset))
            offset += n
        exp.Next()
    if not verts:
        return None
    return trimesh.Trimesh(np.array(verts), np.array(tris), process=False)


def show_step_file(ax, step_path, title):
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    if step_path is None or not os.path.exists(step_path):
        ax.text2D(0.5, 0.5, "N/A\n(not generated yet)", ha='center', va='center', fontsize=9, color='gray', transform=ax.transAxes)
        return
    mesh = tessellate_step(step_path)
    if mesh is None or len(mesh.faces) == 0:
        ax.text2D(0.5, 0.5, "failed to tessellate .step", ha='center', va='center', fontsize=9, color='red', transform=ax.transAxes)
        return
    tri_verts = mesh.vertices[mesh.faces]
    coll = Poly3DCollection(tri_verts, facecolor='steelblue', edgecolor='black', linewidths=0.1)
    ax.add_collection3d(coll)
    vmin, vmax = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    center = (vmin + vmax) / 2
    radius = max((vmax - vmin).max() / 2, 1e-6)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=25, azim=45)
    ax.text2D(0.02, 0.02, f"{len(mesh.faces)} faces", transform=ax.transAxes, fontsize=7, color='gray')


def build_stages(idx, cfg_name, vis_dir, config_dir, step_path, input_normal):
    masked_paths = [os.path.join(config_dir, f"masked_colors_{v}.png") for v in USED_VIEWS]
    # (slug, kind, render_args) -- kind picks which show_* function renders it
    return [
        ("1_input_normal", "image", load_image(input_normal), "1. Input normal map\n(GeoWizard, Step0)"),
        ("2_diffusion_colors", "image", load_image(os.path.join(config_dir, "all_colors.png")), "2. MVDiffusion\n6-view colors"),
        ("3_diffusion_normals", "image", load_image(os.path.join(config_dir, "all_normal.png")), "3. MVDiffusion\n6-view normals (decoder)"),
        ("4_primitive_classification", "image", montage(masked_paths, cols=3), "4. Primitive classification\n(per-pixel label * 50, 6 views)"),
        ("5_neus_mesh", "mesh", os.path.join(vis_dir, "5_neus_mesh.ply"), "5. NEUS mesh\n(no labels)"),
        ("6_primitive_instances_raw", "mesh", os.path.join(vis_dir, "6_primitive_instances_raw.ply"), "6. Raw primitive instances\n(before graph cut)"),
        ("7_graphcut", "mesh", os.path.join(vis_dir, "7_graphcut.ply"), "7. Graph-cut result"),
        ("8_final_segmentation", "mesh", os.path.join(vis_dir, "8_final_segmentation.ply"), "8. Final segmentation\n(post label-smoothing)"),
        ("9_final_brep", "step", step_path, "9. Final BRep (.step)"),
    ]


def render_stage(ax, kind, data, title):
    if kind == "image":
        show_image(ax, data, title)
    elif kind == "mesh":
        show_mesh(ax, data, title)
    elif kind == "step":
        show_step_file(ax, data, title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--cfg", default="cropsize-256-cfg3.0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-dashboard", action="store_true", help="skip the combined dashboard.png, only write per-stage files")
    args = parser.parse_args()

    idx = args.idx
    revision = f"{idx}_0deepcad"
    config_dir = os.path.join(PROJECT_ROOT, "test_outputs", args.cfg, revision)
    vis_dir = os.path.join(PROJECT_ROOT, "neus", "temp_mid_outputs", f"vis_{revision}")
    step_path = os.path.join(PROJECT_ROOT, "neus", "temp_mid_outputs", f"temp_scve{revision}.step")
    input_normal = os.path.join(PROJECT_ROOT, "our_inputs", "test_real_images", f"testnormal_{idx}_0.png")
    os.makedirs(vis_dir, exist_ok=True)

    stages = build_stages(idx, args.cfg, vis_dir, config_dir, step_path, input_normal)

    # 0) tessellate the final .step into a .ply too, so it can be opened/rotated
    # in MeshLab/Blender/VSCode's 3D viewer without a dedicated STEP viewer.
    # (stages 5-8 are already .ply on disk from test_real_images.py itself.)
    if os.path.exists(step_path):
        step_mesh = tessellate_step(step_path)
        if step_mesh is not None and len(step_mesh.faces) > 0:
            ply_path = os.path.join(vis_dir, "9_final_brep.ply")
            step_mesh.export(ply_path)
            print(f"saved {ply_path}")

    # 1) each stage as its own separate file
    for slug, kind, data, title in stages:
        is_3d = kind in ("mesh", "step")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)
        render_stage(ax, kind, data, title)
        stage_path = os.path.join(vis_dir, f"{slug}.png")
        plt.tight_layout()
        plt.savefig(stage_path, dpi=150)
        plt.close(fig)
        print(f"saved {stage_path}")

    # 2) combined dashboard (nice-to-have overview; skip with --no-dashboard)
    if not args.no_dashboard:
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f"Pipeline dashboard — idx={idx}", fontsize=13)
        for i, (slug, kind, data, title) in enumerate(stages, start=1):
            is_3d = kind in ("mesh", "step")
            ax = fig.add_subplot(3, 4, i, projection='3d') if is_3d else fig.add_subplot(3, 4, i)
            render_stage(ax, kind, data, title)
        out_path = args.out or os.path.join(vis_dir, "dashboard.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved dashboard -> {out_path}")


if __name__ == "__main__":
    main()
