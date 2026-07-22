# Stub: visualization disabled (headless environment)
__all__ = [
    'render_all_patches', 'render_seg_vertex_scalar', 'render_seg_face_scalar',
    'render_seg_select_face', 'render_seg_select_vertices', 'render_all_instances',
    'render_edges', 'render_mesh', 'render_primitives', 'render_points', 'save_images',
    'render_simple_trimesh_select_faces', 'render_simple_trimesh', 'render_cad_occ',
    'load_cache_dill', 'render_brep',
]

import numpy as np

def render_all_patches(*a, **kw): pass
def render_seg_vertex_scalar(*a, **kw): pass
def render_seg_face_scalar(*a, **kw): pass
def render_seg_select_face(*a, **kw): pass
def render_seg_select_vertices(*a, **kw): pass
def render_all_instances(*a, **kw): pass
def render_edges(*a, **kw): pass
def render_mesh(*a, **kw): pass
def render_primitives(*a, **kw): pass
def render_points(*a, **kw): pass
def render_simple_trimesh_select_faces(*a, **kw): pass
def render_simple_trimesh(*a, **kw): pass
def render_cad_occ(*a, **kw): pass
def render_brep(*a, **kw): pass
def save_images(*a, **kw): return None, None, None

def load_cache_dill(path):
    import dill
    with open(path, 'rb') as f:
        return dill.load(f)
