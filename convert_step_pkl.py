import os
import pickle
import random
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_BSplineCurve
)


def extract_step_features(step_path):
    """STEP 파일에서 특징 및 위상 정보 추출"""

    reader = STEPControl_Reader()

    if reader.ReadFile(step_path) != 1:
        return None

    reader.TransferRoots()
    shape = reader.OneShape()

    unique_faces = []
    unique_edges = []
    unique_vertices = []

    def get_shape_index(shape_list, target_shape):
        for i, s in enumerate(shape_list):
            if s.IsSame(target_shape):
                return i

        shape_list.append(target_shape)
        return len(shape_list) - 1

    # Face / Edge / Vertex 수집
    for abs_type, shape_list in [
        (TopAbs_FACE, unique_faces),
        (TopAbs_EDGE, unique_edges),
        (TopAbs_VERTEX, unique_vertices),
    ]:
        exp = TopExp_Explorer(shape, abs_type)

        while exp.More():
            get_shape_index(shape_list, exp.Current())
            exp.Next()

    num_faces = len(unique_faces)
    num_edges = len(unique_edges)
    num_vertices = len(unique_vertices)

    if num_faces == 0 or num_edges == 0:
        return None

    # Adjacency Matrix
    FaceEdgeAdj = np.zeros((num_faces, num_edges), dtype=np.float32)
    EdgeVertexAdj = np.zeros((num_edges, num_vertices), dtype=np.float32)

    # Face -> Edge
    for f_idx, face in enumerate(unique_faces):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge = exp.Current()
            e_idx = get_shape_index(unique_edges, edge)
            FaceEdgeAdj[f_idx, e_idx] = 1.0
            exp.Next()

    # Edge -> Vertex
    for e_idx, edge in enumerate(unique_edges):
        exp = TopExp_Explorer(edge, TopAbs_VERTEX)
        while exp.More():
            vertex = exp.Current()
            v_idx = get_shape_index(unique_vertices, vertex)
            EdgeVertexAdj[e_idx, v_idx] = 1.0
            exp.Next()

    # Degree Feature
    face_degree = np.sum(FaceEdgeAdj, axis=1)
    edge_degree = np.sum(FaceEdgeAdj, axis=0)
    vertex_degree = np.sum(EdgeVertexAdj, axis=0)
    
    # Bounding Box
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    cz = (zmin + zmax) * 0.5

    scale = max(
        xmax - xmin,
        ymax - ymin,
        zmax - zmin
    )

    if scale < 1e-8:
        scale = 1.0   
    
    # Surface Features
    surf_wcs = []

    for f_idx, face in enumerate(unique_faces):
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        
        adaptor = BRepAdaptor_Surface(face)
        
        # 중심
        cx_f = (props.CentreOfMass().X() - cx) / scale
        cy_f = (props.CentreOfMass().Y() - cy) / scale
        cz_f = (props.CentreOfMass().Z() - cz) / scale
        
        area = props.Mass() / (scale ** 2)
        
        # Normal
        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) * 0.5
        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) * 0.5
        
        s1 = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
        
        if s1.IsNormalDefined():
            n = s1.Normal()
            nx, ny, nz = n.X(), n.Y(), n.Z()     
        else:
            nx, ny, nz = 0.0, 0.0, 0.0
        
        # surface type
        surf_type = adaptor.GetType()
        type_onehot = [
            int(surf_type == GeomAbs_Plane), # 평면
            int(surf_type == GeomAbs_Cylinder), # 윈기둥
            int(surf_type == GeomAbs_Cone), # 원뿔
            int(surf_type == GeomAbs_Sphere), # 구
            int(surf_type == GeomAbs_BSplineSurface), # 자유곡면
        ]
        
        # edge length stats (boundary 기반)
        edge_indices = np.where(FaceEdgeAdj[f_idx] == 1)[0]
        edge_lengths = []

        for e_idx in edge_indices:
            props_e = GProp_GProps()
            brepgprop.LinearProperties(unique_edges[e_idx], props_e)
            edge_lengths.append(props_e.Mass())

        if len(edge_lengths) > 0:
            edge_lengths = np.array(edge_lengths) / scale
            edge_len_mean = edge_lengths.mean()
            edge_len_std = edge_lengths.std()
            perimeter = edge_lengths.sum()
        else:
            edge_len_mean = edge_len_std = perimeter = 0.0

        # aspect ratio 근사
        if len(edge_lengths) >= 2:
            max_l = np.max(edge_lengths)
            min_l = np.min(edge_lengths) + 1e-6
            aspect_ratio = max_l / min_l
        else:
            aspect_ratio = 1.0
        
        surf_wcs.append([
            cx_f, cy_f, cz_f, # 정규화된 중심 좌표
            area, # 면적
            face_degree[f_idx], # 면의 edge 수 (사각형 = 4, 삼각형 = 3)
            nx, ny, nz, # 법선 벡터
            edge_len_mean, # 테두리의 평균 길이
            edge_len_std, # 테두리의 표준편차
            perimeter, # 면의 둘레
            aspect_ratio, # 가로, 세로 비율
            *type_onehot # 면의 형상 정보
        ])

    # Edge Features
    edge_wcs = []

    for e_idx, edge in enumerate(unique_edges):
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)

        cx_e = (props.CentreOfMass().X() - cx) / scale
        cy_e = (props.CentreOfMass().Y() - cy) / scale
        cz_e = (props.CentreOfMass().Z() - cz) / scale
        
        length = props.Mass() / scale
        
        adaptor = BRepAdaptor_Curve(edge)
        curve_type = adaptor.GetType()

        type_onehot = [
            int(curve_type == GeomAbs_Line), # 직선
            int(curve_type == GeomAbs_Circle), # 둥근 곡선
            int(curve_type == GeomAbs_BSplineCurve), # 자유 곡선
        ]
        
        # vertex endpoints
        exp = TopExp_Explorer(edge, TopAbs_VERTEX)
        pts = []
        while exp.More():
            p = BRep_Tool.Pnt(exp.Current())
            pts.append([p.X(), p.Y(), p.Z()])
            exp.Next()

        if len(pts) >= 2:
            p1, p2 = np.array(pts[0]), np.array(pts[1])
            direction = (p2 - p1)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            dx, dy, dz = direction
            chord = np.linalg.norm(p2 - p1) / scale
        else:
            dx = dy = dz = chord = 0.0

        edge_wcs.append([
            cx_e, cy_e, cz_e, # 정규화된 중심 좌표
            length, # 길이
            edge_degree[e_idx], # 주변 위상 개수
            dx, dy, dz, # 방향 벡터
            chord, # 선의 직선 거리
            *type_onehot # 선 정보
        ])

    # Vertex Features
    vertex_wcs = []

    for i, vertex in enumerate(unique_vertices):
        pnt = BRep_Tool.Pnt(vertex)

        x = (pnt.X() - cx) / scale
        y = (pnt.Y() - cy) / scale
        z = (pnt.Z() - cz) / scale
        
        # incident edge length stats
        connected_edges = np.where(EdgeVertexAdj[:, i] == 1)[0]
        lengths = []

        for e_idx in connected_edges:
            props_e = GProp_GProps()
            brepgprop.LinearProperties(unique_edges[e_idx], props_e)
            lengths.append(props_e.Mass())

        if len(lengths) > 0:
            lengths = np.array(lengths) / scale
            mean_l = lengths.mean()
            std_l = lengths.std()
        else:
            mean_l = std_l = 0.0

        vertex_wcs.append([
            x, y, z, # 정규화된 좌표
            vertex_degree[i], # 연결된 모서리 개수
            mean_l, # 인접 모서리 평균 길이
            std_l, # 인접 모서리 길이 표준편차
        ])
    
    # return
    return {
        "surf_wcs": np.array(surf_wcs, dtype=np.float32),
        "edge_wcs": np.array(edge_wcs, dtype=np.float32),
        "vertex_wcs": np.array(vertex_wcs, dtype=np.float32),
        "FaceEdgeAdj": FaceEdgeAdj,
        "EdgeVertexAdj": EdgeVertexAdj,
    }


def corrupt_topology(face_edge_adj, edge_vertex_adj, surf_wcs, edge_wcs):
    input_FE = face_edge_adj.copy()
    input_EV = edge_vertex_adj.copy()

    face_labels = np.zeros(face_edge_adj.shape[0], dtype=np.int64)
    edge_labels = np.zeros(edge_vertex_adj.shape[0], dtype=np.int64)
    
    num_faces, num_edges = input_FE.shape
    
    face_centers = surf_wcs[:, :3]
    face_normals = surf_wcs[:, 5:8]  # nx, ny, nz

    edge_centers = edge_wcs[:, :3]
    
    ###########################################################
    # 난이도 기반 비율 설정
    ###########################################################
    difficulty = random.choice(["easy", "medium", "hard"])

    if difficulty == "easy":
        ratio = random.uniform(0.02, 0.05)
    elif difficulty == "medium":
        ratio = random.uniform(0.05, 0.1)
    else:
        ratio = random.uniform(0.1, 0.2)

    num_face_errors = max(1, int(num_faces * ratio))
    num_edge_errors = max(1, int(num_edges * ratio))

    ###########################################################
    # Dihedral 기반 similarity 함수 (각도 고려)
    ###########################################################
    def normal_similarity(n1, n2):
        cos_sim = np.dot(n1, n2) / (
            np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-6
        )
        return cos_sim  # 1에 가까울수록 비슷

    ###########################################################
    # 1. Dihedral-aware wrong connection
    ###########################################################
    def dihedral_wrong_connection():

        for _ in range(num_face_errors):
            f = random.randint(0, num_faces - 1)

            # 거리 + normal similarity 동시에 고려
            dists = np.linalg.norm(face_centers - face_centers[f], axis=1)
            sims = np.array([
                normal_similarity(face_normals[f], face_normals[i])
                for i in range(num_faces)
            ])

            # 가까우면서 방향도 비슷한 face들
            score = dists - sims * 0.5  # similarity 높으면 score 낮아짐
            candidates = np.argsort(score)[1:6]

            f2 = random.choice(candidates)

            candidate_edges = np.where(input_FE[f2] == 1)[0]
            if len(candidate_edges) == 0:
                continue

            e = random.choice(candidate_edges)

            if input_FE[f, e] == 0:
                input_FE[f, e] = 1
                face_labels[f] = 1
                edge_labels[e] = 1

    ###########################################################
    # 2. Dihedral-preserving edge swap
    ###########################################################
    def dihedral_edge_swap():
        
        for _ in range(num_edge_errors):
            e1 = random.randint(0, num_edges - 1)

            dists = np.linalg.norm(edge_centers - edge_centers[e1], axis=1)
            candidates = np.argsort(dists)[1:6]

            e2 = random.choice(candidates)

            f1_list = np.where(input_FE[:, e1] == 1)[0]
            f2_list = np.where(input_FE[:, e2] == 1)[0]

            if len(f1_list) == 0 or len(f2_list) == 0:
                continue

            f1 = random.choice(f1_list)
            f2 = random.choice(f2_list)

            # 🔥 핵심: normal이 비슷한 경우만 swap
            sim = normal_similarity(face_normals[f1], face_normals[f2])

            if sim < 0.7:  # 너무 다르면 skip
                continue

            input_FE[f1, e1] = 0
            input_FE[f2, e2] = 0

            input_FE[f1, e2] = 1
            input_FE[f2, e1] = 1

            face_labels[f1] = face_labels[f2] = 1
            edge_labels[e1] = edge_labels[e2] = 1

    ###########################################################
    # 3. subtle topology break (각도 기반 제거)
    ###########################################################
    def dihedral_partial_break():
        count = 0
        for e in range(num_edges):
            if count >= num_edge_errors:
                break

            faces = np.where(input_FE[:, e] == 1)[0]
            if len(faces) < 2:
                continue

            if len(faces) == 2:
                f1, f2 = faces

                sim = normal_similarity(
                    face_normals[f1],
                    face_normals[f2]
                )

                # 거의 평행한 경우만 끊기
                if sim > 0.8:
                    input_FE[f1, e] = 0

                    face_labels[f1] = 1
                    edge_labels[e] = 1
                    
                    count += 1

    ###########################################################
    # 4. soft non-manifold (유사 방향 face 추가)
    ###########################################################
    def dihedral_soft_non_manifold():
        count = 0
        
        for e in range(num_edges):
            
            if count >= num_edge_errors:
                break
            
            faces = np.where(input_FE[:, e] == 1)[0]

            if len(faces) == 2:
                f1 = faces[0]

                for f_new in range(num_faces):
                    if input_FE[f_new, e] == 0:
                        sim = normal_similarity(
                            face_normals[f1],
                            face_normals[f_new]
                        )

                        if sim > 0.7:
                            input_FE[f_new, e] = 1

                            face_labels[f_new] = 1
                            edge_labels[e] = 1
                            
                            count += 1
                            break

    ###########################################################
    # 실행
    ###########################################################
    operations = [
        dihedral_wrong_connection,
        dihedral_edge_swap,
        dihedral_partial_break,
        dihedral_soft_non_manifold,
    ]

    selected = random.sample(operations, random.randint(1, 3))

    for op in selected:
        op()

    return (
        input_FE,
        input_EV,
        face_labels,
        edge_labels,
        {
            "ops": [op.__name__ for op in selected],
            "ratio": ratio,
            "difficulty": difficulty
        }
    )

# 여러 파일
def build_dataset(root_dir, output_dir, max_files=1000):

    os.makedirs(output_dir, exist_ok=True)

    file_count = 1153
    # [1153/1500] 00001157_bc50ee5c437c400aa2ee74a6_step_000.step -> case_1152.pkl 완

    print(f"기존 데이터 {file_count}개 발견")

    processed_step_count = 0

    for current_root, _, files in os.walk(root_dir):

        files.sort()

        for file in files:

            if not file.lower().endswith((".step", ".stp")):
                continue

            # 이미 처리한 STEP은 건너뜀
            if processed_step_count < file_count:
                processed_step_count += 1
                continue

            if file_count >= max_files:
                print(f"\n목표 수량 {max_files}개 완료")
                return

            step_path = os.path.join(current_root, file)

            try:

                features = extract_step_features(step_path)

                if features is None:
                    processed_step_count += 1
                    continue

                (
                    input_FE,
                    input_EV,
                    face_labels,
                    edge_labels,
                    error_type,
                ) = corrupt_topology(
                    features["FaceEdgeAdj"],
                    features["EdgeVertexAdj"],
                    features["surf_wcs"],
                    features["edge_wcs"],
                )

                data = {
                    "surf_wcs": features["surf_wcs"],
                    "edge_wcs": features["edge_wcs"],
                    "vertex_wcs": features["vertex_wcs"],

                    "input_FaceEdgeAdj": input_FE,
                    "input_EdgeVertexAdj": input_EV,

                    "target_FaceEdgeAdj": features["FaceEdgeAdj"],
                    "target_EdgeVertexAdj": features["EdgeVertexAdj"],

                    "changed_face_label": face_labels,
                    "changed_edge_label": edge_labels,

                    "error_type": error_type,
                }

                case_name = f"case_{file_count:04d}"

                save_path = os.path.join(
                    output_dir,
                    f"{case_name}.pkl"
                )

                with open(save_path, "wb") as f:
                    pickle.dump(data, f)

                file_count += 1
                processed_step_count += 1

                print(f"[{file_count}/{max_files}] {file} -> {case_name}.pkl")

            except Exception as e:

                processed_step_count += 1

                print(f"[ERROR] {file}: {e}")

if __name__ == "__main__":

    ABC_ROOT = (
        r"C:\BRepGen_conda\GNN\step_file"
    )

    OUTPUT_DATASET = (
        r"C:\BRepGen_conda\GNN\processed_dataset"
    )

    build_dataset(
        root_dir=ABC_ROOT,
        output_dir=OUTPUT_DATASET,
        max_files=1500,
    )
    
# # 단일 파일
# def build_dataset_single_file(step_path, output_dir, case_id=0):

#     os.makedirs(output_dir, exist_ok=True)

#     print(f"처리 대상: {step_path}")

#     try:
#         features = extract_step_features(step_path)

#         if features is None:
#             print("feature 추출 실패")
#             return

#         (
#             input_FE,
#             input_EV,
#             face_labels,
#             edge_labels,
#             error_type,
#         ) = corrupt_topology(
#             features["FaceEdgeAdj"],
#             features["EdgeVertexAdj"]
#         )

#         data = {
#             "surf_wcs": features["surf_wcs"],
#             "edge_wcs": features["edge_wcs"],

#             "input_FaceEdgeAdj": input_FE,
#             "input_EdgeVertexAdj": input_EV,

#             "target_FaceEdgeAdj": features["FaceEdgeAdj"],
#             "target_EdgeVertexAdj": features["EdgeVertexAdj"],

#             "changed_face_label": face_labels,
#             "changed_edge_label": edge_labels,

#             "error_type": error_type,
#         }

#         case_name = f"case_{case_id:04d}"

#         save_path = os.path.join(output_dir, f"{case_name}.pkl")

#         with open(save_path, "wb") as f:
#             pickle.dump(data, f)

#         print(f"완료 -> {save_path}")

#     except Exception as e:
#         print(f"[ERROR] {step_path}: {e}")

# if __name__ == "__main__":

#     TARGET_FILE = r"C:\BRepGen_conda\GNN\step_file\00000754\00000754_8ab50540e9804365a8717ec0_step_001.step"

#     OUTPUT_DATASET = r"C:\BRepGen_conda\GNN\processed_dataset"

#     build_dataset_single_file(
#         step_path=TARGET_FILE,
#         output_dir=OUTPUT_DATASET,
#         case_id=754 
#     )