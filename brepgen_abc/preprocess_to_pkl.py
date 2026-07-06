#!/usr/bin/env python3
"""
STEP → PKL 변환 (BrepGen 형식)

사용:
  python preprocess_to_pkl.py --input samples_abc --output samples_abc_pkl
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

# BrepGen 임포트
BREPGEN_DIR = Path("BrepGen")
sys.path.insert(0, str(BREPGEN_DIR.absolute()))

from OCC.Extend.DataExchange import read_step_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
import numpy as np

# ============= STEP → pkl 변환 함수 =============

def load_step_file(step_path):
    """STEP 파일 로드"""
    try:
        shape = read_step_file(str(step_path))
        return shape
    except Exception as e:
        print(f"  ✗ 로드 실패: {e}")
        return None

def extract_face_features(face):
    """Face에서 특성 추출 (간략화 버전)"""
    # 실제 구현: UV grid sampling, bounding box 등
    # 현재는 placeholder
    bbox = (0, 0, 0, 1, 1, 1)  # (x_min, y_min, z_min, x_max, y_max, z_max)
    feature = np.random.randn(16, 3)  # 플레이스홀더
    return {"bbox": bbox, "feature": feature}

def extract_edge_features(edge):
    """Edge에서 특성 추출"""
    bbox = (0, 0, 0, 1, 1, 1)
    feature = np.random.randn(4, 3)
    return {"bbox": bbox, "feature": feature}

def step_to_pkl_dict(step_path):
    """
    STEP → pkl 딕셔너리 변환
    
    Returns:
        {
            'surfaces': [...],      # face features
            'edges': [...],         # edge features
            'vertices': [...],      # vertex coordinates
            'topology': {...}       # face-edge 연결 정보
        }
    """
    shape = load_step_file(step_path)
    if shape is None:
        return None
    
    # Face 추출
    faces = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        feature = extract_face_features(face)
        faces.append(feature)
        face_explorer.Next()
    
    # Edge 추출
    edges = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        feature = extract_edge_features(edge)
        edges.append(feature)
        edge_explorer.Next()
    
    # Vertex 추출
    vertices = []
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        # 좌표 추출 (간략)
        vertices.append((0, 0, 0))  # placeholder
        vertex_explorer.Next()
    
    pkl_data = {
        'surfaces': faces,
        'edges': edges,
        'vertices': vertices,
        'num_surfaces': len(faces),
        'num_edges': len(edges),
        'num_vertices': len(vertices),
    }
    
    return pkl_data

# ============= 메인 처리 =============

def main():
    parser = argparse.ArgumentParser(description="STEP → PKL 변환")
    parser.add_argument("--input", type=str, required=True,
                       help="입력 폴더 (생성된 샘플들)")
    parser.add_argument("--output", type=str, required=True,
                       help="출력 폴더 (pkl 저장)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # STEP 파일 목록
    step_files = sorted(input_dir.glob("*.step"))
    
    print(f"{'='*60}")
    print(f"STEP → PKL 변환")
    print(f"{'='*60}")
    print(f"입력: {input_dir}")
    print(f"출력: {output_dir}")
    print(f"파일 수: {len(step_files)}\n")
    
    # 변환
    success, failed = 0, 0
    for step_file in tqdm(step_files, desc="Converting"):
        pkl_data = step_to_pkl_dict(step_file)
        
        if pkl_data is None:
            failed += 1
            continue
        
        # pkl로 저장
        pkl_path = output_dir / step_file.name.replace('.step', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        success += 1
    
    print(f"\n{'='*60}")
    print(f"✓ 성공: {success}")
    print(f"✗ 실패: {failed}")
    print(f"저장 위치: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()