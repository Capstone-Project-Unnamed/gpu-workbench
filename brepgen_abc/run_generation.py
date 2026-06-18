#!/usr/bin/env python3
"""
BrepGen Unconditional Generation - 최소 버전
사용: python run_generation.py --batches 5
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# ============= 경로 설정 =============
BREPGEN_DIR = Path("BrepGen")  # BrepGen이 현재 디렉토리에 있다고 가정
os.chdir(BREPGEN_DIR)
sys.path.insert(0, str(BREPGEN_DIR.absolute()))

# ============= 설정 =============
EVAL_CONFIG = {
    'abc': {
        'surfpos_weight': 'abc_ldm_surfpos.pt',
        'surfz_weight': 'abc_ldm_surfz.pt',
        'edgepos_weight': 'abc_ldm_edgepos.pt',
        'edgez_weight': 'abc_ldm_edgez.pt',
        'surfvae_weight': 'abc_vae_surf.pt',
        'edgevae_weight': 'abc_vae_edge.pt',
        'save_folder': 'samples_abc',
        'batch_size': 4,
        'z_threshold': 0.2,
        'bbox_threshold': 0.08,
        'num_surfaces': 50,
        'num_edges': 40,
        'use_cf': False,
        'class_label': 'uncond'
    }
}

# ============= GPU 확인 =============
print("=" * 50)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (느림)'}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
print("=" * 50)

# ============= 설정 파일 생성 =============
if config_path.exists():
    print(f"✓ eval_config.yaml 있음 ({config_path.absolute()})")
    # 기존 파일 로드해서 모드 표시
else:
    with open('eval_config.yaml', 'w') as f:
        yaml.dump(EVAL_CONFIG, f)
    print("✓ eval_config.yaml 생성\n")

# ============= sample.py 실행 =============
import argparse
from tqdm import tqdm

# 파일 끝의 main 함수 직접 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture'], 
                       default='abc', help="생성 모드")
    parser.add_argument("--batches", type=int, default=1, 
                       help="반복 횟수 (batch_size=4라면 총 4*batches 샘플 생성)")
    args = parser.parse_args()
    
    # sample.py 코드 실행
    sample_code = open('sample.py').read()
    
    # while(True) → for loop으로 변경
    sample_code = sample_code.replace(
        "while(True):\n    sample(eval_args)",
        f"for _ in range({args.batches}):\n    sample(eval_args)"
    )
    
    # 실행
    exec(sample_code)
