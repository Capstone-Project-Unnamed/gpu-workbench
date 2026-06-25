#!/bin/bash
# BrepGen 환경 설정 스크립트 (conda 환경용)
# 사용: bash setup_brepgen.sh

set -e  # 에러 발생시 즉시 종료

echo "=========================================="
echo "BrepGen 환경 설정"
echo "=========================================="

# 1. 기본 패키지 설치
echo "[1/4] 기본 패키지 설치..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y -q

# 2. 필수 라이브러리
echo "[2/4] 필수 라이브러리 설치..."
conda install -c conda-forge pythonocc-core -y -q

pip install --upgrade pip -q
pip install \
    huggingface_hub==0.23.0 \
    diffusers==0.27.0 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    scipy trimesh scikit-learn pyyaml -q

pip install git+https://github.com/AutodeskAILab/occwl -q

# 3. 선택사항: chamferdist (필요한 경우)
# pip install chamferdist -q

# 4. BrepGen 레포 클론
echo "[3/4] BrepGen 레포 클론..."
if [ ! -d "BrepGen" ]; then
    git clone https://github.com/samxuxiang/BrepGen.git
else
    echo "BrepGen 디렉토리 이미 존재"
fi

cd BrepGen

# 5. 모델 가중치 확인
echo "[4/4] 모델 가중치 확인..."
required_files=(
    "abc_vae_surf.pt"
    "abc_vae_edge.pt"
    "abc_ldm_surfpos.pt"
    "abc_ldm_surfz.pt"
    "abc_ldm_edgepos.pt"
    "abc_ldm_edgez.pt"
)

missing=0
for f in "${required_files[@]}"; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ✓ $f ($size)"
    else
        echo "  ✗ $f (필요!)"
        missing=$((missing+1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "⚠️  $missing개 파일이 없습니다."
    echo "Google Drive에서 다운로드 후 BrepGen 디렉토리에 배치하세요:"
    echo "  VAE weights: https://drive.google.com/drive/folders/18Ib9L0kpFf4ylZIRTCYFhXZB_GVIUm53"
    echo "  LDM weights: https://drive.google.com/drive/folders/1hv7ZUcU-L3J0LiONK60-TEh7sAN0zfve"
else
    echo ""
    echo "✅ 모든 준비 완료!"
fi