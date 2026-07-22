#!/bin/bash
set -e

BASE_DIR="/mnt/sdb"
ENV_PATH="${BASE_DIR}/envs/caddreamer"
PYTHON="${ENV_PATH}/bin/python"
PIP="${ENV_PATH}/bin/pip"

# 모든 캐시/임시 파일을 /mnt/sdb로
export TMPDIR="${BASE_DIR}/tmp"
export TEMP="${BASE_DIR}/tmp"
export TMP="${BASE_DIR}/tmp"
export PIP_CACHE_DIR="${BASE_DIR}/pip_cache"
export CONDA_PKGS_DIRS="${BASE_DIR}/conda_pkgs"
export XDG_CACHE_HOME="${BASE_DIR}/xdg_cache"
export HF_HOME="${BASE_DIR}/hf_cache"
export TORCH_HOME="${BASE_DIR}/torch_cache"
export TORCH_EXTENSIONS_DIR="${BASE_DIR}/torch_extensions"
export CUDA_CACHE_PATH="${BASE_DIR}/cuda_cache"
export NUMBA_CACHE_DIR="${BASE_DIR}/numba_cache"
export TRITON_CACHE_DIR="${BASE_DIR}/triton_cache"
export PYTHONPYCACHEPREFIX="${BASE_DIR}/pycache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" \
         "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
         "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

echo "=== [1/6] conda 환경 생성 ==="
if [ ! -f "${PYTHON}" ]; then
    conda env remove -p ${ENV_PATH} -y 2>/dev/null || true
    conda create -p ${ENV_PATH} python=3.10 -y
else
    echo "환경 이미 존재, 스킵"
fi

echo "=== [2/6] pip 업그레이드 ==="
${PIP} install --upgrade pip

echo "=== [3/6] requirements.txt 일괄 설치 ==="
${PIP} install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    torch-scatter==2.1.1+pt20cu118 \
    torch-sparse==0.6.17+pt20cu118 \
    torch-spline-conv==1.2.2+pt20cu118 \
    torch_efficient_distloss==0.1.3 \
    torchmetrics==1.4.1 \
    pytorch-lightning==1.9.5 \
    numpy==1.24.0 \
    scipy==1.15.1 \
    pandas==2.2.2 \
    matplotlib==3.9.1.post1 \
    opencv-python-headless==4.11.0.86 \
    pillow==10.4.0 \
    scikit-learn==1.5.1 \
    scikit-image==0.25.0 \
    h5py==3.11.0 \
    tqdm==4.66.5 \
    requests==2.32.3 \
    transformers==4.44.0 \
    diffusers==0.19.3 \
    huggingface-hub==0.24.5 \
    accelerate==0.33.0 \
    tokenizers==0.19.1 \
    safetensors==0.4.4 \
    einops==0.8.0 \
    omegaconf==2.2.3 \
    xformers==0.0.22 \
    tensorboard==2.17.0 \
    tensorboardX==2.6.2.2 \
    trimesh==3.18.1 \
    open3d==0.18.0 \
    pymeshlab==2023.12.post1 \
    meshio==5.3.5 \
    potpourri3d==1.1.0 \
    polyscope==2.3.0 \
    pyvista==0.44.1 \
    plyfile==1.1 \
    numpy-stl==3.1.2 \
    shapely==2.0.6 \
    networkx==3.2.1 \
    transforms3d==0.4.2 \
    pyquaternion==0.9.9 \
    imageio==2.34.2 \
    rembg==2.0.58 \
    carvekit_colab==4.1.2 \
    segment-anything==1.0 \
    rich==13.7.1 \
    loguru==0.7.2 \
    coloredlogs==15.0.1 \
    ConfigArgParse==1.7 \
    absl-py==2.1.0 \
    addict==2.4.0 \
    ftfy==6.2.3 \
    geomdl==5.3.1 \
    GitPython==3.1.43 \
    gpustat==0.6.0 \
    icecream==2.1.0 \
    joblib==1.4.2 \
    llvmlite==0.43.0 \
    numba==0.60.0 \
    onnxruntime==1.18.1 \
    piq==0.8.0 \
    plotly==5.23.0 \
    psutil==6.0.0 \
    pydantic==2.8.2 \
    pydub==0.25.1 \
    pyhocon==0.3.57 \
    PyMatting==1.1.12 \
    pyransac3d==0.6.0 \
    python-louvain==0.16 \
    retrying==1.3.4 \
    Rtree==1.3.0 \
    svgwrite==1.4.3 \
    sympy==1.12 \
    tenacity==9.0.0 \
    tifffile==2024.7.24 \
    webdataset==0.2.100 \
    xxhash==3.4.1 \
    zstandard==0.23.0 \
    dill \
    optimparallel \
    lapsolver==1.1.0 \
    bleach==5.0.1 \
    decord==0.6.0 \
    bitsandbytes==0.35.4 \
    nvitop \
    lightning-utilities==0.11.6 \
    pooch==1.8.2 \
    pyarrow==17.0.0 \
    Cython==0.29.37 \
    iopath==0.1.10

echo "=== [3.5/6] openmesh (별도 설치) ==="
${PIP} install openmesh==1.2.1 || echo "Warning: openmesh 스킵"

echo "=== [4/6] PyTorch3D (pre-built wheel) ==="
${PIP} install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git" \
    || echo "Warning: PyTorch3D 빌드 실패, 스킵"

echo "=== [5/6] pythonocc (conda로 설치 — 빌드 불필요) ==="
conda install -p ${ENV_PATH} -c conda-forge pythonocc-core=7.7.2 -y \
    || echo "Warning: pythonocc 설치 실패, 스킵"

echo "=== [6/6] nerfacc + tinycudann ==="
${PIP} install nerfacc==0.3.3 || echo "Warning: nerfacc 스킵"
${PIP} install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    || echo "Warning: tinycudann 빌드 실패, 스킵"

echo ""
echo "=== 설치 검증 ==="
${PYTHON} -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import diffusers; print('diffusers:', diffusers.__version__)
import transformers; print('transformers:', transformers.__version__)
import trimesh; print('trimesh:', trimesh.__version__)
import einops; print('einops:', einops.__version__)
import omegaconf; print('omegaconf:', omegaconf.__version__)
import pytorch_lightning; print('pytorch_lightning:', pytorch_lightning.__version__)
try: import pytorch3d; print('pytorch3d:', pytorch3d.__version__)
except Exception as e: print('pytorch3d: FAILED -', e)
try: from OCC.Core.TopoDS import TopoDS_Shape; print('pythonOCC: ok')
except Exception as e: print('pythonOCC: FAILED -', e)
try: import nerfacc; print('nerfacc:', nerfacc.__version__)
except Exception as e: print('nerfacc: FAILED -', e)
try: import tinycudann; print('tinycudann: ok')
except Exception as e: print('tinycudann: FAILED -', e)
"

echo ""
echo "=== 완료! 활성화: conda activate ${ENV_PATH} ==="
