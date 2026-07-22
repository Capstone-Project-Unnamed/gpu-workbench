set -e
set -o pipefail
cd /mnt/sdb/TMEMJ/CADDreamer-jjy

export LD_LIBRARY_PATH=/mnt/sdb/envs/caddreamer-jjy/lib:$LD_LIBRARY_PATH
# pymeshlab이 번들한 구버전 libnghttp2.so.14가 먼저 로드되면 vtk의 libcurl이 필요한
# 심볼(nghttp2_option_set_no_rfc9113_...)을 못 찾고 죽는다 (동일 soname 충돌).
# env의 최신 libnghttp2를 프로세스 시작 시 먼저 강제 로드해서 import 순서와 무관하게 해결.
export LD_PRELOAD=/mnt/sdb/envs/caddreamer-jjy/lib/libnghttp2.so.14:$LD_PRELOAD
export HF_HOME=/mnt/sdb/hf_cache
export PYTHONPATH=/mnt/sdb/TMEMJ/CADDreamer-jjy/pyransac/cmake-build-release:/mnt/sdb/TMEMJ/CADDreamer-jjy
# ninja(C++ 확장 빌드용)가 PATH에 있어야 nerfacc JIT 컴파일 가능
export PATH=/mnt/sdb/envs/caddreamer-jjy/bin:$PATH
# CUDA 확장 컴파일 캐시를 home이 아닌 /mnt/sdb로 (home 사용 금지)
export TORCH_EXTENSIONS_DIR=/mnt/sdb/torch_extensions
mkdir -p /mnt/sdb/torch_extensions
PYTHON=/mnt/sdb/envs/caddreamer-jjy/bin/python3

# GeoWizard needs diffusers 0.25 (CADDreamer pins 0.19) → separate env
GEOWIZARD_DIR=/mnt/sdb/GeoWizard/geowizard
GEOWIZARD_PYTHON=/mnt/sdb/envs/geowizard/bin/python
GEOWIZARD_LIB=/mnt/sdb/envs/geowizard/lib

# defaults
STEP=${1:-all}
IDX=0
GPU=0
REVIEW=False
DOMAIN=object
RGB_INPUT=""

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --idx)    IDX=$2;    shift 2 ;;
        --gpu)    GPU=$2;    shift 2 ;;
        --review) REVIEW=$2; shift 2 ;;
        --domain) DOMAIN=$2; shift 2 ;;
        --rgb)    RGB_INPUT=$2; shift 2 ;;
        *) shift ;;
    esac
done

NORMAL_INPUT="./our_inputs/test_real_images/testnormal_${IDX}_0.png"
# NEUS/BRep use the cfg3.0 multiview (sharper, more consistent than cfg1.0)
CONFIG_DIR="./test_outputs/cropsize-256-cfg3.0/${IDX}_0deepcad"

# ── 실행마다 별도 로그 파일 저장 (idx + 타임스탬프로 구분) ─────────────────
LOG_DIR="./cdlog"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
log_path() { echo "$LOG_DIR/${1}_idx${IDX}_${TS}.log"; }

# ── Step 0: RGB → front-view normal map (GeoWizard) ───────────────────────────
step0() {
    if [[ -z "$RGB_INPUT" ]]; then
        echo "Error: --rgb PATH required for step 0"
        exit 1
    fi
    if [[ ! -f "$RGB_INPUT" ]]; then
        echo "Error: RGB file not found: $RGB_INPUT"
        exit 1
    fi

    echo "=== Step 0: RGB → normal map (GeoWizard, domain=$DOMAIN) ==="
    local tmp_in=$(mktemp -d /mnt/sdb/tmp/geowizard_in_XXXX)
    local tmp_out=/mnt/sdb/tmp/geowizard_out_${IDX}
    mkdir -p "$tmp_out"

    # GeoWizard expects a directory of images
    cp "$RGB_INPUT" "$tmp_in/input.png"

    LD_LIBRARY_PATH="$GEOWIZARD_LIB:$LD_LIBRARY_PATH" \
    PYTHONPATH="$GEOWIZARD_DIR" HF_HOME=/mnt/sdb/hf_cache \
    $GEOWIZARD_PYTHON "$GEOWIZARD_DIR/run_infer.py" \
        --pretrained_model_path lemonaddie/geowizard \
        --input_dir "$tmp_in" \
        --output_dir "$tmp_out" \
        --domain "$DOMAIN" \
        --denoise_steps 10 \
        --ensemble_size 5 \
        --half_precision \
        2>&1 | tee "$(log_path step0)"

    # GeoWizard normal_colored is uint8 RGB [0,255] scaled as (normal+1)/2*255
    # CADDreamer load_normal reads it as uint8 PNG and applies img2normal=(px/255)*2-1
    # → perfect round-trip, no conversion needed
    local normal_png="$tmp_out/normal_colored/input_pred_colored.png"
    if [[ ! -f "$normal_png" ]]; then
        echo "Error: GeoWizard did not produce $normal_png"
        exit 1
    fi

    mkdir -p ./our_inputs/test_real_images
    cp "$normal_png" "$NORMAL_INPUT"
    rm -rf "$tmp_in"
    echo "=== Step 0 완료 → $NORMAL_INPUT (log: $(log_path step0)) ==="
}

# ── Step 1: front-view normal → 6-view normals (MVDiffusion / Wonder3D) ───────
step1() {
    echo "=== Step 1: MVDiffusion + NEUS (idx=$IDX, gpu=$GPU) ==="
    $PYTHON test_mvdiffusion_seq.py \
        --config configs/train/testing_4090_stage_1_cad_6views-lvis.yaml \
        --idx $IDX --gpu $GPU \
        2>&1 | tee "$(log_path step1)" || { echo "!! Step 1 실패"; return 1; }
    # NEUS가 mesh를 실제로 만들었는지 검증
    if [[ ! -f "$CONFIG_DIR/False_mm.obj" ]]; then
        echo "!! Step 1 실패: $CONFIG_DIR/False_mm.obj 가 생성되지 않음 (NEUS 미완료)"
        return 1
    fi
    echo "=== Step 1 완료 → $CONFIG_DIR (False_mm.obj 확인됨, log: $(log_path step1)) ==="
}

# ── Step 2: 6-view normals → NEUS mesh → BRep .step ──────────────────────────
step2() {
    echo "=== Step 2: Segmentation + BRep (config=$CONFIG_DIR, review=$REVIEW) ==="
    $PYTHON test_real_images.py \
        --config $CONFIG_DIR \
        --review $REVIEW \
        2>&1 | tee "$(log_path step2)"
    echo "=== Step 2 완료 → neus/temp_mid_outputs/temp_scve${IDX}_0deepcad.step (log: $(log_path step2)) ==="
}

case $STEP in
    0)   step0 ;;
    1)   step1 ;;
    2)   step2 ;;
    all) step0 && step1 && step2 ;;
    12)  step1 && step2 ;;
    *)
        echo "Usage: bash run_pipeline.sh [0|1|2|12|all] [--rgb PATH] [--idx N] [--gpu N] [--review True/False] [--domain object|indoor|outdoor]"
        exit 1 ;;
esac
