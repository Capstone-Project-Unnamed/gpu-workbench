# BrepGen 로컬 데스크탑 실행 가이드

## 📋 사전 준비

### 1. Conda 환경 설정
```bash
# CUDA 12.1 기준 PyTorch 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 필수 라이브러리
conda install -c conda-forge pythonocc-core -y

# Python 패키지
pip install \
    huggingface_hub==0.23.0 \
    diffusers==0.27.0 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    pyyaml trimesh scikit-learn scipy

pip install git+https://github.com/AutodeskAILab/occwl -q
```

### 2. BrepGen 클론
```bash
git clone https://github.com/samxuxiang/BrepGen.git
cd BrepGen
```

### 3. 모델 가중치 다운로드
Google Drive에서 다운로드 후 `BrepGen/` 디렉토리에 배치:
- **VAE weights**: https://drive.google.com/drive/folders/18Ib9L0kpFf4ylZIRTCYFhXZB_GVIUm53
- **LDM weights**: https://drive.google.com/drive/folders/1hv7ZUcU-L3J0LiONK60-TEh7sAN0zfve

필요 파일:
```
BrepGen/
├─ abc_vae_surf.pt
├─ abc_vae_edge.pt
├─ abc_ldm_surfpos.pt
├─ abc_ldm_surfz.pt
├─ abc_ldm_edgepos.pt
└─ abc_ldm_edgez.pt
```

---

## 🚀 실행 순서

### Step 1: 샘플 생성
```bash
# 배치 1회 반복 (4개 샘플 생성)
python run_generation.py --batches 1

# 배치 5회 반복 (20개 샘플 생성)
python run_generation.py --batches 5
```

**결과 위치**: `BrepGen/samples_abc/`
- `*.step` 파일들
- `*.stl` 파일들

### Step 2: STEP → PKL 변환 (선택사항)
```bash
python preprocess_to_pkl.py \
    --input BrepGen/samples_abc \
    --output BrepGen/samples_abc_pkl
```

**결과 위치**: `BrepGen/samples_abc_pkl/`
- `*.pkl` 파일들 (BrepGen 형식)

---

## 📊 설정 (run_generation.py)

`run_generation.py` 내부의 `EVAL_CONFIG`에서 수정:

```python
'batch_size': 4,          # 한 번에 생성할 샘플 수
'num_surfaces': 50,       # 최대 면(face) 개수
'num_edges': 40,          # 최대 모서리 개수
'z_threshold': 0.2,       # 면 병합 임계값
'bbox_threshold': 0.08,   # 중복 제거 임계값
```

---

## 📁 폴더 구조

```
프로젝트/
├─ BrepGen/                      # 클론한 레포
│  ├─ sample.py
│  ├─ abc_vae_surf.pt           # 다운로드 필수
│  ├─ abc_vae_edge.pt
│  ├─ abc_ldm_surfpos.pt
│  ├─ abc_ldm_surfz.pt
│  ├─ abc_ldm_edgepos.pt
│  ├─ abc_ldm_edgez.pt
│  ├─ samples_abc/              # 생성 결과
│  │  ├─ xxx_0.step
│  │  ├─ xxx_0.stl
│  │  └─ ...
│  └─ samples_abc_pkl/          # pkl 변환 결과
│     ├─ xxx_0.pkl
│     └─ ...
├─ run_generation.py            # Step 1 실행 파일
├─ preprocess_to_pkl.py         # Step 2 실행 파일
└─ setup_brepgen.sh             # 한 번에 설정 (bash)
```

---

## 🐛 트러블슈팅

### "ImportError: cannot import name 'sample' from BrepGen"
```bash
# BrepGen 디렉토리에서 실행하면 해결
cd BrepGen
python ../run_generation.py
```

### GPU 메모리 부족
`EVAL_CONFIG`에서 `batch_size` 줄이기:
```python
'batch_size': 2,  # 4 → 2로 변경
```

### STEP 파일 생성 실패
```bash
# OCC 설치 확인
python -c "from OCC.Extend.DataExchange import write_step_file; print('OK')"
```

---

## 📌 다음 단계: PKL 페어 생성 (Phase 1)

`preprocess_to_pkl.py`를 확장하여:

1. **GT pkl 로드** (ABC 원본 데이터)
2. **생성 pkl과 매칭**
   ```python
   pairs = [
       {'generated': gen_pkl_1, 'gt': gt_pkl_1},
       {'generated': gen_pkl_2, 'gt': gt_pkl_2},
       ...
   ]
   ```
3. **pkl 페어 저장**
   ```python
   with open('refinement_pairs.pkl', 'wb') as f:
       pickle.dump(pairs, f)
   ```

---

## ⚙️ 참고: Colab vs Local 차이

| 항목 | Colab | Local |
|-----|-------|-------|
| 환경 설정 | `condacolab` | 수동 conda |
| 가중치 다운로드 | `gdown` | 수동 다운로드 |
| 드라이브 연동 | `mount()` | - |
| 실행 방식 | Jupyter cell | Python 파일 |

이 스크립트들은 Local 환경에 최적화되어 있습니다.