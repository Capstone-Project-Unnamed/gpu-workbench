#!/usr/bin/env python3
"""
NURBGen Interactive Test Script
사용자가 프롬프트를 입력하고 NURBS JSON 생성, degree 확인, STEP/STL 변환까지

Flash Attention 에러 발생 시 아래의 FLASH_ATTENTION_ENABLED를 False로 변경

📌 do_sample=False(greedy)는 특정 프롬프트에서 결정론적으로 degenerate한
   결과(예: 작은 상자)로 수렴하는 것이 확인되어, 논문/공식 infer_nurbgen.py와
   동일하게 do_sample=True + temperature=0.3 샘플링을 사용합니다.
   샘플링은 확률적이라 매번 품질이 다를 수 있어 기본 N_SAMPLES회 생성합니다.
"""

import torch
import json
import re
import sys
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================================
# ⚙️ 설정
# ============================================================================
FLASH_ATTENTION_ENABLED = False  # ← 문제 생기면 False로 바꾸기
PROMPT_PREFIX = "Generate NURBS for the following: "  # 공식 학습/추론 포맷
TEMPERATURE = 0.3
N_SAMPLES = 3  # 샘플링은 확률적이므로 여러 번 생성해서 비교
NURBGEN_SRC_DIR = Path("./NURBGen/src")
# ============================================================================

print("="*70)
print("NURBGen Interactive Test")
print("="*70)


def prompt_to_slug(prompt: str, max_len: int = 40) -> str:
    """프롬프트를 파일명으로 쓸 수 있는 slug로 변환."""
    slug = prompt.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:max_len].rstrip("_") or "prompt"


def export_json_to_step(json_path: Path, output_dir: Path):
    """generated_cad.json 대신 slug 기반 파일명으로 STEP/STL 변환.
    export.py의 glob(**/*.json)이 metadata.json까지 잡는 버그를 피하기 위해
    변환 대상 JSON만 담은 격리 폴더에서 실행한다.
    """
    isolated_dir = output_dir / "_export_input"
    isolated_dir.mkdir(exist_ok=True, parents=True)
    isolated_json = isolated_dir / json_path.name
    shutil.copy(json_path, isolated_json)

    step_dir = output_dir / "step_files"
    cmd = [
        sys.executable,
        str(NURBGEN_SRC_DIR / "nurbs_representation" / "export.py"),
        "--input_dir", str(isolated_dir),
        "--output_dir", str(step_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    shutil.rmtree(isolated_dir, ignore_errors=True)

    step_path = step_dir / json_path.with_suffix(".step").name
    stl_path = step_dir / json_path.with_suffix(".stl").name
    if step_path.exists():
        print(f"  ✓ STEP 변환 완료: {step_path} ({step_path.stat().st_size / 1024:.1f}KB)")
        if stl_path.exists():
            print(f"  ✓ STL 변환 완료: {stl_path} ({stl_path.stat().st_size / 1024:.1f}KB)")
        return True
    else:
        print(f"  ❌ STEP 변환 실패")
        if result.stderr:
            print(f"     stderr: {result.stderr[-500:]}")
        return False


# ============================================================================
# STEP 0: 프롬프트 입력
# ============================================================================
print("\n🎯 STEP 0: 프롬프트 입력")
print("-" * 70)

print("\n기본 프롬프트들:")
print("  1. Rectangular plate")
print("  2. Cylindrical bushing")
print("  3. Triangular ring")
print("  4. Cube")
print("  5. Socket head cap screw")
print("  6. 직접 입력")

choice = input("\n선택 (1-6): ").strip()

if choice == "1":
    test_prompt = "Rectangular plate"
elif choice == "2":
    test_prompt = "Cylindrical bushing"
elif choice == "3":
    test_prompt = "Triangular ring"
elif choice == "4":
    test_prompt = "Cube"
elif choice == "5":
    test_prompt = "Socket head cap screw"
elif choice == "6":
    test_prompt = input("\n프롬프트 입력: ").strip()
    if not test_prompt:
        test_prompt = "Rectangular plate"
        print(f"기본값 사용: {test_prompt}")
else:
    test_prompt = "Rectangular plate"
    print(f"기본값 사용: {test_prompt}")

print(f"\n선택된 프롬프트: {test_prompt}")

# ============================================================================
# STEP 1: 환경 확인
# ============================================================================
print("\n📋 STEP 1: 환경 확인")
print("-" * 70)

try:
    print(f"✓ GPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  GPU 없음 - CPU 모드 실행")
except Exception as e:
    print(f"❌ GPU 확인 실패: {e}")

# ============================================================================
# STEP 2: 모델 로드
# ============================================================================
print("\n📦 STEP 2: 모델 로드 중...")
print("-" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("Loading base model: Qwen/Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print("✓ Tokenizer loaded")

    # Flash Attention 설정
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }

    if FLASH_ATTENTION_ENABLED:
        try:
            load_kwargs["attn_impl"] = "flash_attn_2"
            print("(Flash Attention 활성화 시도 중...)")
        except:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        **load_kwargs
    )
    print("✓ Base model loaded")

    print("Loading LoRA adapter: SadilKhan/NURBGen...")
    model = PeftModel.from_pretrained(model, "SadilKhan/NURBGen")
    print("✓ LoRA adapter loaded")

    model.eval()
    print("✓ Model in eval mode")

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("\n💡 문제 해결:")
    print("1. Huggingface 로그인: huggingface-cli login")
    print("2. 필요한 패키지: pip install transformers peft")
    print("3. Flash Attention 문제면 위 스크립트에서 FLASH_ATTENTION_ENABLED = False로 변경")
    sys.exit(1)

# ============================================================================
# STEP 3: CAD 생성 (샘플링, N_SAMPLES회 반복)
# ============================================================================
print(f"\n🎯 STEP 3: CAD 생성 (do_sample=True, temperature={TEMPERATURE}, {N_SAMPLES}회 반복)")
print("-" * 70)

full_prompt = PROMPT_PREFIX + test_prompt
slug = prompt_to_slug(test_prompt)

print(f"프롬프트: {full_prompt}\n")

messages = [{"role": "user", "content": full_prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

output_dir = Path(f"./nurbgen_test_output_{slug}")
output_dir.mkdir(exist_ok=True)

trial_summaries = []

for trial in range(1, N_SAMPLES + 1):
    print(f"\n--- Trial {trial}/{N_SAMPLES} ---")
    torch.manual_seed(1000 + trial)
    start_time = time.time()

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=1.0,
            )
        elapsed = time.time() - start_time

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        n_tokens = len(outputs[0]) - inputs.input_ids.shape[1]
        print(f"✓ 생성 완료 ({elapsed:.1f}초, {n_tokens} tokens)")

    except Exception as e:
        print(f"❌ 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        continue

    # JSON 파싱
    json_start = generated_text.find('{')
    if json_start == -1:
        print("❌ 생성된 텍스트에서 JSON을 찾을 수 없습니다")
        continue

    try:
        cad_json = json.loads(generated_text[json_start:])
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        continue

    # Face/degree 분석
    num_faces = len(cad_json)
    degree_dist = {}
    nurbs_count = 0
    trimmed_count = 0
    for key, face in cad_json.items():
        if isinstance(face, dict) and "poles" in face:
            deg_key = f"u{face.get('u_degree', '?')}v{face.get('v_degree', '?')}"
            degree_dist[deg_key] = degree_dist.get(deg_key, 0) + 1
            nurbs_count += 1
        elif isinstance(face, (list, dict)):
            trimmed_count += 1

    print(f"  Face 개수: {num_faces} (NURBS: {nurbs_count}, Trimmed: {trimmed_count})")
    print(f"  Degree 분포: {degree_dist}")

    # 저장 (프롬프트 기반 파일명)
    base_name = f"generated_{slug}_trial{trial}" if N_SAMPLES > 1 else f"generated_{slug}"
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json.dump(cad_json, f, indent=2)
    print(f"  ✓ JSON 저장: {json_path}")

    text_path = output_dir / f"{base_name}_full_output.txt"
    with open(text_path, 'w') as f:
        f.write(generated_text)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": full_prompt,
        "trial": trial,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "generation_time_seconds": elapsed,
        "generated_tokens": n_tokens,
        "num_faces": num_faces,
        "nurbs_count": nurbs_count,
        "trimmed_count": trimmed_count,
        "degree_distribution": degree_dist,
        "success": True,
        "flash_attention_enabled": FLASH_ATTENTION_ENABLED,
    }
    metadata_path = output_dir / f"{base_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # STEP/STL 자동 변환
    print(f"  STEP/STL 변환 중...")
    export_json_to_step(json_path, output_dir)

    trial_summaries.append({
        "trial": trial,
        "base_name": base_name,
        "num_faces": num_faces,
        "degree_distribution": degree_dist,
    })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)

print(f"\n📊 결과 요약 ({len(trial_summaries)}/{N_SAMPLES} 성공):")
for s in trial_summaries:
    print(f"  Trial {s['trial']}: {s['base_name']} — face {s['num_faces']}개, degree {s['degree_distribution']}")

print(f"""
📁 저장 위치: {output_dir}/

💡 degree에 u2/v2 같은 곡면(curved)이나 trimmed face가 섞여 있는 trial일수록
   상자 형태가 아닌 곡면/원형 구조일 가능성이 높습니다.
   전부 u1v1(평면)만 있다면 여전히 degenerate한 결과일 수 있습니다.

💡 여러 프롬프트 비교:
   다시 이 스크립트를 실행하고 다른 프롬프트를 선택하세요!
""")
