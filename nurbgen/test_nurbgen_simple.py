#!/usr/bin/env python3
"""
NURBGen Simple Test Script (Flash Attention 포함)
간단한 프롬프트 → NURBS JSON 생성 → 검증까지

⚠️  Flash Attention 설치 관련 에러가 나면:
    아래 "FLASH_ATTENTION_ENABLED" 부분을 False로 바꾸면 됨
"""

import torch
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# ⚙️ 설정: Flash Attention 사용 여부
# ============================================================================
FLASH_ATTENTION_ENABLED = True  # ← 문제 생기면 False로 바꾸기
# ============================================================================

print("="*70)
print("NURBGen Simple Test")
if FLASH_ATTENTION_ENABLED:
    print("(Flash Attention: ON)")
else:
    print("(Flash Attention: OFF - 속도 저하)")
print("="*70)

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
# STEP 3: 간단한 프롬프트로 생성
# ============================================================================
print("\n🎯 STEP 3: CAD 생성 (간단한 프롬프트)")
print("-" * 70)

test_prompt = "Cylindrical bushing with flanges on both ends, featuring a central hollow bore."

print(f"프롬프트: {test_prompt}\n")
print("생성 중... (이 과정이 좀 오래 걸릴 수 있습니다)")

start_time = time.time()

try:
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,  # Greedy decoding
            temperature=0.3
        )
    
    elapsed = time.time() - start_time
    
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"✓ 생성 완료 ({elapsed:.1f}초)")
    print(f"✓ 생성된 토큰: {len(outputs[0]) - inputs.input_ids.shape[1]}")
    
except Exception as e:
    print(f"❌ 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: JSON 파싱
# ============================================================================
print("\n📄 STEP 4: JSON 파싱 및 검증")
print("-" * 70)

try:
    # JSON 부분 추출
    json_start = generated_text.find('{')
    if json_start == -1:
        print("❌ 생성된 텍스트에서 JSON을 찾을 수 없습니다")
        print("\n생성된 텍스트 (처음 500자):")
        print(generated_text[:500])
        sys.exit(1)
    
    json_str = generated_text[json_start:]
    
    # JSON 파싱 시도
    cad_json = json.loads(json_str)
    print("✓ JSON 파싱 성공")
    
    # 구조 검증
    if "faces" not in cad_json:
        print("⚠️  'faces' 필드 없음")
    else:
        num_faces = len(cad_json["faces"])
        print(f"✓ Face 개수: {num_faces}")
        
        # 각 face 타입 확인
        face_types = {}
        for face in cad_json["faces"]:
            face_type = face.get("type", "unknown")
            face_types[face_type] = face_types.get(face_type, 0) + 1
        
        print(f"✓ Face 타입 분포:")
        for ftype, count in face_types.items():
            print(f"  - {ftype}: {count}")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON 파싱 실패: {e}")
    print(f"에러 위치: line {e.lineno}, col {e.colno}")
    print("\n생성된 텍스트 (마지막 1000자):")
    print(generated_text[-1000:])
    sys.exit(1)

except Exception as e:
    print(f"❌ 검증 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: 결과 저장
# ============================================================================
print("\n💾 STEP 5: 결과 저장")
print("-" * 70)

# 출력 디렉토리 생성
output_dir = Path("./nurbgen_test_output")
output_dir.mkdir(exist_ok=True)

# JSON 저장
json_path = output_dir / "generated_cad.json"
with open(json_path, 'w') as f:
    json.dump(cad_json, f, indent=2)
print(f"✓ JSON 저장: {json_path}")

# 프롬프트 + 결과 메타데이터 저장
metadata = {
    "timestamp": datetime.now().isoformat(),
    "prompt": test_prompt,
    "generation_time_seconds": elapsed,
    "generated_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
    "num_faces": len(cad_json.get("faces", [])),
    "face_types": face_types,
    "success": True,
    "flash_attention_enabled": FLASH_ATTENTION_ENABLED
}

metadata_path = output_dir / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ 메타데이터 저장: {metadata_path}")

# 전체 생성 텍스트 저장 (디버깅용)
text_path = output_dir / "generated_full_output.txt"
with open(text_path, 'w') as f:
    f.write(generated_text)
print(f"✓ 전체 출력 저장: {text_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)

print(f"""
📊 결과 요약:
  • 프롬프트: {test_prompt}
  • 생성 시간: {elapsed:.1f}초
  • 생성 토큰: {len(outputs[0]) - inputs.input_ids.shape[1]}
  • Face 개수: {len(cad_json.get('faces', []))}
  • Face 타입: {face_types}
  • Flash Attention: {"✓ ON" if FLASH_ATTENTION_ENABLED else "✗ OFF"}

📁 저장 위치:
  • JSON: {json_path}
  • 메타데이터: {metadata_path}
  • 전체 출력: {text_path}

🔍 다음 단계:
  1. {json_path} 파일 확인
  2. JSON 구조가 올바른지 검증
  3. STEP 변환 시도:
     python src/nurbs_representation/export.py --input_dir {output_dir} --output_dir {output_dir}/step_files

⚡ Flash Attention 문제 시:
  • 위 스크립트 상단의 FLASH_ATTENTION_ENABLED = False로 변경
  • 속도는 느려지지만 작동함
""")

print("\n✨ NURBGen이 정상 작동합니다!")