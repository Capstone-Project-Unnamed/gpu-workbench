#!/usr/bin/env python3
"""
NURBGen JSON Generation Diagnostic Script
프롬프트 1개로 JSON을 생성하고, 구조를 자세히 분석
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime
import re

print("="*70)
print("NURBGen JSON Generation Diagnostic")
print("="*70)

# ============================================================================
# STEP 1: 모델 로드
# ============================================================================
print("\n📦 STEP 1: 모델 로드")
print("-" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print("Loading Qwen/Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Base model loaded")
    
    print("Loading SadilKhan/NURBGen LoRA...")
    model = PeftModel.from_pretrained(model, "SadilKhan/NURBGen")
    print("✓ LoRA adapter loaded")
    
    model.eval()
    
except Exception as e:
    print(f"❌ 로드 실패: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: 프롬프트 입력
# ============================================================================
print("\n🎯 STEP 2: 프롬프트 입력")
print("-" * 70)

prompt = input("프롬프트 입력 (또는 Enter로 기본값 사용): ").strip()
if not prompt:
    prompt = "Cylindrical bushing with flanges on both ends, featuring a central hollow bore."

print(f"선택된 프롬프트:\n{prompt}\n")

# ============================================================================
# STEP 3: CAD 생성
# ============================================================================
print("🎯 STEP 3: CAD 생성 중...")
print("-" * 70)

import time
start = time.time()

try:
    messages = [{"role": "user", "content": prompt}]
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
            do_sample=False,
            temperature=0.3
        )
    
    elapsed = time.time() - start
    
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"✓ 생성 완료 ({elapsed:.1f}초)")
    print(f"✓ 생성된 토큰: {len(outputs[0]) - inputs.input_ids.shape[1]}")
    
except Exception as e:
    print(f"❌ 생성 실패: {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: 생성된 텍스트 분석
# ============================================================================
print("\n📊 STEP 4: 생성된 텍스트 분석")
print("-" * 70)

print(f"전체 생성 텍스트 길이: {len(generated_text)} 자")
print(f"\n처음 500자:\n{generated_text[:500]}\n")
print(f"마지막 500자:\n{generated_text[-500:]}\n")

# ============================================================================
# STEP 5: JSON 추출 시도
# ============================================================================
print("🔍 STEP 5: JSON 추출")
print("-" * 70)

# JSON 찾기
json_start = generated_text.find('{')
json_end = generated_text.rfind('}')

if json_start == -1 or json_end == -1:
    print("❌ JSON을 찾을 수 없습니다!")
    print(f"첫 번째 {{ 위치: {json_start}")
    print(f"마지막 }} 위치: {json_end}")
    sys.exit(1)

json_str = generated_text[json_start:json_end+1]
print(f"✓ JSON 문자열 추출 ({len(json_str)} 자)")

# JSON 파싱
try:
    cad_json = json.loads(json_str)
    print("✓ JSON 파싱 성공")
except json.JSONDecodeError as e:
    print(f"❌ JSON 파싱 실패: {e}")
    print(f"에러 위치: line {e.lineno}, col {e.colno}")
    print(f"에러 부분:\n{json_str[max(0, e.pos-50):e.pos+50]}")
    sys.exit(1)

# ============================================================================
# STEP 6: JSON 구조 분석
# ============================================================================
print("\n🔬 STEP 6: JSON 구조 분석")
print("-" * 70)

print(f"Top-level keys: {list(cad_json.keys())}")
print(f"Total keys: {len(cad_json.keys())}")

# 각 key의 타입과 구조
for key in cad_json.keys():
    value = cad_json[key]
    if isinstance(value, dict):
        print(f"\n{key} (dict):")
        print(f"  - 내부 keys: {list(value.keys())[:5]}..." if len(value.keys()) > 5 else f"  - 내부 keys: {list(value.keys())}")
        if "poles" in value:
            poles = value["poles"]
            print(f"  - poles 구조: {type(poles).__name__}, 길이={len(poles) if isinstance(poles, (list, dict)) else 'N/A'}")
        if "u_knots" in value:
            print(f"  - u_knots: {value['u_knots'][:3]}..." if len(value['u_knots']) > 3 else f"  - u_knots: {value['u_knots']}")
    elif isinstance(value, list):
        print(f"\n{key} (list):")
        print(f"  - 길이: {len(value)}")
        if len(value) > 0:
            print(f"  - 첫 번째 요소 타입: {type(value[0]).__name__}")
            if isinstance(value[0], dict):
                print(f"  - 첫 번째 요소 keys: {list(value[0].keys())}")

# ============================================================================
# STEP 7: 예상된 구조와 비교
# ============================================================================
print("\n📋 STEP 7: 예상 구조와 비교")
print("-" * 70)

EXPECTED_NURBS_KEYS = {"poles", "u_knots", "v_knots", "u_mults", "v_mults", "u_degree", "v_degree", "weights"}
EXPECTED_TRIMMED_KEYS = {"edges", "is_outer"}

nurbs_count = 0
trimmed_count = 0
other_count = 0

for key, value in cad_json.items():
    if isinstance(value, dict):
        keys_set = set(value.keys())
        if EXPECTED_NURBS_KEYS.intersection(keys_set):
            nurbs_count += 1
            missing = EXPECTED_NURBS_KEYS - keys_set
            if missing:
                print(f"⚠️  {key}: NURBS인데 빠진 필드 = {missing}")
        elif EXPECTED_TRIMMED_KEYS.intersection(keys_set):
            trimmed_count += 1
        else:
            other_count += 1
            print(f"❓ {key}: 인식 못한 구조")
    elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            if "edges" in value[0]:
                trimmed_count += 1
            else:
                other_count += 1
                print(f"❓ {key}: 인식 못한 리스트 구조")

print(f"\n분석 결과:")
print(f"  • NURBS 표면: {nurbs_count}개")
print(f"  • Trimmed face: {trimmed_count}개")
print(f"  • 기타: {other_count}개")

# ============================================================================
# STEP 8: 파일 저장
# ============================================================================
print("\n💾 STEP 8: 결과 저장")
print("-" * 70)

output_dir = Path("./diagnosis_output")
output_dir.mkdir(exist_ok=True)

# JSON 저장
json_path = output_dir / "generated_cad.json"
with open(json_path, 'w') as f:
    json.dump(cad_json, f, indent=2)
print(f"✓ JSON 저장: {json_path}")

# 전체 텍스트 저장
text_path = output_dir / "full_output.txt"
with open(text_path, 'w') as f:
    f.write(generated_text)
print(f"✓ 전체 텍스트 저장: {text_path}")

# 분석 리포트 저장
report_path = output_dir / "analysis_report.txt"
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("NURBGen JSON Generation Analysis Report\n")
    f.write("="*70 + "\n\n")
    f.write(f"프롬프트: {prompt}\n\n")
    f.write(f"생성 시간: {elapsed:.1f}초\n")
    f.write(f"생성 토큰: {len(outputs[0]) - inputs.input_ids.shape[1]}\n")
    f.write(f"JSON 크기: {len(json_str)} 자\n\n")
    f.write(f"구조 분석:\n")
    f.write(f"  - NURBS 표면: {nurbs_count}개\n")
    f.write(f"  - Trimmed face: {trimmed_count}개\n")
    f.write(f"  - 기타: {other_count}개\n\n")
    f.write(f"Top-level keys: {list(cad_json.keys())}\n")

print(f"✓ 분석 리포트 저장: {report_path}")

# ============================================================================
# STEP 9: 다음 단계 제안
# ============================================================================
print("\n" + "="*70)
print("✅ 진단 완료")
print("="*70)

print(f"""
📁 결과 위치: {output_dir}/

다음 단계:
1. generated_cad.json 구조 확인
   - NURBS 표면과 Trimmed face의 구조가 맞는지
   - 필수 필드(poles, knots 등)가 있는지

2. 구조 이상이 있으면:
   - analysis_report.txt의 "기타" 항목 확인
   - full_output.txt에서 생성된 JSON 패턴 분석

3. export.py로 STEP 변환 시도
   python ../NURBGen/src/nurbs_representation/export.py \\
     --input_dir {output_dir} \\
     --output_dir {output_dir}/step_files

4. 결과 확인
   - STEP 파일 생성 여부
   - 에러 메시지 확인
""")

sys.exit(0)