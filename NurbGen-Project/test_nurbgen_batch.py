#!/usr/bin/env python3
"""
NURBGen Batch Processing Script
NurbGen-Project 폴더에서 실행
prompts.txt에서 프롬프트 읽어서 여러 개 생성 및 STEP 변환
"""

import torch
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess

# ============================================================================
# ⚙️ 설정
# ============================================================================
FLASH_ATTENTION_ENABLED = False
PROMPTS_FILE = "prompts.txt"
OUTPUT_BASE_DIR = Path("./nurbgen_examples")
NURBGEN_SRC_DIR = Path("./NURBGen/src")

print("="*70)
print("NURBGen Batch Processing")
print("="*70)

# ============================================================================
# STEP 0: 프롬프트 로드
# ============================================================================
print("\n📋 STEP 0: 프롬프트 로드")
print("-" * 70)

try:
    with open(PROMPTS_FILE, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 프롬프트 로드 성공: {len(prompts)}개")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt[:60]}...")
except FileNotFoundError:
    print(f"❌ 파일 못 찾음: {PROMPTS_FILE}")
    sys.exit(1)

# ============================================================================
# STEP 1: 환경 확인
# ============================================================================
print("\n📋 STEP 1: 환경 확인")
print("-" * 70)

print(f"✓ GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU 개수: {torch.cuda.device_count()}")

# ============================================================================
# STEP 2: 모델 로드 (한 번만)
# ============================================================================
print("\n📦 STEP 2: 모델 로드 중... (한 번만 수행)")
print("-" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print("Loading base model: Qwen/Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print("✓ Tokenizer loaded")
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
    
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
    sys.exit(1)

# ============================================================================
# STEP 3: 배치 처리
# ============================================================================
print("\n🎯 STEP 3: 배치 CAD 생성")
print("-" * 70)

OUTPUT_BASE_DIR.mkdir(exist_ok=True)

results = {
    "timestamp": datetime.now().isoformat(),
    "total_prompts": len(prompts),
    "successful": 0,
    "failed": 0,
    "details": []
}

for idx, prompt in enumerate(prompts, 1):
    print(f"\n[{idx}/{len(prompts)}] {prompt[:60]}...")
    
    # 출력 디렉토리
    output_dir = OUTPUT_BASE_DIR / f"nurbgen_example_{idx}"
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # CAD 생성
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
        
        elapsed = time.time() - start_time
        
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # JSON 파싱
        json_start = generated_text.find('{')
        if json_start == -1:
            raise ValueError("JSON not found in generated text")
        
        json_str = generated_text[json_start:]
        cad_json = json.loads(json_str)
        
        # JSON 저장
        json_path = output_dir / "generated_cad.json"
        with open(json_path, 'w') as f:
            json.dump(cad_json, f, indent=2)
        
        # 메타데이터 저장
        metadata = {
            "index": idx,
            "prompt": prompt,
            "generation_time_seconds": elapsed,
            "generated_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
            "success": True
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 결과 기록
        print(f"  ✓ 생성 완료 ({elapsed:.1f}초)")
        results["successful"] += 1
        results["details"].append({
            "index": idx,
            "prompt": prompt[:100],
            "time": elapsed,
            "status": "success"
        })
        
    except Exception as e:
        print(f"  ❌ 생성 실패: {str(e)[:50]}")
        results["failed"] += 1
        results["details"].append({
            "index": idx,
            "prompt": prompt[:100],
            "error": str(e)[:100],
            "status": "failed"
        })
        continue

# ============================================================================
# STEP 4: STEP 변환
# ============================================================================
print("\n💾 STEP 4: STEP 변환")
print("-" * 70)

for idx in range(1, len(prompts) + 1):
    output_dir = OUTPUT_BASE_DIR / f"nurbgen_example_{idx}"
    json_path = output_dir / "generated_cad.json"
    
    if not json_path.exists():
        print(f"[{idx}] ⊘ JSON 없음 (생성 실패)")
        continue
    
    try:
        print(f"[{idx}] STEP 변환 중...", end=" ")
        
        # STEP 변환 명령어 (NurbGen 폴더 기준)
        cmd = [
            "python",
            str(NURBGEN_SRC_DIR / "nurbs_representation" / "export.py"),
            "--input_dir", str(output_dir),
            "--output_dir", str(output_dir / "step_files")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60
        )
        
        step_file = output_dir / "step_files" / "generated_cad.step"
        if step_file.exists():
            print(f"✓ ({step_file.stat().st_size / 1024:.1f}KB)")
        else:
            print("⊘ STEP 파일 생성 실패")
    
    except subprocess.TimeoutExpired:
        print("⊘ 타임아웃")
    except Exception as e:
        print(f"⊘ {str(e)[:30]}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ BATCH PROCESSING COMPLETE")
print("="*70)

print(f"""
📊 결과 요약:
  • 총 프롬프트: {results['total_prompts']}개
  • 성공: {results['successful']}개 ✅
  • 실패: {results['failed']}개 ❌
  • 성공률: {results['successful']/results['total_prompts']*100:.1f}%

📁 저장 위치:
  • {OUTPUT_BASE_DIR}/nurbgen_example_1/
  • {OUTPUT_BASE_DIR}/nurbgen_example_2/
  • ...
  • {OUTPUT_BASE_DIR}/nurbgen_example_{len(prompts)}/

각 폴더에 포함:
  • generated_cad.json (원본 CAD)
  • metadata.json (통계)
  • step_files/generated_cad.step (변환된 STEP)
  • step_files/generated_cad.stl (변환된 STL)
""")

# 결과 저장
summary_path = OUTPUT_BASE_DIR / "batch_results.json"
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📋 상세 결과 저장: {summary_path}\n")