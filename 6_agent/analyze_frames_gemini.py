"""
Google Gemini API를 사용한 이미지 분석 스크립트 (무료 티어 사용 가능)
"""
import os
import base64
from pathlib import Path

try:
    import google.generativeai as genai

except ImportError:
    print("⚠️  google-generativeai 패키지가 설치되지 않았습니다.")
    print("설치: pip install google-generativeai")
    exit(1)

def analyze_image_with_gemini(image_path, prompt, api_key, model_name=None):
    """Gemini를 사용하여 이미지 분석"""
    base64_image = encode_image(image_path)
    
    # 이미지를 PIL Image로 로드
    from PIL import Image
    image = Image.open(image_path)
    
    # 사용 가능한 Vision 모델: gemini-2.5-flash, gemini-2.0-flash, gemini-flash-latest
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    
    try:
        response = model.generate_content([prompt, image])
        
        return response.text
    except Exception as e:
        return f"오류 발생: {str(e)}"

def encode_image(image_path):
    """이미지를 base64로 인코딩 (참고용)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    # Gemini API 키 설정
    #api_key = os.getenv("GEMINI_API_KEY")
    # if not api_key:
    #     print("⚠️  GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    #     print("\n📌 API 키를 입력해주세요 (발급: https://aistudio.google.com/)")
    #     api_key = input("Gemini API Key: ").strip()
    #     if not api_key:
    #         print("❌ API 키가 입력되지 않았습니다.")
    #         return
    #     print("✅ API 키가 입력되었습니다.\n")
    api_key = 'AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo'
    
    genai.configure(api_key=api_key)
    
    # 모델 이름 설정
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"🤖 사용 모델: {model_name}\n")
    
    # 이미지 폴더 경로 (환경변수 또는 기본값)
    folder_name = os.getenv("FRAMES_FOLDER", "frames_agent1")  # 기본값: frames_agent1
    base_dir = Path("/Users/jeff/python/6_agent")
    frames_dir = base_dir / folder_name
    image_files = sorted(frames_dir.glob("*.jpg"))
    
    # 샘플링 옵션
    sample_only = os.getenv("SAMPLE_ONLY", "false").lower() == "true"
    sample_count = int(os.getenv("SAMPLE_COUNT", "5"))
    
    if sample_only:
        import random
        image_files = random.sample(list[Path](image_files), min(sample_count, len(image_files)))
        print(f"💡 샘플링 모드: {len(image_files)}개의 이미지만 분석합니다.")
    
    prompt = """이 이미지를 분석해주세요. 다음 관점에서 평가해주세요:

1. **구도/프레이밍**: 인물이나 주요 피사체의 위치가 적절한가? 화면 구성이 자연스러운가?
2. **장면의 자연스러움**: 장면이 어색하거나 부자연스러운 점이 있는가?
3. **시각적 품질**: 선명도, 조명, 대비가 적절한가?
4. **컨텍스트**: 배경과 주제의 관계가 자연스러운가?

특히 "인물이 중심에 있지만 장면이 어색한 사진"과 유사한 문제가 있는지 확인해주세요.

답변 형식:
- 문제점: (발견된 문제점 설명)
- 수정 필요: (예/아니오)
- 이유: (구체적인 이유)
"""
    
    print(f"✅ Gemini API 설정 완료 (무료 티어)")
    print(f"📁 분석 대상 폴더: {frames_dir}")
    print(f"📊 총 {len(image_files)}개의 이미지를 분석합니다.\n")
    print("=" * 80)
    
    results = []
    
    for image_file in image_files:
        print(f"\n분석 중: {image_file.name}")
        print("-" * 80)
        
        analysis = analyze_image_with_gemini(image_file, prompt, api_key, model_name)
        print(analysis)
        
        needs_fix = "수정 필요: 예" in analysis or "수정 필요:네" in analysis or "수정 필요: yes" in analysis.lower()
        
        results.append({
            "file": image_file.name,
            "needs_fix": needs_fix,
            "analysis": analysis
        })
        
        print("=" * 80)
    
    # 결과 요약
    print("\n\n" + "=" * 80)
    print("분석 결과 요약")
    print("=" * 80)
    
    needs_fix_images = [r for r in results if r["needs_fix"]]
    
    if needs_fix_images:
        print(f"\n수정이 필요한 이미지: {len(needs_fix_images)}개")
        for result in needs_fix_images:
            print(f"  - {result['file']}")
    else:
        print("\n모든 이미지가 양호한 것으로 판단됩니다.")

if __name__ == "__main__":
    main()

