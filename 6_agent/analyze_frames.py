import os
import base64
from pathlib import Path
from openai import OpenAI

def encode_image(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(client, image_path):
    """단일 이미지를 분석하여 어색한 점이나 수정 필요 여부를 판단"""
    base64_image = encode_image(image_path)
    
    prompt = """이 이미지를 분석해주세요. 다음 관점에서 평가해주세요:

1. **구도/프레이밍**: 인물이나 주요 피사체의 위치가 적절한가? 화면 구성이 자연스러운가?
2. **장면의 자연스러움**: 장면이 어색하거나 부자연스러운 점이 있는가?
3. **시각적 품질**: 선명도, 조명, 대비가 적절한가?
4. **컨텍스트**: 배경과 주제의 관계가 자연스러운가?

특히 agent1_frame_08_t150.58s_score50.843.jpg와 같이 "인물이 중심에 있지만 장면이 어색한 사진"과 유사한 문제가 있는지 확인해주세요.

답변 형식:
- 문제점: (발견된 문제점 설명)
- 수정 필요: (예/아니오)
- 이유: (구체적인 이유)
"""
    
    try:
        # Vision API는 gpt-4o 또는 gpt-4o-2024-08-06 필요
        # gpt-4o-mini는 Vision을 지원하지 않습니다
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")  # 기본값: gpt-4o (Vision 지원)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"오류 발생: {str(e)}"

def main():
    # OpenAI 클라이언트 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("다음 명령으로 설정하세요: export OPENAI_API_KEY='your-api-key'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # 이미지 폴더 경로
    frames_dir = Path("/Users/jeff/python/6_agent/frames_agent1")
    
    # 모든 이미지 파일 찾기
    image_files = sorted(frames_dir.glob("*.jpg"))
    
    # 비용 절감을 위한 샘플링 옵션
    sample_only = os.getenv("SAMPLE_ONLY", "false").lower() == "true"
    sample_count = int(os.getenv("SAMPLE_COUNT", "5"))  # 기본 5개만 샘플링
    
    if sample_only:
        import random
        image_files = random.sample(list(image_files), min(sample_count, len(image_files)))
        print(f"💡 샘플링 모드: {len(image_files)}개의 이미지만 분석합니다.")
    
    print(f"총 {len(image_files)}개의 이미지를 분석합니다.\n")
    print("=" * 80)
    
    results = []
    
    for image_file in image_files:
        print(f"\n분석 중: {image_file.name}")
        print("-" * 80)
        
        analysis = analyze_image(client, image_file)
        print(analysis)
        
        # 수정이 필요한 이미지인지 확인
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
    
    print("\n상세 분석 결과:")
    for result in results:
        if result["needs_fix"]:
            print(f"\n[{result['file']}]")
            print(result["analysis"])

if __name__ == "__main__":
    main()

