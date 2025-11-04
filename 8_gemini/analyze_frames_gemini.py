"""
Google Gemini API를 사용한 이미지 분석 스크립트 (무료 티어 사용 가능)
"""
import os
import base64
from pathlib import Path
import google.generativeai as genai
from PIL import Image


def encode_image(image_path):
    """이미지를 base64로 인코딩 (참고용)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():

    api_key = 'AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo'
    genai.configure(api_key=api_key)
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"Usage model: {model_name}\n")

    model = genai.GenerativeModel(model_name)

    # 10개의 이미지 로드
    frames_dir = Path("/Users/jeff/python/8_gemini/frames")
    image_files = sorted(frames_dir.glob("agent3_frame_*.jpg"))[:10]  # 처음 10개만 선택
    
    print(f"처리할 이미지 개수: {len(image_files)}")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    print()
    
    # 이미지들을 PIL Image 객체로 로드
    images = []
    for img_path in image_files:
        images.append(Image.open(img_path))

    prompt = """You are the best image evaluator in the world.
 
1. Check if there is a person in the center of the photo, and if so, whether the face is clearly visible and the photo is not blurry.
2. If there is no person, check if a product is in the center of the photo, and whether the product is clearly visible and the photo is not blurry.
3. Also, state whether the photo appears to have been taken by a drone or not.

Please analyze each of the 10 images for these three points, one by one.


"""
    
    # 여러 이미지를 한번에 전송
    content = [prompt] + images
    print("이미지 분석 중...")
    response = model.generate_content(content)
    
    # 결과 저장
    output_file = "image_eva_10frames.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== {len(image_files)}개 이미지 분석 결과 ===\n\n")
        f.write(f"모델: {model_name}\n")
        f.write(f"처리된 이미지:\n")
        for i, img_file in enumerate(image_files, 1):
            f.write(f"  {i}. {img_file.name}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(response.text)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")
    print(f"\n응답 길이: {len(response.text)} 문자")



if __name__ == "__main__":
    main()

