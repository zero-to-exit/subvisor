"""사용 가능한 Gemini 모델 확인"""
import os
import google.generativeai as genai

api_key = 'AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo'
genai.configure(api_key=api_key)

print("사용 가능한 모델 목록:")
print("=" * 80)
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"모델: {model.name}")
        print(f"  지원 메서드: {model.supported_generation_methods}")
        print()







