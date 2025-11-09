from pathlib import Path
import json
import shutil
from PIL import Image, ImageOps
from .paths import frame_number_to_path
import os
from google import genai
from google.genai import types

#now we are using the LLM to score the frames.
class Algorithm2_method3:

  def __init__(self, working_dir: Path):
    self.working_dir = working_dir

    #json file path
    self.total_frame_dir = working_dir / "total_frames"
    self.selected_frames_json = working_dir / "selected_frames.json"

    #Gemini API client
    self.gemini_client = genai.Client(api_key='AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo')
    self.reponse_json = self.working_dir / "response.json"
    self.token_json = self.working_dir / "token_usage.json"
  
  @staticmethod
  def prompt_text():
    '''
    create the prompt text for the Gemini API call.
    Extract 15 frames.
    '''

    prompt = """
You are an expert evaluator of frames extracted from portfolio videos to be evaluated for a job application.
Understand the overall story flow and pick 15 non-similar frames based on the following criteria.

Please understand the overall story flow and the purpose of the video first.
Then pick 15 frames based on the following criteria, ensuring the 15 frames are not similar to each other.

Evaluation Criteria:
1. purpose: Purpose fitness (alignment with client goals/brand tone)
2. story: Narrative cues (flow, indication of transition points)
3. edit: Editing quality (rhythm, transitions, timing)
4. cine: Cinematography/composition/exposure/color
5. subject: Key subject/branding visibility (e.g., APEC logo)

Output ONLY the JSON object without any markdown code blocks, explanations, or additional text.
Return ONLY the raw JSON(dictionary format).

- Key: The index of the frame (0-based, where 0 is the first frame, 1 is the second frame, etc.)
- Value: The reason for selection (2 lines explaining why this frame was chosen)

Example (dictonary format json):
{"0": "Reason for selecting first frame...",
  "5": "Reason for selecting sixth frame...",
  ...}
"""
    return prompt





  def gemini_api_call(self, gemini_model = "gemini-2.5-flash"):
    '''
    Method0: input(30 frames) -> output (5 frames wo/ scores)
    '''
    
    # READ Selected image from the json.
    with open(self.selected_frames_json, "r", encoding="utf-8") as f:
      frame_numbers = json.load(f)
    print(f"📸 Loading {len(frame_numbers)} images for evaluation...")
    
    #IMAGE -> BYTE -> PART
    image_parts = []
    for frame_num in frame_numbers:
      frame_path = frame_number_to_path(self.total_frame_dir, frame_num)
      if not frame_path.exists():
        print(f"⚠️  Frame not found: {frame_path}")
        continue
      try:
        # Read image bytes
        with open(frame_path, 'rb') as f:
          image_bytes = f.read()
        
        # Determine MIME type from file extension
        suffix = frame_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg']:
          mime_type = "image/jpeg"
        else:
          raise ValueError(f"Unsupported image format: {suffix}")
        
        # Create Part from bytes
        image_part = types.Part.from_bytes(
          data=image_bytes,
          mime_type=mime_type
        )
        image_parts.append(image_part)
      except Exception as e:
        print(f"⚠️  Failed to load frame {frame_num}: {e}")
        continue
  
    # Create contents array: prompt text first, then all images
    print(f"(ing) Sending {len(image_parts)} images to Gemini API...")
    prompt = self.prompt_text()
    response = self.gemini_client.models.generate_content(
        model=gemini_model,
        contents=[prompt] + image_parts
        # config=types.GenerateContentConfig(
        #     temperature=0.1
        # )
    )

    # Parse response.text to dictionary and save
    try:
      # response.text는 문자열이므로 JSON으로 파싱
      response_dict = json.loads(response.text)
      print(f"✅ JSON 파싱 성공! {len(response_dict)}개 프레임 선택됨")
    except json.JSONDecodeError as e:
      print(f"⚠️  JSON 파싱 실패: {e}")
      print(f"   원본 텍스트를 저장합니다.")
      response_dict = {"raw_response": response.text}
    
    # Save the parsed dictionary as a json file
    with open(self.reponse_json, "w", encoding="utf-8") as f:
      json.dump(response_dict, f, ensure_ascii=False, indent=2)

    # Convert usage_metadata to dictionary for JSON serialization
    usage_dict = {
      "prompt_token_count": response.usage_metadata.prompt_token_count,
      "candidates_token_count": response.usage_metadata.candidates_token_count,
      "total_token_count": response.usage_metadata.total_token_count,
    }
    with open(self.token_json, "w", encoding="utf-8") as f:
      json.dump(usage_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ Response saved. Total tokens: {response.usage_metadata.total_token_count}")
    
    return
  
  def extract_dict_from_response(self):
    '''
    read the response.json and save the selected frames to the working directory.
    '''
    with open(self.reponse_json, "r", encoding="utf-8") as f:
      response_text = json.load(f)
    
    # response_text가 문자열인 경우 dictionary로 변환
    if isinstance(response_text, str):
      try:
        response_dict = json.loads(response_text)
        print(f"✅ 문자열을 dictionary로 변환 성공!")
        print(f"📊 키 개수: {len(response_dict)}")
        print(f"🔑 키 목록: {list(response_dict.keys())}")
        return response_dict
      except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e}")
        return None
    elif isinstance(response_text, dict):
      print(f"✅ 이미 dictionary입니다!")
      print(f"📊 키 개수: {len(response_text)}")
      print(f"🔑 키 목록: {list(response_text.keys())}")
      return response_text
    else:
      print(f"❌ 예상치 못한 타입: {type(response_text)}")
      return None
  
  def save_selected_frames(self, dict_response: dict):
    '''
    start from the dict.
    save the selected real frames from the total_frames folder.
    '''
    # Create selected_frames directory
    # open the self.selected_frames_json file and save the selected frames.
    with open(self.selected_frames_json, "r", encoding="utf-8") as f:
      selected_frame_numbers = json.load(f)

    #create final selected_frames dir
    final_selected_frames_dir = self.working_dir / "final_selected_frames"
    final_selected_frames_dir.mkdir(parents=True, exist_ok=True)
    
    #create a json file to save the reasons (inside final_selected_frames dir)
    reasons_json = final_selected_frames_dir / "reasons.json"
    reasons_dict = {}
    
    for key, reason in dict_response.items():
      # key는 프레임 번호 (문자열), value는 reason
      global_id = selected_frame_numbers[int(key)]
      reasons_dict[global_id] = reason

      frame_path = frame_number_to_path(self.total_frame_dir, global_id)
      if frame_path.exists():
        shutil.copy(frame_path, final_selected_frames_dir / f"{global_id}.jpg")
        print(f"✅ Copied frame {global_id} to {final_selected_frames_dir}")
      else:
        print(f"⚠️  Frame {global_id} not found: {frame_path}")
    
    # Save reasons to JSON file
    with open(reasons_json, "w", encoding="utf-8") as f:
      json.dump(reasons_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(dict_response)} selected frames to {final_selected_frames_dir}")
    print(f"✅ Saved reasons to {reasons_json}")
    return reasons_dict
    
    