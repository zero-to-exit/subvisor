from pathlib import Path
import json
from PIL import Image, ImageOps
from .paths import frame_number_to_path
import os
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("⚠️  google-generativeai package is not installed.")
    print("install: pip install google-generativeai")

#now we are using the LLM to score the frames.
class Algorithm2:

  def __init__(self, working_dir: Path):
    self.working_dir = working_dir

    #json file path
    self.total_frame_dir = working_dir / "total_frames"
    self.selected_frames_json = working_dir / "selected_frames.json"

    #create pdf
    self.pdf_path = self.working_dir / "selected_frames.pdf"
    self.prompt1_text = ""

    #Gemini API client
    self.gemini_client = genai.Client(api_key='AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo')
    self.reponse_json = self.working_dir / "response.json"
    self.token_json = self.working_dir / "token_usage.json"
  
  def create_pdf_file(self, MAX_SIDE = 800):
    '''
    create a pdf file from the selected frames in selected_frames.json
    reads frame numbers from json, loads images from total_frames, and creates PDF
    saves PDF to working_dir
    '''
    # Read selected frame numbers from JSON
    with open(self.selected_frames_json, "r", encoding="utf-8") as f:
      frame_numbers = json.load(f)
    
    if not frame_numbers:
      print("⚠️  No frames found in selected_frames.json")
      return None
    
    # Load images
    images = []
    for frame_num in frame_numbers:
      frame_path = frame_number_to_path(self.total_frame_dir, frame_num)
      if not frame_path.exists():
        print(f"⚠️  Frame not found: {frame_path}")
        continue
      
      try:
        img = Image.open(frame_path)
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode != 'RGB':
          img = img.convert('RGB')
        #print(f"Current image quality: {img.size} pixels, {img.mode} mode")
        img = ImageOps.exif_transpose(img)

        #resize the image if it is too large.
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        images.append(img)
      except Exception as e:
        print(f"⚠️  Failed to load frame {frame_num}: {e}")
        continue
    
    if not images:
      print("⚠️  No valid images to create PDF")
      return None
    
    # Create PDF file path
    pdf_path = self.pdf_path
    
    # Save as PDF (first image saves, others append)
    if images:
      images[0].save(
        pdf_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:] if len(images) > 1 else []
      )
      print(f"✅ Created PDF: {pdf_path} ({len(images)} frames)")
      return pdf_path
    
    return None

  def create_frame_evaluation_prompt(self):
    '''
    Create prompt for evaluating multiple frames at once.
    This prompt will be used with 30 images in a single API call.
    '''
    self.prompt1_text = """
You are an expert evaluator of frames extracted from portfolio videos to be evaluated for a job application.
understand the overall story flow and evaluate each frame (0~1 each) with following criteria.
1. rel: 목적 적합성(의뢰 목적/브랜드 톤과의 부합)
2. story: 내러티브 신호(흐름, 전환 포인트 시사)
3. edit: 편집 완성도(리듬/전환/타이밍)
4. cine: 촬영/구도/노출/색감
5. subject: 핵심 피사체/브랜딩 가시성(APEC 로고 등)

output the scored frames in JSON format(key: file_path, value: scored frames).
{
  "file_path1":{
    "score1": 0.85,
    "score2": 0.75,
    "score3": 0.65,
    "score4": 0.55,
    "score5": 0.45
  },
  "file_path2":{

"""
    return


  @staticmethod
  def count_tokens(prompt, gemini_model = "gemini-2.5-flash-lite"):
    '''
    method to count the tokens used for the prompt. for free.
    '''
    client = genai.Client(api_key='AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo')
    # Count tokens using the new client method.
    total_tokens = client.models.count_tokens(
        model=gemini_model, contents=prompt
    )
    print("total_tokens: ", total_tokens)

    return total_tokens

  def gemini_api_call_method0(self, gemini_model = "gemini-2.5-flash"):
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
  
    #now create prompts
    prompt0_text = """
You are an expert evaluator of frames extracted from portfolio videos to be evaluated for a job application.
Understand the overall story flow and pick 5 non-similar frames based on the following criteria.

Please understand the overall story flow and the purpose of the video first.
Then pick 5 frames based on the following criteria, ensuring the 5 frames are not similar to each other.

Evaluation Criteria:
1. purpose: Purpose fitness (alignment with client goals/brand tone)
2. story: Narrative cues (flow, indication of transition points)
3. edit: Editing quality (rhythm, transitions, timing)
4. cine: Cinematography/composition/exposure/color
5. subject: Key subject/branding visibility (e.g., APEC logo)

Output the 5 selected frames in JSON format. 
- Key: The index of the frame (0-based, where 0 is the first frame, 1 is the second frame, etc.)
- Value: The reason for selection (2 lines explaining why this frame was chosen)

Example format:
{
  "0": "Reason for selecting first frame...",
  "5": "Reason for selecting sixth frame...",
  ...
}
"""
    
    # Create contents array: prompt text first, then all images
    print(f"(ing) Sending {len(image_parts)} images to Gemini API...")
    response = self.gemini_client.models.generate_content(
        model=gemini_model,
        contents=[prompt0_text] + image_parts
        # config=types.GenerateContentConfig(
        #     temperature=0.1
        # )
    )

    # Save the response as a json file
    with open(self.reponse_json, "w", encoding="utf-8") as f:
      json.dump(response.text, f, ensure_ascii=False, indent=2)

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
  
  def save_selected_frames(self):
    '''
    read the response.json and save the selected frames to the working directory.
    '''
    with open(self.reponse_json, "r", encoding="utf-8") as f:
      response_text = json.load(f)
    
    #parse the response text to get the selected frames
    selected_frames = json.loads(response_text)
    

  def gemini_api_call_method1(self, gemini_model = "gemini-2.5-flash-lite"):
    '''
    gemini_model:
    "gemini-2.5-flash"
    "gemini-2.5-pro"
    "gemini-2.5-flash-lite"
    
    30개의 이미지를 한 번에 전달하여 평가합니다.
    '''
    # Read selected frame numbers from JSON
    with open(self.selected_frames_json, "r", encoding="utf-8") as f:
      frame_numbers = json.load(f)
    
    if not frame_numbers:
      print("⚠️  No frames found in selected_frames.json")
      return None
    
    print(f"📸 Loading {len(frame_numbers)} images for evaluation...")
    
    # Load all images and convert to Parts
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
    
    if not image_parts:
      print("⚠️  No valid images to process")
      return None
    
    print(f"✅ Loaded {len(image_parts)} images successfully")
    
    # Create contents array: prompt text first, then all images
    contents = [self.prompt1_text] + image_parts
    print(f"(ing) Sending {len(image_parts)} images to Gemini API...")
    # Call API with all images at once
    from google.genai import types
    response = self.gemini_client.models.generate_content(
        model=gemini_model,
        contents=contents
        # config=types.GenerateContentConfig(
        #     temperature=0.1
        # )
    )

    # Save the response as a json file
    with open(self.working_dir / "video_summary.json", "w", encoding="utf-8") as f:
      json.dump(response.text, f, ensure_ascii=False, indent=2)
      
    # Convert usage_metadata to dictionary for JSON serialization
    usage_dict = {
      "prompt_token_count": response.usage_metadata.prompt_token_count,
      "candidates_token_count": response.usage_metadata.candidates_token_count,
      "total_token_count": response.usage_metadata.total_token_count,
    }
    with open(self.working_dir / "usage_metadata.json", "w", encoding="utf-8") as f:
      json.dump(usage_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ Response saved. Total tokens: {response.usage_metadata.total_token_count}")
    
    return response

  def gemini_api_call_method2(self, gemini_model = "gemini-2.5-flash-lite"):
    '''
    gemini_model:
    "gemini-2.5-flash"
    "gemini-2.5-pro"
    "gemini-2.5-flash-lite"
    
    Using caching:
    - 30개의 이미지를 시간 순서대로 10개씩 3번으로 나눠서 평가합니다.
    '''

    # Read selected frame numbers from JSON
    with open(self.selected_frames_json, "r", encoding="utf-8") as f:
      frame_numbers = json.load(f)
    total_frames = len(frame_numbers)
    print(f"Total {total_frames} frames to evaluate")
    
    # Evaluation prompt (캐싱 없이 사용 - 3번만 사용하므로 캐싱 오버헤드가 더 큼)
    evaluation_prompt = """
You are an expert evaluator of frames extracted from portfolio videos to be evaluated.
Your task is to evaluate the frames based on the following criteria. Each criterion should be scored from 0.0 to 1.0.

Evaluation Criteria:
1. purpose: Purpose fitness (alignment with client goals/brand tone)
2. story: Narrative cues (flow, indication of transition points)
3. edit: Editing quality (rhythm, transitions, timing)
4. cine: Cinematography/composition/exposure/color
5. subject: Key subject/branding visibility (e.g., APEC logo)

Evaluate frames in the order provided (first image = frame 1, second = frame 2, etc.)
Output ONLY valid JSON format (no additional text):

{
  "frames": [
    {
      "frame_id": 1,
      "score1": 0.85,
      "score2": 0.75,
      "score3": 0.65,
      "score4": 0.55,
      "score5": 0.45
    }
    ...
  ]
}
"""

    # Step 2: split 30 frames into 10 frames per batch
    batch_size = 10
    batches = []
    for i in range(0, total_frames, batch_size):
      batch = frame_numbers[i:i+batch_size]
      batches.append(batch)
    
    print(f"📦 Split into {len(batches)} batches: {[len(b) for b in batches]} frames each")
    
    # Step 3: evaluate each batch
    all_responses = []
    for batch_idx, batch_frame_numbers in enumerate(batches, 1):
      print(f"\n{'='*50}")
      print(f"📊 Processing Batch {batch_idx}/{len(batches)} ({len(batch_frame_numbers)} frames)")
      print(f"{'='*50}")
      
      # Load images for this batch
      image_parts = []
      for frame_num in batch_frame_numbers:
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
      
      if not image_parts:
        print(f"⚠️  No valid images in batch {batch_idx}")
        continue
      
      print(f"✅ Loaded {len(image_parts)} images for batch {batch_idx}")
      
      # Create contents: prompt + images (캐싱 사용 안 함)
      contents = [evaluation_prompt] + image_parts
      
      # Call API
      print(f"🚀 Sending batch {batch_idx} to Gemini API...")
      try:
        response = self.gemini_client.models.generate_content(
          model=gemini_model,
          contents=contents,
          # config=types.GenerateContentConfig(
          #   temperature=0.1
          # )
        )

        # Save response for this batch
        batch_result_file = self.working_dir / f"video_summary_batch{batch_idx}.json"
        with open(batch_result_file, "w", encoding="utf-8") as f:
          json.dump(response.text, f, ensure_ascii=False, indent=2)
        print(f"✅ Batch {batch_idx} result saved: {batch_result_file}")
        
        # Save usage metadata for this batch
        usage_dict = {
          "prompt_token_count": response.usage_metadata.prompt_token_count,
          "candidates_token_count": response.usage_metadata.candidates_token_count,
          "total_token_count": response.usage_metadata.total_token_count,
        }
        if hasattr(response.usage_metadata, 'cached_content_token_count'):
          usage_dict["cached_content_token_count"] = response.usage_metadata.cached_content_token_count
        
        usage_file = self.working_dir / f"usage_metadata_batch{batch_idx}.json"
        with open(usage_file, "w", encoding="utf-8") as f:
          json.dump(usage_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Batch {batch_idx} tokens saved: {usage_file}")
        print(f"   Total tokens: {response.usage_metadata.total_token_count}")
        
        all_responses.append(response)
        
      except Exception as e:
        print(f"❌ Error processing batch {batch_idx}: {e}")
        continue
    
    print(f"\n{'='*50}")
    print(f"✅ Completed processing {len(batches)} batches")
    print(f"{'='*50}")
    
    return all_responses






  def save_selected_frames_method1(self, top_n=5):
    '''
    Read the json file and select the top N frames with the highest score.
    Exclude frames from the same group to ensure diversity.
    Copy selected frames to selected_frames folder.
    '''
    import shutil
    import re
    
    # Read groups.json to get frame-to-group mapping
    groups_json = self.working_dir / "groups.json"
    frame_to_group = {}  # frame_num -> group_id
    if groups_json.exists():
      with open(groups_json, "r", encoding="utf-8") as f:
        groups_data = json.load(f)
      
      # Create mapping: frame_num -> group_id
      for group_id, frame_list in groups_data.items():
        for frame_num in frame_list:
          frame_to_group[frame_num] = group_id
      print(f"📋 Loaded {len(groups_data)} groups, {len(frame_to_group)} frames mapped")
    else:
      print(f"⚠️  groups.json not found at {groups_json}")
      print("   Proceeding without group filtering...")
    
    # Read JSON file
    json_file = self.working_dir / "video_summary_method1.json"
    with open(json_file, "r", encoding="utf-8") as f:
      response_text = json.load(f)
    
    # Parse JSON from response text (may contain markdown code blocks or be a string)
    data = None
    
    # Try to extract JSON from markdown code blocks (non-greedy match)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
      json_str = json_match.group(1)
      try:
        data = json.loads(json_str)
      except json.JSONDecodeError:
        pass
    
    # If not found, try to find JSON object directly (balanced braces)
    if data is None:
      # Find JSON object with balanced braces
      start_idx = response_text.find('{')
      if start_idx >= 0:
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(response_text)):
          if response_text[i] == '{':
            brace_count += 1
          elif response_text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
              end_idx = i + 1
              break
        if end_idx > start_idx:
          json_str = response_text[start_idx:end_idx]
          try:
            data = json.loads(json_str)
          except json.JSONDecodeError:
            pass
    
    # If still not found, try parsing the whole string
    if data is None:
      try:
        if isinstance(response_text, str):
          data = json.loads(response_text)
        else:
          data = response_text
      except (json.JSONDecodeError, TypeError):
        # Last resort: try to extract JSON manually
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
          json_str = response_text[start_idx:end_idx]
          try:
            data = json.loads(json_str)
          except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from {json_file}")
    
    if data is None:
      raise ValueError(f"Could not extract JSON data from {json_file}")
    
    # Calculate weighted total score for each frame
    # Weights: rel(0.15), story(0.20), edit(0.20), cine(0.15), subject(0.15)
    frame_scores = []
    for frame_key, scores in data.items():
      # Extract frame number from key (e.g., "1.png" -> 1)
      frame_num = int(re.search(r'\d+', frame_key).group())
      
      # Calculate weighted total score
      total_score = (
        scores.get('rel', 0) * 0.15 +
        scores.get('story', 0) * 0.20 +
        scores.get('edit', 0) * 0.20 +
        scores.get('cine', 0) * 0.15 +
        scores.get('subject', 0) * 0.15
      )
      
      # Get group ID for this frame
      group_id = frame_to_group.get(frame_num, None)
      
      frame_scores.append({
        'frame_num': frame_num,
        'frame_key': frame_key,
        'scores': scores,
        'total_score': total_score,
        'group_id': group_id
      })
    
    # Sort by total score (descending)
    frame_scores.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Select top N frames, prioritizing diversity but allowing duplicates if needed
    # Strategy: Allow up to 3 frames per group to ensure we can select top_n frames
    top_frames = []
    selected_groups = {}  # Track groups: group_id -> count of frames selected from this group
    max_per_group = 3  # Maximum frames allowed from the same group (increased to ensure top_n selection)
    
    for frame_info in frame_scores:
      if len(top_frames) >= top_n:
        break
      
      frame_group = frame_info['group_id']
      
      # If groups.json exists and frame has a group
      if frame_to_group and frame_group is not None:
        group_count = selected_groups.get(frame_group, 0)
        
        # Allow up to max_per_group frames from the same group
        if group_count >= max_per_group:
          continue  # Already have enough from this group, skip
        
        # Update group count
        selected_groups[frame_group] = group_count + 1
      # If no group info, allow selection
      
      top_frames.append(frame_info)
    
    print(f"\n📊 Top {top_n} frames selected (ensuring group diversity):")
    for i, frame_info in enumerate(top_frames, 1):
      group_info = f"Group {frame_info['group_id']}" if frame_info['group_id'] is not None else "No group"
      print(f"  {i}. Frame {frame_info['frame_num']} ({group_info}): {frame_info['total_score']:.4f} "
            f"(rel:{frame_info['scores'].get('rel', 0):.2f}, "
            f"story:{frame_info['scores'].get('story', 0):.2f}, "
            f"edit:{frame_info['scores'].get('edit', 0):.2f}, "
            f"cine:{frame_info['scores'].get('cine', 0):.2f}, "
            f"subject:{frame_info['scores'].get('subject', 0):.2f})")
    
    if len(top_frames) < top_n:
      print(f"\n⚠️  Warning: Only {len(top_frames)} frames selected (requested {top_n})")
      print(f"   This may be due to group filtering limiting available frames.")
    
    # Create selected_frames directory
    selected_frames_dir = self.working_dir / "selected_frames"
    selected_frames_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created directory: {selected_frames_dir}")
    
    # Copy selected frames from total_frames to selected_frames
    copied_files = []
    for frame_info in top_frames:
      frame_num = frame_info['frame_num']
      source_path = frame_number_to_path(self.total_frame_dir, frame_num)
      
      if not source_path.exists():
        print(f"⚠️  Frame {frame_num} not found: {source_path}")
        continue
      
      # Copy to selected_frames directory
      dest_path = selected_frames_dir / f"{frame_num}.jpg"
      shutil.copy2(source_path, dest_path)
      copied_files.append(frame_num)
      print(f"✅ Copied frame {frame_num}: {source_path.name} -> {dest_path.name}")
    
    # Save selected frame numbers to JSON
    selected_frames_json = self.working_dir / "selected_frames_final.json"
    with open(selected_frames_json, "w", encoding="utf-8") as f:
      json.dump(copied_files, f, indent=2)
    print(f"\n✅ Saved selected frame numbers to: {selected_frames_json}")
    print(f"✅ Total {len(copied_files)} frames copied to {selected_frames_dir}")
    
    return copied_files

  def save_selected_frames_method2(self):
    '''
    read the json file and select the 5 frames with the highest score.
    '''
    for batch_idx in range(1, 4):
      with open(self.working_dir / f"video_summary_batch{batch_idx}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
      print(data)
    pass
