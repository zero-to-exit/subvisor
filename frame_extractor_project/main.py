'''
All the procedures for the demo

1. Get the video from the web (save_video_from_web.py)
2. Now read the video from the working dir.
3. Extract frames from video(1000-1300)
4. Dividde the frames into groups based on the similarity
(hard thing in here afrter the division)
(30)

5. select the few frames on each group(Algorithm 1)
(Selection algoirthm will be implemented)

6. save the selected frames(20~30 frames)
'''

import os
import time
from pathlib import Path
from save_video_from_web import set_upload_dir, get_latest_uploaded_video


def extract_frames(video_path: Path, working_dir: Path):
    """
    업로드된 비디오를 처리하는 메인 로직
    
    Args:
        video_path: 처리할 비디오 파일 경로
        working_dir: 작업 디렉토리
    """
    print(f"📹 Extract frames from video: {video_path}")

    from models.algorithm1 import Algorithm1

    #Inital the process path
    alg1 = Algorithm1(working_dir, str(video_path))

    #Extract all frames from the video
    alg1.extract_all_frames(ratio=0.1)

    #Divide the frames into groups(30) based on the similarity
    alg1.divide_frames_into_groups(group_count=30)

    #Score the frames based on the criteria(30 frames) and select a frames on each group.
    alg1.score_frames(target_count=30, per_group_selected=1, yolo_weights="yolov8n.pt")

    #Save the selected frames
    alg1.save_selected_frames()

    
    from models.algorithm2 import Algorithm2

    #initalize GEMINI API
    GEMINI_API_KEY = "AIzaSyAURDg_1WcC7g2gx7-NZQ5JS-FHPTZlUvo"
    alg2 = Algorithm2(working_dir, api_key=GEMINI_API_KEY)
    
    #Call the Gemini API to score the frames we got Raw response in here from gemini.
    alg2.gemini_api_call(gemini_model="gemini-2.5-flash")

    #Now from the raw response, we need to extract the dictionary from the response.
    #in the dict -> key: frame number, value: reason for selection
    dict_response = alg2.extract_dict_from_response()
    
    if dict_response is None:
        print("❌ dict_response가 None입니다. Algorithm2를 건너뜁니다.")
    elif isinstance(dict_response, dict) and "raw_response" in dict_response:
        print("❌ JSON 파싱 실패로 인해 Algorithm2를 건너뜁니다.")
        print("   response.json 파일을 확인하여 Gemini API 응답 형식을 확인하세요.")
    else:

        #Save the selected frames to the working directory
        alg2.save_selected_frames(dict_response)
    

def main(working_dir: Path, upload_dir: Path = None, wait_for_upload: bool = True):
    """
    메인 함수: 서버를 시작하고 업로드된 비디오를 처리
    
    Args:
        working_dir: 작업 디렉토리
        upload_dir: 업로드 디렉토리 (None이면 working_dir/uploaded_videos 사용)
        wait_for_upload: 업로드를 기다릴지 여부 (False면 서버만 시작)
    """
    if upload_dir is None:
        upload_dir = working_dir / "uploaded_videos"
    
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(exist_ok=True)
    
    # 업로드 디렉토리 설정
    set_upload_dir(upload_dir)
    
    # 작업 디렉토리 설정 (프레임 표시용)
    from save_video_from_web import set_working_dir
    set_working_dir(working_dir)
    
    # 서버 시작
    print(f"🚀 Video Upload Server is starting")
    print(f"📁 Upload Directory: {upload_dir}")
    print(f"🌐 Please access http://localhost:8001 in your browser")
    
    # Run server in background
    import threading
    import uvicorn
    
    def run_server_thread():
        uvicorn.run(
            "save_video_from_web:app",
            host="0.0.0.0",
            port=8001,
            reload=False
        )
    
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    if wait_for_upload:
        print("\n⏳ Waiting for video upload... (will be processed automatically when upload is complete)")
        print("   브라우저에서 http://localhost:8001 에 접속하여 비디오를 업로드하세요.\n")
        
        # Detect and process uploaded videos
        processed_videos = set()
        
        try:
            while True:
                latest_video = get_latest_uploaded_video(upload_dir)
                
                if latest_video and latest_video not in processed_videos:
                    print(f"✅ New video detected: {latest_video.name}")
                    processed_videos.add(latest_video)
                    # Process the video
                    extract_frames(latest_video, working_dir)

                time.sleep(1)  # Check every 1 second
                
        except KeyboardInterrupt:
            print("\n\n👋 Server is shutting down.")
    else:
        print("Server is running. Please upload a video.")

    
    # Updated 1214
    # 프레임 표시 기능은 save_video_from_web.py에 통합됨
    # 별도의 서버가 필요 없음
    
    

if __name__ == "__main__":

    #프레임 제작 및 저장 공간.
    working_dir = Path(os.getcwd()) / "frames_1214"
    working_dir.mkdir(exist_ok=True)
    main(working_dir, wait_for_upload=True)
