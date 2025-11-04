import cv2
import numpy as np
import os
from typing import List, Tuple

def get_video_info(video_path: str) -> dict:
    """영상 파일의 기본 정보를 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

def calculate_frame_score(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                          next_frame: np.ndarray = None) -> float:
    """프레임의 편집 기술 점수를 계산합니다."""
    if prev_frame is None:
        return 0.0
    
    scores = []
    
    # 1. 프레임 간 차이 (scene change detection)
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff_score = np.mean(diff) / 255.0
    scores.append(diff_score * 2.0)  # 가중치 2.0
    
    # 2. 밝기 변화 (fade in/out)
    prev_brightness = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    curr_brightness = np.mean(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
    brightness_change = abs(curr_brightness - prev_brightness) / 255.0
    scores.append(brightness_change * 1.5)  # 가중치 1.5
    
    # 3. 색상 히스토그램 변화
    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    hist_score = 1.0 - hist_diff  # 상관관계가 낮을수록 변화가 큼
    scores.append(hist_score * 1.2)  # 가중치 1.2
    
    # 4. 엣지 검출 (움직임/액션)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = cv2.Canny(prev_gray, 50, 150)
    curr_edges = cv2.Canny(curr_gray, 50, 150)
    edge_diff = np.sum(cv2.absdiff(prev_edges, curr_edges)) / (prev_edges.shape[0] * prev_edges.shape[1])
    scores.append(edge_diff * 1.0)  # 가중치 1.0
    
    # 5. 다음 프레임과의 비교 (변화의 시작점 감지)
    if next_frame is not None:
        next_diff = cv2.absdiff(curr_frame, next_frame)
        next_diff_score = np.mean(next_diff) / 255.0
        # 현재 프레임이 변화의 시작점인 경우 점수 추가
        if next_diff_score > diff_score * 1.5:
            scores.append(0.5)
    
    return sum(scores)

def extract_best_frames(video_path: str, num_frames: int = 20, output_dir: str = "frames") -> List[Tuple[int, float]]:
    """영상에서 편집 기술을 잘 보여주는 프레임을 추출합니다."""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 영상 정보 확인
    print("=" * 60)
    print("영상 정보 확인 중...")
    info = get_video_info(video_path)
    print(f"영상 길이: {info['duration']:.2f}초")
    print(f"FPS: {info['fps']:.2f}")
    print(f"총 프레임 수: {info['frame_count']}")
    print(f"해상도: {info['width']}x{info['height']}")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    frame_scores = []
    prev_frame = None
    prev_gray = None
    
    print("\n프레임 분석 중...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 매 N번째 프레임만 분석 (성능 최적화)
        if frame_idx % 10 == 0:  # 10프레임마다 분석
            if prev_frame is not None:
                # 다음 프레임 미리 읽기
                next_frame = None
                next_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret_next, next_frame = cap.read()
                if ret_next:
                    score = calculate_frame_score(prev_frame, frame, next_frame)
                    frame_scores.append((frame_idx, score))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                else:
                    score = calculate_frame_score(prev_frame, frame)
                    frame_scores.append((frame_idx, score))
                
                if frame_idx % 100 == 0:
                    print(f"  분석 진행: {frame_idx}/{info['frame_count']} 프레임 처리 완료")
        
        prev_frame = frame.copy()
        frame_idx += 1
    
    cap.release()
    
    # 점수 기준으로 정렬하고 상위 N개 선택
    print(f"\n총 {len(frame_scores)}개의 프레임 후보를 분석했습니다.")
    frame_scores.sort(key=lambda x: x[1], reverse=True)
    selected_frames = frame_scores[:num_frames]
    
    print(f"\n상위 {num_frames}개 프레임 선택 완료")
    print("=" * 60)
    
    # 선택된 프레임 추출 및 저장
    print("\n프레임 추출 및 저장 중...")
    cap = cv2.VideoCapture(video_path)
    
    for idx, (frame_num, score) in enumerate(selected_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # 시간 계산
            timestamp = frame_num / info['fps']
            filename = f"frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  [{idx:2d}/20] 프레임 {frame_num} 저장: {filename} (점수: {score:.3f})")
    
    cap.release()
    
    print(f"\n✅ 총 {len(selected_frames)}개의 프레임이 '{output_dir}' 디렉토리에 저장되었습니다.")
    
    return selected_frames

if __name__ == "__main__":
    video_path = "Fishing.mp4"
    extract_best_frames(video_path, num_frames=20, output_dir="frames")

