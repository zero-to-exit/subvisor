import cv2
import numpy as np
import os
from typing import List, Tuple
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

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
    scores.append(diff_score * 2.0)
    
    # 2. 밝기 변화 (fade in/out)
    prev_brightness = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    curr_brightness = np.mean(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
    brightness_change = abs(curr_brightness - prev_brightness) / 255.0
    scores.append(brightness_change * 1.5)
    
    # 3. 색상 히스토그램 변화
    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    hist_score = 1.0 - hist_diff
    scores.append(hist_score * 1.2)
    
    # 4. 엣지 검출 (움직임/액션)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_edges = cv2.Canny(prev_gray, 50, 150)
    curr_edges = cv2.Canny(curr_gray, 50, 150)
    edge_diff = np.sum(cv2.absdiff(prev_edges, curr_edges)) / (prev_edges.shape[0] * prev_edges.shape[1])
    scores.append(edge_diff * 1.0)
    
    return sum(scores)

def extract_frames_hybrid(video_path: str, num_frames: int = 20, output_dir: str = "frames_hybrid") -> List[Tuple[int, float]]:
    """
    하이브리드 방식: 시간 기반 분할(기승전결) + 각 구간에서 Scene Detection
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("하이브리드 방식 프레임 추출 시작")
    print("=" * 80)
    
    # 영상 정보 확인
    info = get_video_info(video_path)
    duration = info['duration']
    fps = info['fps']
    
    print(f"\n영상 정보:")
    print(f"  - 길이: {duration:.2f}초")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 총 프레임 수: {info['frame_count']}")
    print(f"  - 해상도: {info['width']}x{info['height']}")
    
    # 기승전결 비율로 구간 분할
    segments = [
        (0.0, 0.25, 5, "기 (도입)"),      # 5장
        (0.25, 0.55, 7, "승 (전개)"),      # 7장
        (0.55, 0.85, 7, "전 (절정)"),      # 7장
        (0.85, 1.0, 3, "결 (결말)")       # 3장
    ]
    
    selected_frames = []
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n구간별 분석 시작...")
    
    for seg_idx, (seg_start, seg_end, frame_count, seg_name) in enumerate(segments, 1):
        start_time = duration * seg_start
        end_time = duration * seg_end
        
        print(f"\n[{seg_idx}/4] {seg_name} 구간 분석 중... ({start_time:.1f}초 ~ {end_time:.1f}초)")
        
        # 구간 내에서 Scene Detection 실행
        try:
            from scenedetect import detect, ContentDetector, split_video_ffmpeg
            
            # Scene Detection (전체 영상에서 실행 후 구간 필터링)
            scene_list_all = detect(video_path, ContentDetector())
            
            # 현재 구간에 해당하는 장면만 필터링
            scene_list = []
            for start_scene, end_scene in scene_list_all:
                start_sec = start_scene.get_seconds()
                end_sec = end_scene.get_seconds()
                
                # 구간과 겹치는 장면만 선택
                if start_sec < end_time and end_sec > start_time:
                    # 구간 내로 클리핑
                    clipped_start = max(start_sec, start_time)
                    clipped_end = min(end_sec, end_time)
                    scene_list.append((clipped_start, clipped_end))
            
            if not scene_list:
                print(f"  장면 전환을 찾지 못했습니다. 균등 분할 사용")
                scene_list = [(start_time, end_time)]
            else:
                print(f"  {len(scene_list)}개의 장면을 감지했습니다")
        
        except Exception as e:
            print(f"  Scene Detection 오류: {e}. 균등 분할 사용")
            scene_list = [(start_time, end_time)]
        
        # 각 장면에서 프레임 선택
        segment_scores = []
        prev_frame = None
        
        # 장면별로 프레임 수 할당
        frames_per_scene = max(1, frame_count // len(scene_list))
        remaining_frames = frame_count
        
        for scene_idx, (start_scene, end_scene) in enumerate(scene_list):
            # 시간 처리 (float 또는 FrameTimecode)
            if hasattr(start_scene, 'get_seconds'):
                start_scene_sec = start_scene.get_seconds()
            else:
                start_scene_sec = float(start_scene)
            
            if hasattr(end_scene, 'get_seconds'):
                end_scene_sec = end_scene.get_seconds()
            else:
                end_scene_sec = float(end_scene)
            
            start_frame = int(start_scene_sec * fps)
            end_frame = int(end_scene_sec * fps)
            
            # 장면 내에서 샘플링 (최대 15개 샘플)
            num_samples = min(15, end_frame - start_frame)
            if num_samples > 0:
                sample_frames = np.linspace(start_frame, end_frame, num_samples, dtype=int)
            else:
                sample_frames = [start_frame]
            
            scene_scores = []
            
            for frame_num in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    if prev_frame is not None:
                        score = calculate_frame_score(prev_frame, frame)
                        scene_scores.append((frame_num, score))
                    prev_frame = frame.copy()
            
            # 장면 내에서 상위 프레임 선택
            if scene_scores:
                scene_scores.sort(key=lambda x: x[1], reverse=True)
                selected_from_scene = min(frames_per_scene, remaining_frames, len(scene_scores))
                segment_scores.extend(scene_scores[:selected_from_scene])
                remaining_frames -= selected_from_scene
        
        # 구간 내 프레임 선택
        if segment_scores:
            segment_scores.sort(key=lambda x: x[1], reverse=True)
            selected_frames.extend(segment_scores[:frame_count])
            print(f"  {len(segment_scores[:frame_count])}개 프레임 선택 완료")
        else:
            print(f"  경고: 프레임을 선택하지 못했습니다")
    
    cap.release()
    
    # 최소 시간 간격 보장 (너무 가까운 프레임 제거)
    min_interval = duration * 0.02  # 최소 2% 간격 (더 관대하게)
    filtered_frames = []
    selected_frames.sort(key=lambda x: x[0])
    
    print(f"\n최소 시간 간격 필터링 중... (최소 간격: {min_interval:.2f}초)")
    for frame_num, score in selected_frames:
        if not filtered_frames:
            filtered_frames.append((frame_num, score))
        else:
            last_frame = filtered_frames[-1][0]
            time_diff = abs(frame_num - last_frame) / fps
            if time_diff >= min_interval:
                filtered_frames.append((frame_num, score))
    
    # 부족한 경우 점수 기준으로 추가 선택
    if len(filtered_frames) < num_frames:
        print(f"  프레임이 부족합니다 ({len(filtered_frames)}/{num_frames}). 추가 선택 중...")
        remaining = [f for f in selected_frames if f not in filtered_frames]
        remaining.sort(key=lambda x: x[1], reverse=True)
        
        for frame_num, score in remaining:
            if len(filtered_frames) >= num_frames:
                break
            # 이미 선택된 것과 충분히 떨어져 있는지 확인
            can_add = True
            for existing_frame, _ in filtered_frames:
                time_diff = abs(frame_num - existing_frame) / fps
                if time_diff < min_interval:
                    can_add = False
                    break
            if can_add:
                filtered_frames.append((frame_num, score))
        
        # 여전히 부족하면 간격을 무시하고 추가
        if len(filtered_frames) < num_frames:
            remaining = [f for f in selected_frames if f not in filtered_frames]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for frame_num, score in remaining:
                if len(filtered_frames) >= num_frames:
                    break
                filtered_frames.append((frame_num, score))
    
    # 최종 정렬
    filtered_frames.sort(key=lambda x: x[0])
    filtered_frames = filtered_frames[:num_frames]
    
    print(f"\n최종 {len(filtered_frames)}개 프레임 추출 완료")
    print("=" * 80)
    
    # 프레임 저장
    print(f"\n프레임 저장 중...")
    cap = cv2.VideoCapture(video_path)
    
    for idx, (frame_num, score) in enumerate(filtered_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            filename = f"hybrid_frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  [{idx:2d}/20] 프레임 {frame_num} 저장: {filename} (점수: {score:.3f})")
    
    cap.release()
    
    # 시간 분포 확인
    timestamps = [frame_num / fps for frame_num, _ in filtered_frames]
    print(f"\n시간 분포:")
    print(f"  - 범위: {min(timestamps):.2f}초 ~ {max(timestamps):.2f}초")
    print(f"  - 평균 간격: {np.mean(np.diff(sorted(timestamps))):.2f}초")
    
    print(f"\n✅ 모든 프레임이 '{output_dir}' 디렉토리에 저장되었습니다.")
    print("=" * 80)
    
    return filtered_frames

if __name__ == "__main__":
    video_path = "Fishing.mp4"
    extract_frames_hybrid(video_path, num_frames=20, output_dir="frames_hybrid")

