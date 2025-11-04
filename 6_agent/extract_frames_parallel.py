import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from sklearn.cluster import KMeans
import time

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

# ============================================================================
# AGENT 1: 시간 기반 균등 분할 (기승전결)
# ============================================================================

def agent1_time_based(video_path: str, num_frames: int = 20, output_dir: str = "frames_agent1") -> List[Tuple[int, float]]:
    """시간 기반으로 기승전결 구조에 맞게 프레임 추출"""
    os.makedirs(output_dir, exist_ok=True)
    
    info = get_video_info(video_path)
    duration = info['duration']
    fps = info['fps']
    
    # 기승전결 비율: 25%, 30%, 30%, 15%
    segments = [
        (0.0, 0.25, 5),      # 기: 5장
        (0.25, 0.55, 7),      # 승: 7장
        (0.55, 0.85, 7),      # 전: 7장
        (0.85, 1.0, 3)       # 결: 3장
    ]
    
    selected_frames = []
    cap = cv2.VideoCapture(video_path)
    
    for seg_start, seg_end, frame_count in segments:
        start_time = duration * seg_start
        end_time = duration * seg_end
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # 구간을 균등 분할
        sub_intervals = np.linspace(start_frame, end_frame, frame_count + 1)
        
        segment_scores = []
        prev_frame = None
        
        for i in range(len(sub_intervals) - 1):
            sub_start = int(sub_intervals[i])
            sub_end = int(sub_intervals[i + 1])
            sub_mid = (sub_start + sub_end) // 2
            
            # 서브구간에서 프레임 분석
            cap.set(cv2.CAP_PROP_POS_FRAMES, sub_start)
            ret, frame = cap.read()
            if ret:
                if prev_frame is not None:
                    score = calculate_frame_score(prev_frame, frame)
                    segment_scores.append((sub_mid, score))
                prev_frame = frame.copy()
        
        # 각 서브구간에서 최고 점수 프레임 선택
        if segment_scores:
            segment_scores.sort(key=lambda x: x[1], reverse=True)
            selected_frames.extend(segment_scores[:frame_count])
    
    cap.release()
    
    # 최소 시간 간격 보장 (너무 가까운 프레임 제거)
    min_interval = duration * 0.03  # 최소 3% 간격
    filtered_frames = []
    selected_frames.sort(key=lambda x: x[0])
    
    for frame_num, score in selected_frames:
        if not filtered_frames:
            filtered_frames.append((frame_num, score))
        else:
            last_frame = filtered_frames[-1][0]
            time_diff = abs(frame_num - last_frame) / fps
            if time_diff >= min_interval:
                filtered_frames.append((frame_num, score))
    
    # 최종 20개 선택
    filtered_frames = filtered_frames[:num_frames]
    
    # 저장
    cap = cv2.VideoCapture(video_path)
    for idx, (frame_num, score) in enumerate(filtered_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            filename = f"agent1_frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
    cap.release()
    
    return filtered_frames

# ============================================================================
# AGENT 2: PySceneDetect 기반
# ============================================================================

def agent2_scene_detection(video_path: str, num_frames: int = 20, output_dir: str = "frames_agent2") -> List[Tuple[int, float]]:
    """PySceneDetect로 장면 전환 감지 후 각 장면에서 프레임 추출"""
    os.makedirs(output_dir, exist_ok=True)
    
    info = get_video_info(video_path)
    fps = info['fps']
    
    # Scene detection
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    
    video_manager.set_duration()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    
    if not scene_list:
        # Scene이 없으면 시간 기반으로 분할
        duration = info['duration']
        num_scenes = 10
        scene_length = duration / num_scenes
        scene_list = [(i * scene_length, (i + 1) * scene_length) for i in range(num_scenes)]
    
    # 각 장면에서 프레임 선택
    frames_per_scene = max(1, num_frames // len(scene_list))
    selected_frames = []
    
    cap = cv2.VideoCapture(video_path)
    
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = int(start_time.get_seconds() * fps) if hasattr(start_time, 'get_seconds') else int(start_time * fps)
        end_frame = int(end_time.get_seconds() * fps) if hasattr(end_time, 'get_seconds') else int(end_time * fps)
        
        scene_scores = []
        prev_frame = None
        
        # 장면 내에서 샘플링
        sample_frames = np.linspace(start_frame, end_frame, min(20, end_frame - start_frame), dtype=int)
        
        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                if prev_frame is not None:
                    score = calculate_frame_score(prev_frame, frame)
                    scene_scores.append((frame_num, score))
                prev_frame = frame.copy()
        
        if scene_scores:
            scene_scores.sort(key=lambda x: x[1], reverse=True)
            selected_frames.extend(scene_scores[:frames_per_scene])
    
    cap.release()
    
    # 최종 20개 선택
    selected_frames = selected_frames[:num_frames]
    
    # 저장
    cap = cv2.VideoCapture(video_path)
    for idx, (frame_num, score) in enumerate(selected_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            filename = f"agent2_frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
    cap.release()
    
    return selected_frames

# ============================================================================
# AGENT 3: 색상 클러스터링 기반
# ============================================================================

def agent3_color_clustering(video_path: str, num_frames: int = 20, output_dir: str = "frames_agent3") -> List[Tuple[int, float]]:
    """색상 클러스터링으로 시각적 다양성 보장하며 프레임 추출"""
    os.makedirs(output_dir, exist_ok=True)
    
    info = get_video_info(video_path)
    fps = info['fps']
    
    # 샘플링 (성능 최적화)
    sample_interval = max(1, info['frame_count'] // 200)  # 최대 200개 샘플
    cap = cv2.VideoCapture(video_path)
    
    frame_features = []
    frame_indices = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            # 색상 히스토그램을 특징으로 사용
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            feature = hist.flatten()
            frame_features.append(feature)
            frame_indices.append(frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frame_features) < num_frames:
        # 샘플이 부족하면 모든 프레임 사용
        frame_features = []
        frame_indices = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            frame_features.append(hist.flatten())
            frame_indices.append(frame_idx)
            frame_idx += 1
        cap.release()
    
    # K-means 클러스터링
    features_array = np.array(frame_features)
    kmeans = KMeans(n_clusters=num_frames, random_state=42, n_init=10)
    kmeans.fit(features_array)
    
    # 각 클러스터에서 중심에 가장 가까운 프레임 선택
    selected_indices = []
    for cluster_id in range(num_frames):
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_features = features_array[cluster_mask]
        cluster_frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if cluster_mask[i]]
        
        if len(cluster_features) > 0:
            # 클러스터 중심과의 거리 계산
            distances = np.linalg.norm(cluster_features - kmeans.cluster_centers_[cluster_id], axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(cluster_frame_indices[closest_idx])
    
    # 저장
    cap = cv2.VideoCapture(video_path)
    selected_frames = []
    for idx, frame_num in enumerate(selected_indices[:num_frames], 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            filename = f"agent3_frame_{idx:02d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            selected_frames.append((frame_num, 0.0))  # 점수는 0으로 설정
    cap.release()
    
    return selected_frames

# ============================================================================
# AGENT 4: 하이브리드 (시간 + Scene Detection)
# ============================================================================

def agent4_hybrid(video_path: str, num_frames: int = 20, output_dir: str = "frames_agent4") -> List[Tuple[int, float]]:
    """시간 기반 분할 + 각 구간에서 Scene Detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    info = get_video_info(video_path)
    duration = info['duration']
    fps = info['fps']
    
    # 시간 기반으로 4개 구간 분할
    segments = [
        (0.0, 0.25, 5),
        (0.25, 0.55, 7),
        (0.55, 0.85, 7),
        (0.85, 1.0, 3)
    ]
    
    selected_frames = []
    cap = cv2.VideoCapture(video_path)
    
    for seg_start, seg_end, frame_count in segments:
        start_time = duration * seg_start
        end_time = duration * seg_end
        
        # 구간 내에서 Scene Detection
        temp_video_path = video_path
        video_manager = VideoManager([temp_video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        
        video_manager.set_duration(start_time=start_time, end_time=end_time)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        if not scene_list:
            # Scene이 없으면 균등 분할
            scene_list = [(start_time, end_time)]
        
        # 각 장면에서 프레임 선택
        frames_per_scene = max(1, frame_count // len(scene_list))
        segment_scores = []
        prev_frame = None
        
        for start_scene, end_scene in scene_list:
            start_frame = int(start_scene.get_seconds() * fps) if hasattr(start_scene, 'get_seconds') else int(start_scene * fps)
            end_frame = int(end_scene.get_seconds() * fps) if hasattr(end_scene, 'get_seconds') else int(end_scene * fps)
            
            sample_frames = np.linspace(start_frame, end_frame, min(10, end_frame - start_frame), dtype=int)
            
            for frame_num in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    if prev_frame is not None:
                        score = calculate_frame_score(prev_frame, frame)
                        segment_scores.append((frame_num, score))
                    prev_frame = frame.copy()
        
        if segment_scores:
            segment_scores.sort(key=lambda x: x[1], reverse=True)
            selected_frames.extend(segment_scores[:frame_count])
    
    cap.release()
    
    # 최소 시간 간격 보장
    min_interval = duration * 0.03
    filtered_frames = []
    selected_frames.sort(key=lambda x: x[0])
    
    for frame_num, score in selected_frames:
        if not filtered_frames:
            filtered_frames.append((frame_num, score))
        else:
            last_frame = filtered_frames[-1][0]
            time_diff = abs(frame_num - last_frame) / fps
            if time_diff >= min_interval:
                filtered_frames.append((frame_num, score))
    
    filtered_frames = filtered_frames[:num_frames]
    
    # 저장
    cap = cv2.VideoCapture(video_path)
    for idx, (frame_num, score) in enumerate(filtered_frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_num / fps
            filename = f"agent4_frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
    cap.release()
    
    return filtered_frames

# ============================================================================
# AGENT 5: Max-Min Diversity
# ============================================================================

def agent5_maxmin_diversity(video_path: str, num_frames: int = 20, output_dir: str = "frames_agent5") -> List[Tuple[int, float]]:
    """Max-Min Diversity: 다양성과 품질 균형"""
    os.makedirs(output_dir, exist_ok=True)
    
    info = get_video_info(video_path)
    fps = info['fps']
    
    # 샘플링
    sample_interval = max(1, info['frame_count'] // 100)  # 최대 100개 샘플
    cap = cv2.VideoCapture(video_path)
    
    candidate_frames = []
    frame_data = []
    prev_frame = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            score = 0.0
            if prev_frame is not None:
                score = calculate_frame_score(prev_frame, frame)
            
            # 특징 벡터 (히스토그램 + 점수)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            feature = hist.flatten()
            
            candidate_frames.append((frame_idx, score))
            frame_data.append(feature)
            prev_frame = frame.copy()
        
        frame_idx += 1
    
    cap.release()
    
    if len(candidate_frames) < num_frames:
        # 샘플이 부족하면 더 많이 샘플링
        candidate_frames = []
        frame_data = []
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            score = 0.0
            if prev_frame is not None:
                score = calculate_frame_score(prev_frame, frame)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            candidate_frames.append((frame_idx, score))
            frame_data.append(hist.flatten())
            prev_frame = frame.copy()
            frame_idx += 1
        cap.release()
    
    features_array = np.array(frame_data)
    
    # 첫 프레임: 최고 점수
    selected_indices = []
    candidate_frames.sort(key=lambda x: x[1], reverse=True)
    selected_indices.append(candidate_frames[0][0])
    
    # 이후: 이미 선택된 것과 가장 다르면서 점수가 높은 것
    for _ in range(num_frames - 1):
        best_frame_idx = None
        best_score = -1
        
        selected_features = features_array[[candidate_frames.index((idx, s)) for idx, s in candidate_frames if idx in selected_indices]]
        
        for idx, (frame_idx, score) in enumerate(candidate_frames):
            if frame_idx in selected_indices:
                continue
            
            # 최소 거리 계산 (다양성)
            distances = np.linalg.norm(selected_features - features_array[idx], axis=1)
            min_distance = np.min(distances)
            
            # 다양성 점수 + 품질 점수
            diversity_score = min_distance * 0.7 + score * 0.3
            
            if diversity_score > best_score:
                best_score = diversity_score
                best_frame_idx = frame_idx
        
        if best_frame_idx is not None:
            selected_indices.append(best_frame_idx)
    
    # 저장
    cap = cv2.VideoCapture(video_path)
    selected_frames = []
    for idx, frame_num in enumerate(selected_indices[:num_frames], 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            score = next((s for f_idx, s in candidate_frames if f_idx == frame_num), 0.0)
            timestamp = frame_num / fps
            filename = f"agent5_frame_{idx:02d}_t{timestamp:.2f}s_score{score:.3f}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            selected_frames.append((frame_num, score))
    cap.release()
    
    return selected_frames

# ============================================================================
# 병렬 실행 래퍼 함수
# ============================================================================

def run_agent(agent_func, agent_name, video_path, num_frames, output_dir):
    """Agent 실행 래퍼"""
    start_time = time.time()
    try:
        result = agent_func(video_path, num_frames, output_dir)
        elapsed = time.time() - start_time
        return {
            'agent': agent_name,
            'success': True,
            'frames': result,
            'time': elapsed,
            'output_dir': output_dir
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'agent': agent_name,
            'success': False,
            'error': str(e),
            'time': elapsed,
            'output_dir': output_dir
        }

# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    video_path = "Fishing.mp4"
    num_frames = 20
    
    agents = [
        (agent1_time_based, "Agent1_시간기반", "frames_agent1"),
        (agent2_scene_detection, "Agent2_SceneDetection", "frames_agent2"),
        (agent3_color_clustering, "Agent3_클러스터링", "frames_agent3"),
        (agent4_hybrid, "Agent4_하이브리드", "frames_agent4"),
        (agent5_maxmin_diversity, "Agent5_MaxMinDiversity", "frames_agent5"),
    ]
    
    print("=" * 80)
    print("5개 Agent 병렬 실행 시작")
    print("=" * 80)
    
    results = []
    
    # 병렬 실행 (ThreadPoolExecutor 사용 - OpenCV/PySceneDetect 호환성)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_agent, func, name, video_path, num_frames, output_dir): name
            for func, name, output_dir in agents
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✅ 성공" if result['success'] else "❌ 실패"
                print(f"{result['agent']}: {status} ({result['time']:.2f}초)")
                if not result['success']:
                    print(f"  오류: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"{agent_name}: ❌ 예외 발생 - {str(e)}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    
    for result in results:
        if result['success']:
            print(f"\n{result['agent']}:")
            print(f"  - 처리 시간: {result['time']:.2f}초")
            print(f"  - 추출된 프레임: {len(result['frames'])}개")
            print(f"  - 저장 위치: {result['output_dir']}/")
            
            # 시간 분포 확인
            timestamps = [frame_num / 24.0 for frame_num, _ in result['frames']]
            print(f"  - 시간 범위: {min(timestamps):.2f}초 ~ {max(timestamps):.2f}초")
    
    print("\n" + "=" * 80)
    print("✅ 모든 Agent 실행 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

