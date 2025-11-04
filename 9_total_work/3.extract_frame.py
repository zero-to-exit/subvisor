'''
Now we need to extrat frames from the downloaded videos.

here are the rules.
extract total duration(s) as frames.
for example, if the duration is 10 seconds, extract 10 frames.

first we don't need to extract the similar farmes right?
결국 비슷한 프레임끼리는 묶어서 거기시 가장 퀄리티가 좋은 대표샷을 남기는게 좋아.
그렇기에 장면전환을 pySceneDetect를 사용하여 추출하고, 그 그룹들 중에서 가장 퀄리티가 좋은 대표샷을 남기는게 좋아.
'''
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
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


def calculate_blur_score(frame: np.ndarray) -> float:
    """
    프레임의 블러 정도를 계산합니다.
    Laplacian variance를 사용 (값이 낮을수록 블러가 심함)
    Returns: 블러 점수 (낮을수록 블러가 심함)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var


def calculate_quality_score(frame: np.ndarray) -> float:
    """
    프레임의 품질 점수를 계산합니다.
    기준: 블러, 클리핑, 밝기 균형, 대비, 색상 풍부도
    Returns: 품질 점수 (높을수록 좋음)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    score = 1.0  # 기본 점수
    
    # 1. 블러 체크 (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 20:  # 심한 블러
        score *= 0.3
    elif lap_var < 50:  # 약간 블러
        score *= 0.7
    elif lap_var > 200:  # 매우 선명
        score *= 1.2
    
    # 2. 클리핑 체크 (너무 어둡거나 밝은 픽셀 비율)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum() + 1e-6
    clip_rate = (hist[:2].sum() + hist[-2:].sum()) / total  # 0-1, 254-255
    if clip_rate > 0.15:  # 15% 이상 클리핑
        score *= 0.5
    elif clip_rate > 0.10:
        score *= 0.8
    
    # 3. 밝기 균형 (중간 밝기 100-180 범위가 좋음)
    mean_brightness = np.mean(gray)
    if 100 <= mean_brightness <= 180:
        score *= 1.1  # 좋은 밝기
    elif mean_brightness < 50 or mean_brightness > 230:
        score *= 0.6  # 너무 어둡거나 밝음
    
    # 4. 대비 (표준 편차가 높을수록 대비가 좋음)
    std_dev = np.std(gray)
    if std_dev > 30:
        score *= 1.1  # 좋은 대비
    elif std_dev < 15:
        score *= 0.8  # 낮은 대비
    
    # 5. 색상 풍부도 (BGR 채널 간 차이)
    b, g, r = cv2.split(frame)
    color_variance = np.mean([
        np.std(b), np.std(g), np.std(r)
    ])
    if color_variance > 25:
        score *= 1.05  # 색상이 풍부함
    
    return score


def extract_blurry_frames(video_path: str, fps: float, blur_threshold: float = 50.0) -> List[Tuple[int, float, np.ndarray, float]]:
    """
    블러가 심한(선명하지 않은) 프레임만 추출
    Args:
        video_path: 영상 파일 경로
        fps: 프레임레이트
        blur_threshold: 블러 임계값 (이 값보다 낮으면 블러가 심한 것으로 간주)
    Returns: List of (frame_index, time_sec, frame_array, blur_score)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    blurry_frames = []
    frame_interval = int(fps)  # 초당 1프레임
    frame_idx = 0
    
    print(f"  블러 임계값: {blur_threshold:.2f}")
    print(f"  블러 점수가 {blur_threshold:.2f} 미만인 프레임만 추출합니다")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 초당 1프레임만 체크
        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / fps
            blur_score = calculate_blur_score(frame)
            
            # 블러가 심한 프레임만 저장
            if blur_score < blur_threshold:
                blurry_frames.append((frame_idx, time_sec, frame.copy(), blur_score))
        
        frame_idx += 1
    
    cap.release()
    return blurry_frames


def quick_quality_check(frame: np.ndarray) -> bool:
    """
    초기 필터링을 위한 빠른 품질 체크
    심각하게 나쁜 프레임은 제외 (너무 느린 계산은 피함)
    Returns: True if frame passes basic quality criteria
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 심한 블러 체크 (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 10:  # 너무 심한 블러는 제외
        return False
    
    # 2. 심한 클리핑 체크
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum() + 1e-6
    clip_rate = (hist[:2].sum() + hist[-2:].sum()) / total
    if clip_rate > 0.25:  # 25% 이상 클리핑은 제외
        return False
    
    # 3. 너무 어둡거나 밝은 프레임 제외
    mean_brightness = np.mean(gray)
    if mean_brightness < 30 or mean_brightness > 240:  # 너무 극단적
        return False
    
    return True


def extract_frames_by_duration(video_path: str, fps: float, 
                                apply_quality_filter: bool = True,
                                frames_per_second: int = 1) -> List[Tuple[int, float, np.ndarray]]:
    """
    영상에서 프레임 추출
    기본 품질 필터링 적용 옵션
    
    Args:
        video_path: 영상 파일 경로
        fps: 프레임레이트
        apply_quality_filter: True면 기본 품질 기준으로 필터링
        frames_per_second: 초당 추출할 프레임 수 (기본값: 1, 전체 프레임 추출하려면 fps 값 사용)
    
    Returns: List of (frame_index, time_sec, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    frames = []
    
    if frames_per_second >= fps:
        # 전체 프레임 추출
        frame_interval = 1
        print(f"  전체 프레임 추출 모드 (FPS: {fps:.2f})")
    else:
        # 초당 N프레임 추출
        frame_interval = int(fps / frames_per_second)
        print(f"  초당 {frames_per_second}프레임 추출 모드 (간격: {frame_interval}프레임)")
    
    total_checked = 0
    filtered_out = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 초당 1프레임만 추출
        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / fps
            total_checked += 1
            
            # 기본 품질 체크 (선택적)
            if apply_quality_filter:
                if quick_quality_check(frame):
                    frames.append((frame_idx, time_sec, frame.copy()))
                else:
                    filtered_out += 1
            else:
                frames.append((frame_idx, time_sec, frame.copy()))
        
        frame_idx += 1
    
    cap.release()
    
    if apply_quality_filter:
        print(f"  체크한 프레임: {total_checked}개, 통과: {len(frames)}개, 필터링됨: {filtered_out}개")
    
    return frames


def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """
    PySceneDetect를 사용하여 장면 전환 감지
    Returns: List of (start_time, end_time) tuples in seconds
    """
    print("\n[1/4] 장면 전환 감지 중...")
    
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    
    video_manager.set_duration()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    
    if not scene_list:
        print("  ⚠️  장면 전환을 감지하지 못했습니다. 전체를 하나의 장면으로 처리합니다.")
        return []
    
    # 시간(초)로 변환
    scenes = []
    for start_scene, end_scene in scene_list:
        start_sec = start_scene.get_seconds()
        end_sec = end_scene.get_seconds()
        scenes.append((start_sec, end_sec))
    
    print(f"  ✅ {len(scenes)}개의 장면을 감지했습니다")
    for i, (start, end) in enumerate(scenes, 1):
        print(f"     장면 {i}: {start:.2f}초 ~ {end:.2f}초 ({end-start:.2f}초)")
    
    return scenes


def calculate_similarity(frame1: np.ndarray, frame2: np.ndarray, resize_to: Tuple[int, int] = (64, 64)) -> float:
    """
    두 프레임 간의 유사도를 계산합니다.
    히스토그램 상관관계를 사용 (0.0 ~ 1.0, 높을수록 유사)
    성능 최적화를 위해 작은 해상도로 리사이즈 후 비교
    """
    # 성능 최적화: 작은 해상도로 리사이즈
    if resize_to:
        frame1_small = cv2.resize(frame1, resize_to)
        frame2_small = cv2.resize(frame2, resize_to)
    else:
        frame1_small = frame1
        frame2_small = frame2
    
    # 히스토그램 계산 (BGR 각 채널)
    hist1 = []
    hist2 = []
    for i in range(3):  # B, G, R 채널
        h1 = cv2.calcHist([frame1_small], [i], None, [256], [0, 256])
        h2 = cv2.calcHist([frame2_small], [i], None, [256], [0, 256])
        hist1.append(h1.flatten())
        hist2.append(h2.flatten())
    
    # 각 채널의 상관관계 계산 후 평균
    similarities = []
    for h1, h2 in zip(hist1, hist2):
        # 정규화
        h1_norm = h1 / (h1.sum() + 1e-6)
        h2_norm = h2 / (h2.sum() + 1e-6)
        # 상관관계 (Correlation coefficient)
        corr = np.corrcoef(h1_norm, h2_norm)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        similarities.append((corr + 1.0) / 2.0)  # -1~1을 0~1로 변환
    
    return np.mean(similarities)


def group_frames_by_scenes(frames: List[Tuple[int, float, np.ndarray]], 
                           scenes: List[Tuple[float, float]]) -> Dict[int, List[Tuple[int, float, np.ndarray]]]:
    """
    프레임을 장면별로 그룹화
    Returns: {scene_idx: [(frame_idx, time, frame), ...]}
    """
    if not scenes:
        # 장면이 없으면 전체를 하나의 그룹으로
        return {0: frames}
    
    grouped = {}
    
    for scene_idx, (scene_start, scene_end) in enumerate(scenes):
        scene_frames = []
        for frame_idx, time_sec, frame in frames:
            if scene_start <= time_sec < scene_end:
                scene_frames.append((frame_idx, time_sec, frame))
        grouped[scene_idx] = scene_frames
    
    # 마지막 프레임들 처리 (마지막 장면에 포함)
    if scenes:
        last_scene_start = scenes[-1][0]
        for frame_idx, time_sec, frame in frames:
            if time_sec >= last_scene_start:
                if len(grouped) > 0:
                    last_scene_idx = max(grouped.keys())
                    if (frame_idx, time_sec, frame) not in grouped[last_scene_idx]:
                        grouped[last_scene_idx].append((frame_idx, time_sec, frame))
    
    return grouped


def group_frames_by_similarity_temporal(all_frames: List[Tuple[int, float, np.ndarray]], 
                                         similarity_threshold: float = 0.85,
                                         time_window: float = 5.0) -> List[List[Tuple[int, float, np.ndarray]]]:
    """
    시간 순서를 고려하여 유사도 기준으로 프레임들을 그룹화
    장면 전환이 없는 부분은 비슷하므로 같은 그룹에 묶음
    Args:
        all_frames: 시간순으로 정렬된 프레임 리스트
        similarity_threshold: 유사도 임계값
        time_window: 같은 그룹에 속할 수 있는 최대 시간 간격 (초)
    Returns: List of groups, each group contains similar frames
    """
    print(f"\n[3/4] 시간 순서 기반 유사도 그룹화 중... (threshold: {similarity_threshold:.3f}, 시간 윈도우: {time_window:.1f}초)")
    
    groups = []
    
    # 시간순으로 정렬되어 있다고 가정
    for frame_idx, time_sec, frame in all_frames:
        assigned = False
        
        # 시간 순서상 최근 그룹들부터 확인 (시간적으로 가까운 그룹 우선)
        # 마지막 그룹부터 역순으로 검사
        for group in reversed(groups):
            # 그룹의 마지막 프레임 시간 확인 (시간 순서상 가장 가까운 것)
            if not group:
                continue
            last_frame_time = group[-1][1]  # time_sec
            
            # 시간 윈도우를 벗어나면 비교하지 않음
            if abs(time_sec - last_frame_time) > time_window:
                continue
            
            # 그룹의 대표 프레임(마지막 프레임 또는 평균)과 비교
            # 성능을 위해 그룹의 마지막 몇 개 프레임과만 비교
            representative_frames = group[-3:] if len(group) >= 3 else group
            similarities = []
            for existing_idx, existing_time, existing_frame in representative_frames:
                sim = calculate_similarity(frame, existing_frame)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # 평균 유사도가 threshold 이상이면 같은 그룹에 추가
            if avg_similarity >= similarity_threshold:
                group.append((frame_idx, time_sec, frame))
                assigned = True
                break
        
        # 기존 그룹에 속하지 않으면 새 그룹 생성
        if not assigned:
            groups.append([(frame_idx, time_sec, frame)])
    
    print(f"  ✅ {len(groups)}개의 그룹 생성됨")
    for i, group in enumerate(groups, 1):
        if group:
            start_time = group[0][1]
            end_time = group[-1][1]
            print(f"     그룹 {i}: {len(group)}개 프레임 ({start_time:.2f}초 ~ {end_time:.2f}초)")
    
    return groups


def save_frames_by_groups(groups: List[List[Tuple[int, float, np.ndarray]]], 
                          output_dir: str, video_id: str,
                          keep_top_n: int = 2):
    """
    각 그룹을 별도 폴더에 저장
    각 그룹에서 품질 점수가 높은 상위 N개만 저장
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[4/4] 그룹별 프레임 저장 중... (각 그룹당 상위 {keep_top_n}개만 유지)")
    print(f"  저장 경로: {output_path}")
    
    saved_groups = {}
    total_deleted = 0
    
    for group_idx, group in enumerate(groups, 1):
        # 각 그룹별 폴더 생성
        group_dir = output_path / f"group_{group_idx:02d}"
        group_dir.mkdir(exist_ok=True)
        
        # 품질 점수 계산 및 정렬
        scored_frames = []
        for frame_idx, time_sec, frame in group:
            quality_score = calculate_quality_score(frame)
            scored_frames.append((frame_idx, time_sec, frame, quality_score))
        
        # 품질 점수순으로 정렬 (높은 점수부터)
        scored_frames.sort(key=lambda x: x[3], reverse=True)
        
        # 상위 N개만 선택
        top_frames = scored_frames[:keep_top_n]
        
        # 선택된 프레임들 저장
        saved_files = []
        for idx, (frame_idx, time_sec, frame, quality_score) in enumerate(top_frames, 1):
            filename = group_dir / f"frame_{idx:02d}_t{time_sec:.2f}s_score{quality_score:.3f}_{video_id}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_files.append(str(filename))
        
        # 원래 그룹에서 제외된 프레임 수
        deleted_count = len(group) - len(saved_files)
        total_deleted += deleted_count
        
        saved_groups[group_idx] = {
            'folder': str(group_dir),
            'frames': saved_files,
            'count': len(saved_files),
            'original_count': len(group)
        }
        
        if deleted_count > 0:
            print(f"  그룹 {group_idx}: {len(saved_files)}개 프레임 저장 (원래 {len(group)}개 중 상위 {keep_top_n}개 선택)")
        else:
            print(f"  그룹 {group_idx}: {len(saved_files)}개 프레임 저장")
    
    total_frames = sum(g['count'] for g in saved_groups.values())
    print(f"\n✅ 총 {len(groups)}개 그룹, {total_frames}개 프레임 저장 완료")
    print(f"  제거된 프레임: {total_deleted}개 (각 그룹당 상위 {keep_top_n}개만 유지)")
    return saved_groups


def save_blurry_frames(blurry_frames: List[Tuple[int, float, np.ndarray, float]], 
                       output_dir: str, video_id: str):
    """블러가 심한 프레임들을 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n블러 프레임 저장 중...")
    print(f"  저장 경로: {output_path}")
    
    saved_files = []
    for idx, (frame_idx, time_sec, frame, blur_score) in enumerate(blurry_frames, 1):
        filename = output_path / f"blurry_{idx:03d}_t{time_sec:.2f}s_blur{blur_score:.2f}_{video_id}.jpg"
        cv2.imwrite(str(filename), frame)
        saved_files.append(str(filename))
    
    print(f"\n✅ 총 {len(saved_files)}개의 블러 프레임 저장 완료")
    return saved_files


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Extract, filter (blur), detect scenes, and group frames.")
    parser.add_argument("--video_path", type=str, default="/Users/jeff/python/9_total_work/downloads/index0/video_0_iZXK1R_nqgc.mp4")
    parser.add_argument("--output_dir", type=str, default="/Users/jeff/python/9_total_work/downloads/index0/extracted_frames")
    parser.add_argument("--keep_top_n", type=int, default=2)
    parser.add_argument("--blur_threshold", type=float, default=150.0)
    args = parser.parse_args()

    # 입력 파일 및 출력 경로 설정
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("프레임 추출: 시간 순서 기반 유사도 그룹화")
    print("="*80)
    print(f"영상: {video_path.name}")
    print(f"출력: {output_dir}")
    
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        return
    
    try:
        # 영상 정보 확인
        info = get_video_info(str(video_path))
        print(f"\n영상 정보:")
        print(f"  길이: {info['duration']:.2f}초")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  해상도: {info['width']}x{info['height']}")
        
        # 1. 모든 프레임 추출 (전체 프레임 또는 더 많은 샘플링)
        print(f"\n[1/4] 프레임 추출 중...")
        # 전체 프레임 추출하려면 frames_per_second에 fps 값을 전달
        # 또는 초당 여러 프레임 추출하려면 원하는 값을 전달 (예: 5 = 초당 5프레임)
        all_frames = extract_frames_by_duration(str(video_path), info['fps'], 
                                                 apply_quality_filter=False,
                                                 frames_per_second=int(info['fps']))  # 전체 프레임 추출
        print(f"  ✅ {len(all_frames)}개의 프레임 추출됨")
        
        # 2. 블러 프레임 필터링 및 별도 저장
        print(f"\n[2/4] 블러 프레임 필터링 중...")
        blur_threshold = args.blur_threshold  # 더 엄격한 블러 감지 (기존: 50.0)
        video_id = video_path.stem.split('_')[-1] if '_' in video_path.stem else video_path.stem
        
        clear_frames = []  # 블러가 아닌 프레임들
        blurry_frame_data = []  # 블러 프레임 데이터
        
        for frame_idx, time_sec, frame in all_frames:
            blur_score = calculate_blur_score(frame)
            if blur_score < blur_threshold:
                # 블러 프레임 저장용 데이터
                blurry_frame_data.append((frame_idx, time_sec, frame, blur_score))
            else:
                # 선명한 프레임만 그룹화에 사용
                clear_frames.append((frame_idx, time_sec, frame))
        
        print(f"  블러 프레임: {len(blurry_frame_data)}개")
        print(f"  선명한 프레임: {len(clear_frames)}개")
        
        # 블러 프레임 별도 저장
        if blurry_frame_data:
            blurry_output_dir = output_dir / "blurry_frames"
            save_blurry_frames(blurry_frame_data, str(blurry_output_dir), video_id)
        
        # 3. 선명한 프레임들을 시간 순서 기반으로 그룹화 (목표: 28~32개 그룹)
        if clear_frames:
            # 시간순 정렬 확인
            clear_frames.sort(key=lambda x: x[1])  # time_sec로 정렬
            
            # 목표 그룹 수: 영상 길이(초)만큼, 오차 범위 ±5초
            target_group_count = int(round(info['duration']))  # 30초면 30개
            tolerance = 5  # ±5초 허용
            min_groups = max(1, target_group_count - tolerance)  # 최소 1개는 보장
            max_groups = target_group_count + tolerance
            
            print(f"\n[3/4] 유사도 threshold 최적화 중...")
            print(f"  목표: {target_group_count}개 그룹 (영상 길이: {info['duration']:.2f}초)")
            print(f"  허용 범위: {min_groups}~{max_groups}개 (±{tolerance}초 오차)")
            
            # 이진 탐색으로 최적 threshold 찾기
            # threshold가 높을수록 더 엄격하게 구분 → 그룹 수 증가
            # threshold가 낮을수록 느슨하게 묶음 → 그룹 수 감소
            low_thresh = 0.70
            high_thresh = 0.99
            best_threshold = 0.85
            best_groups = None
            iteration = 0
            max_iterations = 20
            
            while iteration < max_iterations:
                mid_thresh = (low_thresh + high_thresh) / 2.0
                test_groups = group_frames_by_similarity_temporal(clear_frames, similarity_threshold=mid_thresh, time_window=5.0)
                num_groups = len(test_groups)
                
                if min_groups <= num_groups <= max_groups:
                    # 목표 범위 내
                    best_threshold = mid_thresh
                    best_groups = test_groups
                    print(f"  ✅ 찾음! threshold={mid_thresh:.4f}, 그룹={num_groups}개")
                    break
                elif num_groups < min_groups:
                    # 그룹이 너무 적음 -> threshold 높여서 더 쪼개기 (더 엄격하게)
                    low_thresh = mid_thresh
                    if best_groups is None or abs(len(best_groups) - target_group_count) > abs(num_groups - target_group_count):
                        best_threshold = mid_thresh
                        best_groups = test_groups
                else:
                    # 그룹이 너무 많음 -> threshold 낮춰서 덜 쪼개기 (더 느슨하게)
                    high_thresh = mid_thresh
                    if best_groups is None or abs(len(best_groups) - target_group_count) > abs(num_groups - target_group_count):
                        best_threshold = mid_thresh
                        best_groups = test_groups
                
                iteration += 1
                if iteration % 3 == 0:
                    print(f"    반복 {iteration}: threshold={mid_thresh:.4f}, 그룹={num_groups}개 (목표: {target_group_count}개)")
            
            # 최종 그룹화
            if best_groups is None:
                # 탐색 실패 시 기본값으로
                print(f"  ⚠️  목표 범위를 찾지 못해 최적값 사용: threshold={best_threshold:.4f}")
                best_groups = group_frames_by_similarity_temporal(clear_frames, similarity_threshold=best_threshold, time_window=5.0)
            
            groups = best_groups
            
            # 4. 그룹별로 프레임 저장 (각 그룹당 상위 2개만)
            saved_groups = save_frames_by_groups(groups, str(output_dir), video_id, keep_top_n=args.keep_top_n)
            
            print("\n" + "="*80)
            print("✅ 완료!")
            print(f"  생성된 그룹: {len(groups)}개")
            print(f"  선명한 프레임: {len(clear_frames)}개")
            print(f"  블러 프레임: {len(blurry_frame_data)}개")
            print(f"  저장 위치: {output_dir}")
            print("="*80)
        else:
            print("\n⚠️  선명한 프레임이 없습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
