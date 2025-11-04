#!/usr/bin/env python3
"""
영상에서 criteria.txt 기준에 따라 상위 20장의 핵심 프레임을 추출합니다.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from quality_model import (
    CriteriaWeights,
    FeatureExtractor,
    WeightedScorer,
    select_diverse_top_k,
)


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


def select_frames_with_coverage(
    frame_indices: List[int],
    scores: List[float],
    times_sec: List[float],
    duration: float,
    k: int = 20,
) -> List[int]:
    """
    타임라인 커버리지를 고려하여 프레임을 선별합니다.
    도입/전개/클라이맥스/아웃트로 구간에 고른 분배를 보장합니다.
    """
    if len(frame_indices) <= k:
        return frame_indices
    
    # 구간별 할당 (도입/전개/절정/결말)
    segments = [
        (0.0, 0.25, k // 4),      # 도입
        (0.25, 0.55, k // 3),      # 전개
        (0.55, 0.85, k // 3),      # 절정
        (0.85, 1.0, k - (k // 4 + 2 * (k // 3))),  # 결말 (나머지)
    ]
    
    selected: List[int] = []
    indices_array = np.array(frame_indices)
    scores_array = np.array(scores)
    times_array = np.array(times_sec)
    
    for seg_start, seg_end, count in segments:
        # 해당 구간의 프레임 필터링
        mask = (times_array >= seg_start * duration) & (times_array < seg_end * duration)
        if mask.sum() == 0:
            continue
        
        seg_indices = indices_array[mask].tolist()
        seg_scores = scores_array[mask].tolist()
        
        # 점수 순으로 정렬
        sorted_pairs = sorted(zip(seg_indices, seg_scores), key=lambda x: x[1], reverse=True)
        # 상위 count개 선택
        for idx, _ in sorted_pairs[:count]:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= k:
                    break
        
        if len(selected) >= k:
            break
    
    # 부족하면 전체에서 점수 높은 순으로 추가
    if len(selected) < k:
        all_sorted = sorted(zip(frame_indices, scores), key=lambda x: x[1], reverse=True)
        for idx, _ in all_sorted:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= k:
                    break
    
    return selected[:k]


def extract_top_frames(
    video_path: str,
    num_frames: int = 20,
    output_dir: str = "top_frames",
    sample_interval: int = 5,  # N프레임마다 샘플링 (성능 최적화)
) -> List[Tuple[int, float, float]]:
    """
    영상에서 criteria.txt 기준에 따라 상위 20장의 프레임을 추출합니다.
    
    Returns:
        List of (frame_index, score, time_sec) tuples
    """
    print("=" * 80)
    print("프레임 품질 평가 및 선별 시작")
    print("=" * 80)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 영상 정보 확인
    info = get_video_info(video_path)
    print(f"\n영상 정보:")
    print(f"  - 경로: {video_path}")
    print(f"  - 길이: {info['duration']:.2f}초")
    print(f"  - FPS: {info['fps']:.2f}")
    print(f"  - 총 프레임 수: {info['frame_count']}")
    print(f"  - 해상도: {info['width']}x{info['height']}")
    print(f"  - 샘플링 간격: {sample_interval}프레임마다")
    
    # 품질 평가 모델 초기화
    weights = CriteriaWeights()
    extractor = FeatureExtractor()
    scorer = WeightedScorer(weights)
    
    # 프레임 읽기 및 평가
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    print(f"\n프레임 분석 중...")
    frame_indices: List[int] = []
    scores: List[float] = []
    times_sec: List[float] = []
    
    # 프레임을 미리 읽어서 버퍼 구성 (성능 최적화)
    frame_buffer: List[Tuple[int, np.ndarray]] = []
    total_frames = info['frame_count']
    
    # 샘플링할 프레임만 미리 읽기
    sampled_indices = list(range(0, total_frames, sample_interval))
    print(f"  샘플링 대상: {len(sampled_indices)}개 프레임")
    
    for sample_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame = cap.read()
        if ret:
            frame_buffer.append((sample_idx, frame.copy()))
        
        if sample_idx % (sample_interval * 50) == 0:
            progress = len(frame_buffer) / len(sampled_indices) * 100
            print(f"  프레임 읽기: {len(frame_buffer)}/{len(sampled_indices)} ({progress:.1f}%)")
    
    print(f"  프레임 평가 시작: {len(frame_buffer)}개")
    
    # 버퍼된 프레임들 평가
    for i, (frame_idx, frame) in enumerate(frame_buffer):
        # 이전/다음 프레임 가져오기
        prev_bgr_local = None
        next_bgr = None
        
        if i > 0:
            prev_bgr_local = frame_buffer[i-1][1]
        if i < len(frame_buffer) - 1:
            next_bgr = frame_buffer[i+1][1]
        
        # 특징 추출 및 점수 계산
        features = extractor.compute_features(
            frame,
            prev_bgr=prev_bgr_local,
            next_bgr=next_bgr,
        )
        overall_score, per_key_scores = scorer.score_features(features)
        
        # 벌점 적용 (무조건 배제 조건 확인)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 심한 블러 체크
        if lap_var < 20:
            overall_score *= 0.3
        
        # 클리핑 체크 (간단화)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum() + 1e-6
        clip_rate = (hist[:2].sum() + hist[-2:].sum()) / total
        if clip_rate > 0.15:
            overall_score *= 0.5
        
        # 점수 저장
        frame_indices.append(frame_idx)
        scores.append(overall_score)
        times_sec.append(frame_idx / info['fps'])
        
        if (i + 1) % 50 == 0:
            print(f"  평가 진행: {i+1}/{len(frame_buffer)} ({((i+1)/len(frame_buffer)*100):.1f}%)")
    
    # 프레임 버퍼를 다시 사용하기 위해 저장
    frames_bgr = [frame for _, frame in frame_buffer]
    
    cap.release()
    
    print(f"\n총 {len(scores)}개의 프레임 후보를 평가했습니다.")
    
    if len(scores) == 0:
        raise RuntimeError("평가된 프레임이 없습니다.")
    
    # 타임라인 커버리지를 고려한 선별
    print(f"\n타임라인 커버리지를 고려하여 {num_frames}개 선별 중...")
    selected_by_coverage = select_frames_with_coverage(
        frame_indices, scores, times_sec, info['duration'], k=num_frames
    )
    
    # 다양성을 고려한 최종 선별 (pHash 기반)
    print(f"다양성을 고려한 최종 선별 중...")
    # 인덱스 매핑 생성 (더 빠른 조회)
    idx_to_pos = {idx: i for i, idx in enumerate(frame_indices)}
    selected_frames_bgr = [frames_bgr[idx_to_pos[idx]] for idx in selected_by_coverage]
    selected_scores = [scores[idx_to_pos[idx]] for idx in selected_by_coverage]
    
    diverse_indices = select_diverse_top_k(
        selected_frames_bgr,
        selected_scores,
        k=num_frames,
        min_hamming=8,  # 다양성 임계값
    )
    
    final_selected = [selected_by_coverage[i] for i in diverse_indices]
    final_scores = [selected_scores[i] for i in diverse_indices]
    final_times = [times_sec[frame_indices.index(idx)] for idx in final_selected]
    
    # 프레임 저장
    print(f"\n선별된 {len(final_selected)}개 프레임 저장 중...")
    cap = cv2.VideoCapture(video_path)
    
    results: List[Tuple[int, float, float]] = []
    
    for rank, (frame_idx, score, time_sec) in enumerate(zip(final_selected, final_scores, final_times), 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = f"frame_{rank:02d}_t{time_sec:.2f}s_score{score:.3f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            results.append((frame_idx, score, time_sec))
            print(f"  [{rank:2d}/{num_frames}] 프레임 {frame_idx} 저장: {filename} (점수: {score:.3f}, 시간: {time_sec:.2f}초)")
    
    cap.release()
    
    print(f"\n✅ 총 {len(results)}개의 프레임이 '{output_dir}' 디렉토리에 저장되었습니다.")
    print(f"점수 범위: {min(final_scores):.2f} ~ {max(final_scores):.2f}")
    
    return results


if __name__ == "__main__":
    video_path = "/Users/jeff/python/6_agent/Fishing.mp4"
    output_dir = "/Users/jeff/python/7_agent2/top_frames"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if not os.path.exists(video_path):
        print(f"오류: 영상 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)
    
    try:
        results = extract_top_frames(
            video_path=video_path,
            num_frames=20,
            output_dir=output_dir,
            sample_interval=15,  # 15프레임마다 샘플링 (성능 최적화)
        )
        
        print("\n" + "=" * 80)
        print("완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

