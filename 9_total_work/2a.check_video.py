"""
다운로드한 영상의 정보를 확인하고 몇 개의 프레임을 추출합니다.
"""
import cv2
import os
from pathlib import Path
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
    
    # 추가 정보
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'fourcc': fourcc_str,
        'file_size_mb': Path(video_path).stat().st_size / (1024 * 1024)
    }


def extract_sample_frames(video_path: str, num_frames: int = 5, output_dir: str = "sample_frames"):
    """영상에서 샘플 프레임을 균등 간격으로 추출합니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    # 영상 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 추출할 프레임 인덱스 계산 (균등 간격)
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
    else:
        step = frame_count / (num_frames + 1)
        frame_indices = [int(i * step) for i in range(1, num_frames + 1)]
    
    extracted_frames = []
    
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # 시간 계산
            time_sec = frame_num / fps
            
            # 파일명: frame_{index}_{time_sec}s_{frame_num}.jpg
            filename = output_dir / f"frame_{idx+1:02d}_t{time_sec:.2f}s_f{frame_num}.jpg"
            cv2.imwrite(str(filename), frame)
            
            extracted_frames.append({
                'frame_index': frame_num,
                'time_sec': time_sec,
                'filename': filename.name,
                'filepath': str(filename)
            })
            
            print(f"  프레임 {idx+1}/{num_frames}: {time_sec:.2f}초 (프레임 #{frame_num}) -> {filename.name}")
    
    cap.release()
    return extracted_frames


def print_video_info(info: dict):
    """영상 정보를 보기 좋게 출력"""
    print("\n" + "="*80)
    print("영상 정보")
    print("="*80)
    print(f"해상도: {info['width']}x{info['height']}")
    print(f"FPS: {info['fps']:.2f}")
    print(f"총 프레임 수: {info['frame_count']:,}개")
    print(f"영상 길이: {info['duration']:.2f}초 ({info['duration']:.1f}분)")
    print(f"코덱: {info['fourcc']}")
    print(f"파일 크기: {info['file_size_mb']:.2f} MB")
    print(f"비트레이트 (추정): {info['file_size_mb'] * 8 / info['duration']:.2f} Mbps")
    print("="*80)


def main():
    """메인 함수"""
    video_path = Path("/Users/jeff/python/9_total_work/downloads/video_0_iZXK1R_nqgc.mp4")
    output_dir = Path("/Users/jeff/python/9_total_work/sample_frames")
    
    print("="*80)
    print("다운로드 영상 확인 및 샘플 프레임 추출")
    print("="*80)
    
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        return
    
    print(f"\n영상 파일: {video_path.name}")
    
    try:
        # 영상 정보 확인
        print("\n[1/2] 영상 정보 확인 중...")
        info = get_video_info(str(video_path))
        print_video_info(info)
        
        # 샘플 프레임 추출
        print("\n[2/2] 샘플 프레임 추출 중...")
        print(f"출력 디렉토리: {output_dir}")
        frames = extract_sample_frames(str(video_path), num_frames=5, output_dir=str(output_dir))
        
        print(f"\n✅ 완료!")
        print(f"  추출된 프레임: {len(frames)}개")
        print(f"  저장 위치: {output_dir}")
        
        # 추출된 프레임 목록
        print(f"\n추출된 프레임 목록:")
        for f in frames:
            print(f"  - {f['filename']} ({f['time_sec']:.2f}초)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

