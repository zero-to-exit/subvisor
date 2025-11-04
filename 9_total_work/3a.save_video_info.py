"""
영상 정보를 JSON 파일로 저장
"""
import json
import csv
import cv2
from pathlib import Path


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
        'fps': float(fps),
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration_seconds': float(duration),
        'duration_formatted': f"{int(duration // 60)}:{int(duration % 60):02d}"
    }


def get_csv_info(video_id: str) -> dict:
    """CSV 파일에서 영상 메타데이터 가져오기"""
    csv_file = Path("/Users/jeff/python/9_total_work/youtube_ads_1000.csv")
    
    if not csv_file.exists():
        return {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                return {
                    'index': int(row['index']),
                    'video_id': row['video_id'],
                    'url': row['url'],
                    'title': row['title'],
                    'duration': row['duration'],
                    'duration_seconds': int(row.get('duration_seconds', 0)),
                    'views': int(row.get('views', 0)) if row.get('views') else 0,
                    'upload_date': row.get('upload_date', ''),
                    'channel': row.get('channel', '')
                }
    
    return {}


def save_video_info_json(video_path: str, output_dir: str):
    """영상 정보를 JSON으로 저장"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Video ID 추출 (파일명에서)
    # 파일명 형식: video_0_iZXK1R_nqgc.mp4 -> video_id는 iZXK1R_nqgc
    stem = video_path.stem
    if '_' in stem:
        # video_0_iZXK1R_nqgc -> ['video', '0', 'iZXK1R', 'nqgc']
        parts = stem.split('_')
        # index 이후부터 모두 합치기 (video_id가 _를 포함할 수 있음)
        if len(parts) >= 3:
            video_id = '_'.join(parts[2:])  # 인덱스 이후 모두
        else:
            video_id = stem
    else:
        video_id = stem
    
    # 영상 파일 정보
    video_info = get_video_info(str(video_path))
    
    # CSV에서 메타데이터 가져오기
    csv_info = get_csv_info(video_id)
    
    # 추출된 프레임 목록 가져오기
    extracted_frames = []
    frames_dir = Path(output_dir)
    if frames_dir.exists():
        for frame_file in sorted(frames_dir.glob('scene_*.jpg')):
            # 파일명 파싱: scene_01_t0.00s_score1.386_nqgc.jpg
            filename = frame_file.name
            parts = filename.replace('.jpg', '').split('_')
            
            frame_info = {
                'filename': filename,
                'file_path': str(frame_file),
                'file_size_kb': round(frame_file.stat().st_size / 1024, 2)
            }
            
            # 시간과 점수 추출
            for part in parts:
                if part.startswith('t') and part.endswith('s'):
                    try:
                        frame_info['time_seconds'] = float(part[1:-1])
                    except:
                        pass
                elif part.startswith('score'):
                    try:
                        frame_info['quality_score'] = float(part[5:])
                    except:
                        pass
            
            extracted_frames.append(frame_info)
    
    # 통합 정보
    combined_info = {
        'video_file': {
            'filename': video_path.name,
            'file_path': str(video_path),
            'file_size_mb': round(video_path.stat().st_size / (1024 * 1024), 2)
        },
        'video_metadata': csv_info,
        'video_properties': {
            'resolution': f"{video_info['width']}x{video_info['height']}",
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': video_info['fps'],
            'total_frames': video_info['frame_count'],
            'duration_seconds': video_info['duration_seconds'],
            'duration_formatted': video_info['duration_formatted']
        },
        'extraction_info': {
            'extracted_frames_dir': str(output_dir),
            'extraction_method': 'scene_detection_with_quality_scoring',
            'frames_per_second_sampled': 1.0,
            'total_scenes_detected': len(extracted_frames),
            'extracted_frames': extracted_frames
        }
    }
    
    # JSON 파일로 저장
    json_file = output_dir / 'video_info.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 영상 정보 저장 완료: {json_file}")
    return combined_info


def main():
    """메인 함수"""
    video_path = "/Users/jeff/python/9_total_work/downloads/index0/video_0_iZXK1R_nqgc.mp4"
    output_dir = "/Users/jeff/python/9_total_work/downloads/index0/extracted_frames"
    
    print("="*80)
    print("영상 정보 JSON 저장")
    print("="*80)
    
    info = save_video_info_json(video_path, output_dir)
    
    print("\n저장된 정보 요약:")
    print(f"  제목: {info['video_metadata'].get('title', 'N/A')}")
    print(f"  영상 길이: {info['video_properties']['duration_formatted']} ({info['video_properties']['duration_seconds']:.2f}초)")
    print(f"  총 프레임 수: {info['video_properties']['total_frames']:,}개")
    print(f"  FPS: {info['video_properties']['fps']:.2f}")
    print(f"  해상도: {info['video_properties']['resolution']}")
    print(f"  파일 크기: {info['video_file']['file_size_mb']} MB")
    print(f"  조회수: {info['video_metadata'].get('views', 0):,}")
    print(f"  채널: {info['video_metadata'].get('channel', 'N/A')}")


if __name__ == "__main__":
    main()

