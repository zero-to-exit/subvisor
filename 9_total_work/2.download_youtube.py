'''
Now let us download the youtube videos from the csv file.
yt-dlp를 사용하여 YouTube 영상을 다운로드합니다.
'''
import csv
import subprocess
import os
from pathlib import Path


def download_video(url: str, video_id: str, output_dir: str = "downloads", index: int = 0):
    """yt-dlp를 사용하여 영상 다운로드"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 출력 파일명: video_{index}_{video_id}.mp4
    output_file = output_dir / f"video_{index}_{video_id}.mp4"
    
    print(f"\n다운로드 시작:")
    print(f"  URL: {url}")
    print(f"  Video ID: {video_id}")
    print(f"  저장 경로: {output_file}")
    
    try:
        # yt-dlp 명령어
        # -f: 최고 품질 (비디오+오디오 포함된 단일 파일 우선)
        # -o: 출력 파일명
        # --no-playlist: 재생목록이 아닌 단일 영상만
        # --merge-output-format: 병합 시 mp4 형식 사용
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            '--merge-output-format', 'mp4',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
            url
        ]
        
        print(f"  명령어 실행 중...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # 실제 저장된 파일 확인
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ 다운로드 완료!")
                print(f"  파일 크기: {file_size:.2f} MB")
                return True
            else:
                print(f"  ⚠️  다운로드는 성공했지만 파일을 찾을 수 없습니다.")
                return False
        else:
            print(f"  ❌ 다운로드 실패")
            print(f"  오류: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ 타임아웃 (5분 초과)")
        return False
    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        return False


def main():
    """메인 함수 - CSV에서 첫 번째 영상만 다운로드 (실험)"""
    csv_file = Path("/Users/jeff/python/9_total_work/youtube_ads_1000.csv")
    output_dir = Path("/Users/jeff/python/9_total_work/downloads")
    
    print("="*80)
    print("YouTube 영상 다운로드 실험 (첫 1개)")
    print("="*80)
    
    if not csv_file.exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file}")
        return
    
    # CSV 파일 읽기
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ CSV 파일이 비어있습니다.")
        return
    
    # 첫 번째 영상만 다운로드 (index 0)
    first_video = rows[0]
    
    index = int(first_video['index'])
    video_id = first_video['video_id']
    url = first_video['url']
    title = first_video['title']
    
    print(f"\n다운로드할 영상:")
    print(f"  Index: {index}")
    print(f"  제목: {title}")
    print(f"  URL: {url}")
    
    # 다운로드 실행
    success = download_video(url, video_id, output_dir, index)
    
    if success:
        print(f"\n✅ 실험 완료! 첫 번째 영상 다운로드 성공")
        print(f"다음 단계: 전체 영상 다운로드를 위해 스크립트를 수정하세요.")
    else:
        print(f"\n❌ 다운로드 실패. 오류를 확인하세요.")


if __name__ == "__main__":
    main()
