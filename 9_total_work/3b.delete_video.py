'''
Now cuz of the storage,
프레임 추출이 완료된 후 저장 공간 확보를 위해 영상 파일 삭제
'''
import os
from pathlib import Path


def delete_video(video_path: str):
    """영상 파일을 삭제"""
    video_path = Path(video_path)
    if video_path.exists():
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        video_path.unlink()
        print(f"✅ 영상 파일 삭제 완료: {video_path.name}")
        print(f"   파일 크기: {file_size_mb:.2f} MB")
        return True
    else:
        print(f"⚠️  영상 파일을 찾을 수 없습니다: {video_path}")
        return False


def delete_videos_in_directory(directory: str, pattern: str = "*.mp4"):
    """디렉토리 내의 모든 영상 파일 삭제"""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {directory}")
        return
    
    video_files = list(dir_path.glob(pattern))
    if not video_files:
        print(f"⚠️  삭제할 영상 파일이 없습니다: {directory}")
        return
    
    print(f"\n디렉토리 내 영상 파일 삭제: {directory}")
    print(f"찾은 영상 파일: {len(video_files)}개")
    
    total_size_mb = 0
    deleted_count = 0
    
    for video_file in video_files:
        if video_file.exists():
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            video_file.unlink()
            total_size_mb += file_size_mb
            deleted_count += 1
            print(f"  ✅ 삭제: {video_file.name} ({file_size_mb:.2f} MB)")
    
    print(f"\n✅ 총 {deleted_count}개 파일 삭제 완료")
    print(f"   절약된 공간: {total_size_mb:.2f} MB")


def main():
    """메인 함수"""
    # 테스트: 현재 작업 중인 영상 파일 삭제
    video_path = Path("/Users/jeff/python/9_total_work/downloads/index0/video_0_iZXK1R_nqgc.mp4")
    
    print("="*80)
    print("영상 파일 삭제")
    print("="*80)
    
    if video_path.exists():
        delete_video(str(video_path))
    else:
        print(f"⚠️  영상 파일이 없습니다: {video_path}")
        print("   (이미 삭제되었거나 다른 경로에 있을 수 있습니다)")
    
    # 또는 특정 디렉토리의 모든 영상 파일 삭제
    # download_dir = "/Users/jeff/python/9_total_work/downloads/index0"
    # delete_videos_in_directory(download_dir)


if __name__ == "__main__":
    main()