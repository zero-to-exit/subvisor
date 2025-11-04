"""
yt-dlp를 사용하여 YouTube 검색 결과 수집
yt-dlp는 YouTube 검색 API를 직접 사용하므로 더 안정적입니다.
"""
import subprocess
import json
import csv
import re
from pathlib import Path
from typing import List, Dict
import time

class YouTubeYTDLPCollector:
    """yt-dlp를 사용한 YouTube URL 수집"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collected_videos: Dict[str, Dict] = {}
        self.search_keywords = [
            "광고 1분",
            "advertisement 1 minute",
            "commercial 1 min",
            "cm 60초",
            "tv commercial 60 seconds",
            "ad 1 minute",
            "광고",
            "advertisement",
            "commercial",
            "cm",
            "tv ad",
            "brand advertisement",
            "product commercial",
            "advertising",
            "tv spot",
            "commercial break",
            "ad campaign",
            "promotional video",
            "ad video",
            "marketing video",
        ]
    
    def search_with_ytdlp(self, query: str, max_results: int = 50) -> List[Dict]:
        """yt-dlp로 YouTube 검색"""
        print(f"\n검색어: '{query}' (최대 {max_results}개)")
        
        try:
            # yt-dlp 검색 명령어
            # yt-dlp는 기본적으로 검색 결과를 가져올 수 있습니다
            cmd = [
                'yt-dlp',
                f'ytsearch{max_results}:{query}',
                '--flat-playlist',
                '--dump-json',
                '--no-warnings',
                '--quiet'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"  ⚠️  오류: {result.stderr[:200]}")
                return []
            
            # JSON 결과 파싱 (각 줄이 하나의 JSON 객체)
            videos = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    video_data = json.loads(line)
                    video_id = video_data.get('id', '')
                    
                    if video_id:
                        duration = video_data.get('duration', 0)  # 초 단위
                        duration = int(float(duration)) if duration else 0
                        
                        videos.append({
                            'video_id': video_id,
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'title': video_data.get('title', '')[:200],
                            'duration_seconds': duration,
                            'duration': self.format_duration(duration),
                            'views': video_data.get('view_count', 0),
                            'upload_date': video_data.get('upload_date', ''),
                            'channel': video_data.get('channel', ''),
                        })
                except json.JSONDecodeError:
                    continue
            
            print(f"  ✅ {len(videos)}개 영상 수집됨")
            return videos
            
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  타임아웃")
            return []
        except FileNotFoundError:
            print(f"  ❌ yt-dlp가 설치되지 않았습니다.")
            print(f"     설치: pip install yt-dlp")
            return []
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            return []
    
    def format_duration(self, seconds) -> str:
        """초를 "M:SS" 또는 "H:MM:SS" 형식으로 변환"""
        if not seconds:
            return ''
        
        # float일 수 있으니 int로 변환
        seconds = int(float(seconds))
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def filter_by_duration(self, videos: List[Dict], min_seconds: int = 30, max_seconds: int = 90) -> List[Dict]:
        """1분 ±30초 범위로 필터링"""
        filtered = []
        for video in videos:
            duration_sec = video.get('duration_seconds', 0)
            if min_seconds <= duration_sec <= max_seconds:
                filtered.append(video)
        return filtered
    
    def collect_all(self, target_count: int = 1000):
        """모든 검색어로 수집"""
        print("="*80)
        print("yt-dlp를 사용한 YouTube 광고 영상 수집")
        print("="*80)
        print(f"목표: {target_count}개 영상 (1분 ±30초)")
        print()
        
        for keyword in self.search_keywords:
            if len(self.collected_videos) >= target_count:
                print(f"\n✅ 목표 달성! {len(self.collected_videos)}개 수집됨")
                break
            
            videos = self.search_with_ytdlp(keyword, max_results=200)
            
            # 중복 제거하며 추가
            for video in videos:
                video_id = video['video_id']
                if video_id not in self.collected_videos:
                    self.collected_videos[video_id] = video
            
            # 필터링 적용
            filtered = self.filter_by_duration(list(self.collected_videos.values()))
            print(f"  현재 수집: {len(self.collected_videos)}개, 필터 후: {len(filtered)}개 (30-90초)")
            
            time.sleep(2)  # API 제한 고려
    
    def save_to_csv(self, filename: str = "youtube_ads_1000.csv"):
        """CSV로 저장"""
        csv_path = self.output_dir / filename
        
        # 필터링
        filtered = self.filter_by_duration(list(self.collected_videos.values()))
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index', 'video_id', 'url', 'title', 'duration', 
                'duration_seconds', 'views', 'upload_date', 'channel'
            ])
            
            for idx, video in enumerate(filtered):
                writer.writerow([
                    idx,
                    video.get('video_id', ''),
                    video.get('url', ''),
                    video.get('title', ''),
                    video.get('duration', ''),
                    video.get('duration_seconds', 0),
                    video.get('views', 0),
                    video.get('upload_date', ''),
                    video.get('channel', ''),
                ])
        
        print(f"\n✅ CSV 저장 완료: {csv_path}")
        print(f"   전체: {len(self.collected_videos)}개")
        print(f"   필터 후 (30-90초): {len(filtered)}개")
    
    def save_to_json(self, filename: str = "youtube_ads_1000.json"):
        """JSON으로 저장"""
        json_path = self.output_dir / filename
        filtered = self.filter_by_duration(list(self.collected_videos.values()))
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON 저장 완료: {json_path}")


def main():
    """메인 실행"""
    collector = YouTubeYTDLPCollector(output_dir="/Users/jeff/python/9_total_work")
    collector.collect_all(target_count=1000)
    
    if collector.collected_videos:
        collector.save_to_csv()
        collector.save_to_json()
    else:
        print("\n⚠️  수집된 영상이 없습니다.")


if __name__ == "__main__":
    main()

