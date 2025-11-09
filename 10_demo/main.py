'''
All the procedures for the demo

1. Download video from url
2. Extract frames from video(1000-1300)
3. Dividde the frames into groups based on the similarity
(hard thing in here afrter the division)
(30)

4. select the few frames on each group(Algorithm 1)
(Selection algoirthm will be implemented)

5. save the selected frames(20~30 frames)
6. Now we are going to score these frames from LLM(Algorithm 2)
(Prompting tech should be used for the evaluation.)

7. After the scoring, we are going to select the best frames(5~10) in sequence.
8. Export them in time series.
'''

import os
import pandas as pd
import subprocess
from pathlib import Path


def download_video(url: str, output_file: Path):
    """Use yt-dlp to download the video"""
    print(f"\nDownloading video from {url} to {output_file}")
    print(f"  URL: {url}")
    print(f"  Video ID: {output_file.name}")        
    try:
        # yt-dlp 명령어
        # -f: highest quality (single file with video and audio)
        # -o: output file name
        # --no-playlist: no playlist, only single video
        # --merge-output-format: merge output format to mp4
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            '--merge-output-format', 'mp4',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
            url
        ]
        
        print(f"  Executing command...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # check if the file is actually saved
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ Download completed!")
                print(f"  File size: {file_size:.2f} MB")
                return output_file
            else:
                print(f"  ⚠️  Download is successful but the file is not found.")
                return None
        else:
            print(f"  ❌ Download failed")
            print(f"  Error: {result.stderr[:500]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout (5 minutes exceeded)")
        return None
    except Exception as e:
        print(f"  ❌ Error occurred: {str(e)}")
        return None

if __name__ == "__main__":

    #read the csv file extrac the url from the csv file.
    csv_file = os.path.join(os.getcwd(), "youtube_ads_1000.csv")
    df = pd.read_csv(csv_file)

    #deom w/ index 1 11/7
    id = 3
    url = df.iloc[id]['url']
    title = df.iloc[id]['title']

    print("="*40)
    print("index: ", id)
    print("title: ", title)
    print("url: ", url)
    print("="*40)

    #step0. download the video to the working directory.
    working_dir = Path(os.getcwd()) / f"demo_1107/v_{id}"
    video_path = working_dir / f"video_{str(id)}.mp4"
    download_video(url, video_path)

    #step1. Algorithm1(all frames -> 30 frames)
    from models.algorithm1 import Algorithm1
    alg1 = Algorithm1(working_dir, video_path)
    alg1.extract_all_frames(ratio=0.5)
    alg1.divide_frames_into_groups(group_count = 30)
    alg1.score_frames(target_count=30, per_group_selected=1, yolo_weights="yolov8n.pt")
    alg1.save_selected_frames()

    #step2. Algorithm2(30framse -> 5 frames)
    from models.algorithm2_method3 import Algorithm2_method3
    alg2 = Algorithm2_method3(working_dir)
    alg2.gemini_api_call(gemini_model="gemini-2.5-flash")
    dict_response = alg2.extract_dict_from_response()
    alg2.save_selected_frames(dict_response)

