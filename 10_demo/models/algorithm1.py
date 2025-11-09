'''
First let us talk about the Algorithm 1 in detail.

the scene transformation in between two different groups may be important.

let us contain all the 

col_frames = []

Group1
p1,p2,p3...pm1

Group2
p1,p2,p3...pm2

...

Groupk
p1,p2,p3...pmk

(total k groups, n frames)

several critieria to score
just use weight linear function to score.
#divide the group and each group score w/ the 6 different ciriteria
and sum up them.

1. total 3k -1 frames = extracted frames(k is a group #)
2. Detect the person or object from YOLO model is it possible?


'''
from pathlib import Path
import cv2
import numpy as np
import json
from .paths import frame_number_to_path as _frame_number_to_path 

class Algorithm1:
    '''
    Algorithm 1:
    - extract all the frames from the video.
    - divide the frames into groups based on the similarity
    - select the few frames on each group
    - save the selected frames
    '''
    def __init__(self, working_dir: Path, video_path: str):
        
        self.working_dir = working_dir # default: demo_1107/v_1
        self.video_path = video_path # default: demo_1107/v_1/video_0001.mp4
        self.total_frame_dir = self.working_dir / "total_frames"
        self.total_frame_dir.mkdir(parents=True, exist_ok=True)
        
        #json file path
        self.groups_json_path = self.working_dir / "groups.json"
        self.selected_json_path = self.working_dir / "selected_frames.json"
    
    def extract_all_frames(self, ratio = 0.5) -> Path:
        '''
        extract frames from the video with sampling ratio.
        
        used parameters:
        self.video_path: (.mp4 path)
        ratio: sampling ratio (0.0 ~ 1.0). ratio=0.5 means save 50% of frames (every 2nd frame)
        
        output:
        save_dir: folder path which has all the extracted frames
        '''
        if not (0 < ratio <= 1.0):
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
        
        #read the video file.
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        # Calculate frame interval (ratio=0.5 -> interval=2, save every 2nd frame)
        interval = int(1.0 / ratio) if ratio > 0 else 1
        
        # Stream frames: read and save immediately (memory efficient)
        saved_count = 0
        frame_index = 0  # Original video frame index
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save only frames at the interval (e.g., ratio=0.5 saves every 2nd frame)
            if frame_index % interval == 0:
                out_path = _frame_number_to_path(self.total_frame_dir, saved_count)
                # JPEG quality 90 = good balance between quality and file size
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_count += 1
                if saved_count % 100 == 0:  # Print every 100 frames to reduce output
                    print(f"Saved {saved_count} frames (processed {frame_index + 1} frames from video)")
            
            frame_index += 1
        
        cap.release()
        print(f"✅ Extracted {saved_count} frames from {frame_index} total frames (ratio={ratio:.2f})")
        return

    def divide_frames_into_groups(self, scene_threshold: float = 0.5, group_count: int = 30):
        '''
        divide the frames into groups based on the similarity
        group should be continuous frames.

        used parameters:
        scene_threshold: Bhattacharyya distance; larger => different
        group_count: target group count

        output:
        groups: dictionary of group id and list of frame paths
        '''

        def extract_frame_number(fp: Path) -> int:
            """Extract frame number from filename (123.jpg, f123.jpg, or frame_0123.jpg)"""
            name = fp.stem  # filename without extension

            return int(name)

        def compute_histogram(img_bgr):
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist

        groups_list = []
        current_group = []

        prev_hist = None
        # Get all frame paths - prioritize numeric format (번호.jpg)
        frame_paths = []
        
        # First try numeric format (0.jpg, 1.jpg, etc.)
        numeric_jpg = [p for p in self.total_frame_dir.glob("*.jpg") if p.stem.isdigit()]
        numeric_png = [p for p in self.total_frame_dir.glob("*.png") if p.stem.isdigit()]
        
        if numeric_jpg:
            frame_paths = numeric_jpg
        elif numeric_png:
            # Fallback to PNG if JPG not found
            frame_paths = numeric_png
        else:
            # Fallback to other formats
            frame_paths = list(self.total_frame_dir.glob("f*.jpg"))
            if not frame_paths:
                frame_paths = list(self.total_frame_dir.glob("*.jpg"))
            if not frame_paths:
                frame_paths = list(self.total_frame_dir.glob("frame_*.jpg"))
            if not frame_paths:
                frame_paths = list(self.total_frame_dir.glob("*.png"))
        
        if not frame_paths:
            return {}
        
        # Sort by frame number (numeric, not string) - ensures time order
        frame_paths = sorted(frame_paths, key=lambda p: extract_frame_number(p))
        
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            hist = compute_histogram(img)
            frame_num = extract_frame_number(fp)
            
            if prev_hist is None:
                current_group = [frame_num]
                prev_hist = hist
                continue

            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > scene_threshold and len(current_group) >= 1:
                groups_list.append(current_group)
                current_group = [frame_num]
            else:
                current_group.append(frame_num)
            prev_hist = hist

        if current_group:
            groups_list.append(current_group)

        # Merge tiny groups into previous to keep continuity
        merged = []
        for g in groups_list:
            if merged and len(g) < group_count:
                merged[-1].extend(g)
            else:
                merged.append(g)

        # Ensure minimum 30 groups by splitting large groups if needed
        min_groups = 30
        if len(merged) < min_groups:
            # Split the largest groups to reach minimum group count
            while len(merged) < min_groups:
                # Find the largest group
                largest_idx = max(range(len(merged)), key=lambda i: len(merged[i]))
                largest_group = merged[largest_idx]
                
                if len(largest_group) < 2:
                    # Can't split further
                    break
                
                # Split in half (maintain time order)
                mid = len(largest_group) // 2
                first_half = largest_group[:mid]
                second_half = largest_group[mid:]
                
                # Replace with two halves
                merged[largest_idx] = first_half
                merged.insert(largest_idx + 1, second_half)
        
        # Sort frame numbers within each group to ensure time order
        for g in merged:
            g.sort()
        
        groups = {i: grp for i, grp in enumerate(merged)}

        #save json file at the working directory (frame numbers only).
        with open(self.groups_json_path, "w") as f:
            json.dump(groups, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved groups to {self.groups_json_path}")
        return

    def score_frames(self, target_count: int = 30, per_group_selected: int = 1, yolo_weights: str = "yolov8n.pt"):
        '''
        read the josn file
        score the frames based on the criteria.
        - 
        
        #YOLO models:
        "yolov8n.pt" (nano) - 가장 빠름, 정확도 낮음
        "yolov8s.pt" (small) - 균형
        "yolov8m.pt" (medium) - 더 정확
        "yolov8l.pt" (large) - 매우 정확
        "yolov8x.pt" (extra large) - 가장 정확, 느림
        '''
        from .score import (
            sharpness as s_sharp,
            subject_visibility as s_subj,
            aesthetic_score as s_aesth,
            combine_scores,
        )

        # optional YOLO model
        yolo_model = None
        if yolo_weights is not None:
            print(f"🔍 Attempting to load YOLO model: {yolo_weights}")
            try:
                from ultralytics import YOLO  # type: ignore
                print("  ✅ ultralytics module imported successfully")
                print(f"  🔄 Loading model weights: {yolo_weights}")
                yolo_model = YOLO(yolo_weights)
                print(f"  ✅ YOLO model loaded successfully")
            except Exception as e:
                if isinstance(e, ImportError):
                    print(f"  Failed to import ultralytics: {e}")
                else:
                    print(f"  Failed to load YOLO model: {type(e).__name__}: {e}")
                yolo_model = None
        else:
            print("Skipping YOLO model")
        
        elected_frame_numbers = []
        global_scored = []

        # read the json file at the working directory (frame numbers).
        with open(self.groups_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        # Convert frame numbers to paths using paths.py function
        groups = {int(g): [_frame_number_to_path(self.total_frame_dir, fn) for fn in lst] 
                 for g, lst in d.items()}

        num_groups = len(groups)
        if num_groups == 0:
            print("⚠️  No groups found")
            return []
                
        print(f"Groups: {num_groups}, Per group: {per_group_selected} (fixed), Expected total: ~{num_groups * per_group_selected}")

        # score the frames based on the criteria.
        for gid, paths in groups.items():
            per_group_scores = []  # (total, frame_number)
            for fp in paths:
                img = cv2.imread(str(fp))
                if img is None:
                    continue

                # three different images scoring.
                v1 = s_sharp(img)
                v2 = s_subj(img, yolo_model=yolo_model)
                v3 = s_aesth(img)
                total = combine_scores(v1, v2, v3)
                # Extract frame number from path
                # Support both f123.jpg and 123.jpg formats
                stem = fp.stem
                if stem.startswith("f") and stem[1:].isdigit():
                    frame_num = int(stem[1:])  # f123 -> 123
                elif stem.isdigit():
                    frame_num = int(stem)  # 123 -> 123
                else:
                    # Try to extract from legacy frame_0123 format
                    frame_num = int(stem.split("_")[-1]) if "_" in stem else int(stem)
                per_group_scores.append((total, frame_num))
            if not per_group_scores:
                continue
            per_group_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select fixed number of frames per group
            topk = per_group_scores[:per_group_selected]
            for total, frame_num in topk:
                elected_frame_numbers.append(frame_num)
                global_scored.append((total, frame_num))

        # If we have more frames than target_count, select top frames globally
        if len(elected_frame_numbers) > target_count:
            global_scored.sort(key=lambda x: x[0], reverse=True)
            elected_frame_numbers = [fn for _, fn in global_scored[:target_count]]
            print(f"⚠️  Selected {len(elected_frame_numbers)} frames (limited by target_count={target_count})")

        # save the selected frames to the json file (frame numbers only).
        print(f"Selected frames: {elected_frame_numbers}")
        with open(self.selected_json_path, "w", encoding="utf-8") as f:
            json.dump(elected_frame_numbers, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved selected frames to {self.selected_json_path} (n={len(elected_frame_numbers)})")

        # Return paths for backward compatibility
        elected_frames = [_frame_number_to_path(self.total_frame_dir, fn) for fn in elected_frame_numbers]
        return elected_frames
     
    def save_selected_frames(self):
        '''
        save the selected frames to the working directory
        '''
        # read the json file
        with open(self.selected_json_path, "r", encoding="utf-8") as f:
            elected_frame_numbers = json.load(f)
        
        save_dir = self.working_dir / "selected_frames"
        save_dir.mkdir(parents=True, exist_ok=True)
        # save the selected frames to the working directory
        for fn in elected_frame_numbers:
            img = cv2.imread(str(_frame_number_to_path(self.total_frame_dir, fn)))
            cv2.imwrite(str(save_dir / f"{fn}.jpg"), img)
        print(f"✅ Saved selected frames to {save_dir} (n={len(elected_frame_numbers)})")
        return