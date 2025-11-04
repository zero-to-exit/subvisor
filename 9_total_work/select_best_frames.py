#!/usr/bin/env python3
"""
Select best frames from extracted frame groups by quality and timeline coverage.

Inputs:
  - base_dir: path to `extracted_frames` directory containing group_XX folders and optional video_info.json
Outputs:
  - Creates `selected_frames_story/` under base_dir with ordered copies of selected frames
  - Writes `selection_report.json` with selection metadata and rationale

This script intentionally avoids third-party dependencies (uses stdlib only).
"""

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GROUP_DIR_PATTERN = re.compile(r"^group_(\d+)$")
TIME_PATTERN = re.compile(r"t(?P<time>[0-9]+(?:\.[0-9]+)?)s")
SCORE_PATTERN = re.compile(r"score(?P<score>[0-9]+(?:\.[0-9]+)?)")


@dataclass
class FrameMeta:
    path: str
    filename: str
    group_index: int
    time_s: float
    quality_score: float
    file_size_bytes: int


def parse_time_from_filename(name: str) -> Optional[float]:
    m = TIME_PATTERN.search(name)
    return float(m.group("time")) if m else None


def parse_score_from_filename(name: str) -> Optional[float]:
    m = SCORE_PATTERN.search(name)
    return float(m.group("score")) if m else None


def load_video_info_scores(video_info_path: Path) -> Dict[str, Dict]:
    if not video_info_path.exists():
        return {}
    try:
        with video_info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    # Try to index by filename
    mapping: Dict[str, Dict] = {}
    frames = data.get("extracted_frames") or data.get("frames") or []
    for entry in frames:
        # Common keys we expect: filename, path, time, quality_score
        filename = entry.get("filename") or os.path.basename(entry.get("path", ""))
        if not filename:
            continue
        mapping[filename] = entry
    return mapping


def discover_groups(base_dir: Path) -> List[Tuple[int, Path]]:
    groups: List[Tuple[int, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        m = GROUP_DIR_PATTERN.match(child.name)
        if m:
            groups.append((int(m.group(1)), child))
    groups.sort(key=lambda x: x[0])
    return groups


def collect_frames(base_dir: Path) -> List[FrameMeta]:
    video_info_path = base_dir / "video_info.json"
    info_scores = load_video_info_scores(video_info_path)

    frames: List[FrameMeta] = []
    for group_index, group_dir in discover_groups(base_dir):
        for entry in sorted(group_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            filename = entry.name
            time_s = parse_time_from_filename(filename)
            score = None

            # Prefer video_info.json score if present
            info = info_scores.get(filename)
            if info is not None:
                score = info.get("quality_score")
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except Exception:
                        score = None
            # Fallback: parse from filename
            if score is None:
                score = parse_score_from_filename(filename) or 0.0

            # Fallback: parse time from video_info.json if missing
            if time_s is None and info is not None:
                t = info.get("time") or info.get("time_s")
                if isinstance(t, (int, float)):
                    time_s = float(t)

            if time_s is None:
                # If no time, approximate using group order with small offset
                time_s = float(group_index)

            try:
                size_b = entry.stat().st_size
            except Exception:
                size_b = 0

            frames.append(
                FrameMeta(
                    path=str(entry),
                    filename=filename,
                    group_index=group_index,
                    time_s=float(time_s),
                    quality_score=float(score),
                    file_size_bytes=size_b,
                )
            )

    # Sort by time (stable)
    frames.sort(key=lambda x: (x.time_s, x.group_index, x.filename))
    return frames


def select_best_per_group(frames: List[FrameMeta]) -> List[FrameMeta]:
    best_by_group: Dict[int, FrameMeta] = {}
    for f in frames:
        best = best_by_group.get(f.group_index)
        if best is None or f.quality_score > best.quality_score:
            best_by_group[f.group_index] = f
    selected = list(best_by_group.values())
    selected.sort(key=lambda x: x.time_s)
    return selected


def segment_timeline(frames: List[FrameMeta], segments: int) -> List[Tuple[float, float]]:
    if not frames:
        return []
    start = min(f.time_s for f in frames)
    end = max(f.time_s for f in frames)
    if end <= start:
        return [(start, end)]
    step = (end - start) / float(segments)
    bounds: List[Tuple[float, float]] = []
    seg_start = start
    for i in range(segments):
        seg_end = end if i == segments - 1 else (start + (i + 1) * step)
        bounds.append((seg_start, seg_end))
        seg_start = seg_end
    return bounds

def select_story_coverage(candidates: List[FrameMeta], segments: int, top_overall: int,
                          min_gap_seconds: float = 1.0) -> List[FrameMeta]:
    if not candidates:
        return []

    # Segment coverage: pick best by quality within each time segment
    bounds = segment_timeline(candidates, segments)
    by_segment: List[FrameMeta] = []
    for (lo, hi) in bounds:
        in_seg = [f for f in candidates if (f.time_s >= lo and f.time_s <= hi)]
        if not in_seg:
            continue
        best = max(in_seg, key=lambda f: (f.quality_score, -f.time_s))
        by_segment.append(best)

    # Global top add-on while keeping spacing and avoiding duplicates
    remaining = sorted(candidates, key=lambda f: f.quality_score, reverse=True)
    selected: List[FrameMeta] = []

    def far_enough(a: FrameMeta, b: FrameMeta) -> bool:
        return abs(a.time_s - b.time_s) >= min_gap_seconds

    for f in by_segment:
        if f not in selected:
            selected.append(f)

    for f in remaining:
        if f in selected:
            continue
        if all(far_enough(f, s) for s in selected):
            selected.append(f)
        if len(selected) >= max(len(by_segment), top_overall):
            break

    # Order by time for storytelling
    selected.sort(key=lambda x: x.time_s)
    return selected

def write_report(out_dir: Path, all_frames: List[FrameMeta],
                 best_per_group: List[FrameMeta],
                 final_selection: List[FrameMeta]) -> None:
    report = {
        "total_frames": len(all_frames),
        "total_groups": len({f.group_index for f in all_frames}),
        "best_per_group_count": len(best_per_group),
        "final_selection_count": len(final_selection),
        "final_selection_paths": [f.path for f in final_selection],
        "final_selection": [asdict(f) for f in final_selection],
        "best_per_group": [asdict(f) for f in best_per_group],
    }
    (out_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def copy_selection(out_dir: Path, selection: List[FrameMeta]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(selection, start=1):
        dst_name = f"S{idx:02d}_t{f.time_s:.2f}s_score{f.quality_score:.3f}_g{f.group_index}{Path(f.path).suffix.lower()}"
        shutil.copy2(f.path, out_dir / dst_name)

def main() -> None:
    parser = argparse.ArgumentParser(description="Select best frames for evaluation and storytelling coverage.")
    parser.add_argument("base_dir", type=str, help="Path to extracted_frames directory")
    parser.add_argument("--segments", type=int, default=6, help="Timeline segments for coverage selection")
    parser.add_argument("--top_overall", type=int, default=12, help="Target total selection size")
    parser.add_argument("--min_gap", type=float, default=1.0, help="Minimum seconds between selected frames")
    parser.add_argument("--out_name", type=str, default="selected_frames_story", help="Output folder name under base_dir")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    all_frames = collect_frames(base_dir)
    if not all_frames:
        raise SystemExit("No frames found under group_* directories.")

    best_per_group = select_best_per_group(all_frames)
    selection = select_story_coverage(
        candidates=best_per_group,
        segments=args.segments,
        top_overall=args.top_overall,
        min_gap_seconds=args.min_gap,
    )

    out_dir = base_dir / args.out_name
    copy_selection(out_dir, selection)
    write_report(out_dir, all_frames, best_per_group, selection)

    print(f"Selected {len(selection)} frames → {out_dir}")





if __name__ == "__main__":
    main()


