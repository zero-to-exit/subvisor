# 데이터셋 입력/출력 스키마 (학습용)

## 입력(데이터)
- 원본 단위: 영상 파일(예: mp4)
- 학습 샘플 단위: 단일 프레임 또는 짧은 클립의 중앙 프레임
- 권장 단위: 프레임 기반(옵션으로 이전/다음 프레임 제공 → 액션/모션 특징 개선)

### 메타/라벨 파일(JSONL 권장)
각 줄에 하나의 샘플:
```json
{
  "sample_id": "videoA_f012345",
  "video_path": "/abs/path/videoA.mp4",
  "time_sec": 123.45,
  "frame_path": "/abs/path/frames/videoA/012345.jpg",
  "prev_frame_path": "/abs/path/frames/videoA/012344.jpg",
  "next_frame_path": "/abs/path/frames/videoA/012346.jpg",
  "labels": {
    "overall": 82.0,
    "sharpness": 0.9,
    "exposure": 0.8,
    "noise_free": 0.75,
    "motion_blur_free": 0.85,
    "stability_proxy": 0.8,
    "composition": 0.7,
    "color_quality": 0.85,
    "graphics_legibility": 0.6,
    "subject_clearness": 0.9,
    "action_peakness": 0.5,
    "brand_presence": 0.3
  }
}
```
- `overall`은 0~100 권장(또는 0~1).
- 세부 항목은 0~1 범위 권장. 라벨이 없으면 생략 가능(학습 시 해당 특징 제외 혹은 전체만 학습).
- `prev_frame_path`/`next_frame_path`는 선택 항목(없으면 액션/모션 특징은 중립값 처리).

## 출력(모델)
- 프레임 단위 추론 결과(JSON Lines 또는 CSV)
  - `overall_pred`: 0~100
  - `per_key_pred`: 각 세부 항목 0~100

예시(JSON):
```json
{
  "sample_id": "videoA_f012345",
  "overall_pred": 84.2,
  "per_key_pred": {
    "sharpness": 92.1,
    "exposure": 80.4,
    "noise_free": 78.6,
    "motion_blur_free": 86.7,
    "stability_proxy": 80.0,
    "composition": 73.2,
    "color_quality": 88.5,
    "graphics_legibility": 65.3,
    "subject_clearness": 91.7,
    "action_peakness": 51.0,
    "brand_presence": 30.0
  }
}
```

- 20장 선별 결과(JSON)
```json
{
  "video_path": "/abs/path/videoA.mp4",
  "selected_indices": [12, 456, ...],
  "selected_times_sec": [25.50, 42.50, ...],
  "scores": [91.2, 88.0, ...]
}
```

## 파이프라인 권장
1) 프레임/이웃프레임 추출 → 2) `FeatureExtractor.compute_features`로 특징 추출 →
3) `WeightedScorer` 또는 `TrainableLinearModel`로 점수 예측 →
4) `select_diverse_top_k`로 20장 다양성/중복 억제 선별

## 주의
- 프레임 단위 점수는 `criteria.txt`의 가중치 기준을 반영.
- 세트(20장) 수준의 타임라인 커버리지/다양성은 선별 단계(후처리)에서 보장.





