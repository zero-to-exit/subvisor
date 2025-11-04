# 동영상 프레임 샘플링 및 선별 방법

## 일반적으로 사용되는 방법들

### 1. **균등 간격 샘플링 (Uniform Sampling)**
가장 간단한 방법
- **방법**: N프레임마다 하나씩 추출
- **예시**: 10프레임마다 하나씩 → 30fps 영상에서 3fps로 축소
- **장점**: 빠르고 간단, 시간대 골고루 분포
- **단점**: 중요한 순간을 놓칠 수 있음

```python
sample_interval = 10  # 10프레임마다
sampled_indices = list(range(0, total_frames, sample_interval))
```

### 2. **씬 전환 기반 샘플링 (Scene Change Detection)**
PySceneDetect 같은 라이브러리 사용
- **방법**: 씬이 바뀌는 지점 감지 → 각 씬에서 대표 프레임 추출
- **예시**: 씬 1, 씬 2, 씬 3... 각 씬에서 3-5개씩
- **장점**: 스토리 구조 반영, 의미 있는 구간 보장
- **단점**: 빠른 컷 편집에서 오탐지 가능

```python
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector())
# 각 씬에서 샘플링
```

### 3. **품질 점수 기반 선별**
프레임별 점수를 매겨 상위 N개만 선택
- **방법**: 
  - 블러 정도 (Laplacian variance)
  - 색상/밝기 균형
  - 클리핑 체크
  - 화면 구성 (중앙 객체, 얼굴 유무)
- **장점**: 품질이 좋은 프레임만 선택
- **단점**: 시간 분산이 고르지 않을 수 있음

```python
# 현재 코드베이스에서 사용 중
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # 블러 체크
# 점수 높은 순으로 정렬 후 상위 N개 선택
```

### 4. **시간 분산 + 품질 점수 조합** (추천!)
- **방법**: 
  1. 먼저 균등 간격으로 샘플링 (예: 5프레임마다)
  2. 품질 점수 계산
  3. 시간대별로 나눠서 각 구간에서 상위 프레임 선택
- **장점**: 시간 분산 + 품질 모두 고려
- **단점**: 복잡도 증가

```python
# 시간대를 나눠서 각 구간에서 선택
segments = divide_video_by_time(duration, num_segments)
for segment in segments:
    frames_in_segment = filter_frames_by_time(frames, segment)
    best_frame = select_by_score(frames_in_segment)
```

### 5. **색상 클러스터링 기반**
시각적 다양성 보장
- **방법**: 
  1. 프레임별 색상 히스토그램 계산
  2. K-means 클러스터링
  3. 각 클러스터에서 대표 프레임 선택
- **장점**: 다양한 장면 보장
- **단점**: 내용 중요도는 반영 안 됨

### 6. **변화량 기반 (Motion Detection)**
움직임이 많은 구간에서 더 많이 샘플링
- **방법**: 
  - 이전/다음 프레임과 차이 계산 (optical flow)
  - 변화가 큰 구간에서 더 많이 샘플링
- **장점**: 액션이나 중요한 이벤트 포착
- **단점**: 정적인 품질 프레임 놓칠 수 있음

### 7. **AI 모델 기반 선별**
CLIP, Gemini Vision 등 사용
- **방법**: 
  1. 먼저 간단히 샘플링 (예: 100개)
  2. 각 프레임을 AI로 분석
  3. 프롬프트 기반 점수 매기기 (예: "product shot", "close-up")
  4. 상위 N개 선택
- **장점**: 의미 기반 선별 가능
- **단점**: 비용과 시간 소요

## 실제 구현 예시 (현재 코드베이스)

### 방법 1: 균등 간격 + 품질 점수
```python
# extract_top_frames.py에서 사용
sample_interval = 5  # 5프레임마다
sampled_indices = list(range(0, total_frames, sample_interval))

# 각 프레임 점수 계산
for frame in sampled_frames:
    score = calculate_quality_score(frame)
    
# 시간 분산 고려하며 상위 20개 선택
selected = select_frames_with_coverage(indices, scores, times, duration, k=20)
```

### 방법 2: Gemini에 여러 프레임 한번에 전송
```python
# analyze_frames_gemini.py에서 사용
images = [Image.open(img) for img in image_files[:10]]
content = [prompt] + images  # 프롬프트 + 10개 이미지
response = model.generate_content(content)
```

## Gemini/Vision API 제한사항

### Gemini 2.0 Flash
- **최대 이미지**: 10-20개 (프롬프트 포함)
- **토큰 제한**: 약 1M 토큰
- **방법**: 여러 프레임을 배열로 전송

### GPT-4 Vision / Claude Vision
- **최대 이미지**: 보통 10-20개
- **방법**: base64 인코딩으로 배열 전송

### Google Cloud Vision API
- **한 번에**: 1개 이미지 (또는 batch)
- **방법**: API를 여러 번 호출하거나 batch API 사용

## 추천 워크플로우

1. **1차 필터링**: 균등 간격 샘플링 (전체 → ~200개)
2. **2차 평가**: 품질 점수 계산 (블러, 클리핑 등)
3. **3차 선별**: 시간 분산 고려하며 상위 N개 선택 (~50개)
4. **4차 AI 분석**: Gemini/Vision에 전송하여 최종 평가 (~20개)

이렇게 하면 비용과 품질의 균형을 맞출 수 있습니다!


