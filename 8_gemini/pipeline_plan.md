# YouTube 광고 영상 고품질 프레임 추출 파이프라인 계획

## 전체 프로세스 개요

```
YouTube URL 수집 (1000개)
    ↓
영상 다운로드
    ↓
1차 필터링 (유사도 기반 + 품질 필터링) → ~1000장
    ↓
3x3 그리드 생성 (시간순 정렬)
    ↓
ChatGPT 평가 (Playwright 자동화)
    ↓
최종 상위 20장 선정 (시간 분산 고려)
    ↓
저장 및 결과 리포트
```



---

## Phase 1: YouTube 광고 영상 URL 수집

### 1.1 YouTube MCP를 통한 검색 및 수집
- **목표**: 1분 안팎의 광고 영상 URL 1000개 수집
- **방법**: 
  - YouTube MCP `searchVideos` 활용
  - 검색 키워드: "광고", "advertisement", "commercial", "cm" 등
  - 필터: `videoDuration: "short"` (4분 이하)
  - `maxResults: 500` (여러 검색어로 반복하여 1000개 확보)

### 1.2 메타데이터 수집 및 검증
- 각 영상의 `duration` 확인 (1분 ±30초 범위)
- 영상 길이, 해상도, 조회수 등 메타데이터 수집
- 광고 특성 검증 (설명/태그에 광고 키워드 포함 여부)

### 1.3 CSV 파일 저장
- **파일명**: `youtube_ads_1000.csv`
- **컬럼 구조**:
  ```csv
  index,url,video_id,title,duration,resolution,views,upload_date,category
  0,https://youtube.com/watch?v=xxx,video_id_0,광고제목,65,1920x1080,12345,2024-01-01,광고
  1,https://youtube.com/watch?v=yyy,video_id_1,광고제목2,58,1280x720,67890,2024-01-02,광고
  ...
  999,...
  ```

---

## Phase 2: 영상 다운로드

### 2.1 다운로드 도구 선택
- **옵션 1**: `yt-dlp` (추천)
  - 명령어: `yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]" -o "downloads/{video_id}.mp4"`
- **옵션 2**: YouTube MCP가 직접 다운로드 지원하는지 확인
- **옵션 3**: Playwright로 자동 다운로드 (느리고 불안정할 수 있음)

### 2.2 다운로드 디렉토리 구조
```
downloads/
├── video_0.mp4
├── video_1.mp4
├── ...
└── video_999.mp4
```

### 2.3 다운로드 실행
- CSV 파일을 순차적으로 읽어서 다운로드
- 실패한 항목은 재시도 (최대 3회)
- 다운로드 실패 리스트 별도 저장

### 2.4 검증
- 다운로드 완료된 파일 개수 확인
- 파일 크기/길이 검증 (손상된 파일 제외)

---

## Phase 3: 1차 필터링 - 고품질 프레임 추출

### 3.1 프레임 추출 전략
- **목표**: 각 영상에서 유사한 장면은 스킵하고 고품질 프레임만 추출
- **목표 프레임 수**: 100장 (평균 영상당 1장)

### 3.2 유사도 기반 필터링
**Step 1: 균등 샘플링**
- 각 영상에서 1초마다 프레임 추출 (60프레임/영상)
- 총 60,000 프레임 (1000 영상 × 60)

**Step 2: 유사도 계산 및 클러스터링**
- 각 프레임을 이미지 벡터로 변환 (VGG/ResNet feature extraction)
- Cosine similarity 또는 Euclidean distance 계산
- 유사도 임계값 설정 (예: 0.95 이상 = 동일 장면)
- 유사한 프레임들을 클러스터로 그룹화
- 각 클러스터에서 **1장만** 선택

**Step 3: 품질 평가 및 필터링**
각 클러스터에서 선택할 프레임 평가 기준:
1. **선명도 (Sharpness)**
   - Laplacian variance 계산
   - 높을수록 선명
   - 임계값: 100 이상

2. **블러 검출 (Blur Detection)**
   - FFT 기반 블러 메트릭
   - 블러 점수 낮을수록 좋음
   - 임계값: 0.5 이하

3. **해상도 및 비트레이트**
   - 원본 영상 해상도 확인
   - 낮은 해상도 프레임 제외

4. **흔들림 검출 (Motion Blur)**
   - 연속 프레임 간 차이 분석
   - 급격한 변화 = 흔들림 가능성
   - 안정적인 프레임 우선 선택

**Step 4: 클러스터별 최고 품질 프레임 선택**
- 각 클러스터에서 위 기준의 합산 점수가 가장 높은 1장 선택
- 목표: ~1000장 (클러스터 수와 비슷하게)

### 3.3 중간 저장
- 필터링된 프레임들을 `filtered_frames/` 디렉토리에 저장
- 파일명: `video_{index}_frame_{timestamp}_{score}.jpg`
- 메타데이터 저장: `filtered_frames_metadata.json`

---

## Phase 4: 3x3 그리드 생성 (시간순 정렬)

### 4.1 프레임 시간순 정렬
- 각 프레임의 타임스탬프를 기준으로 전체 프레임 정렬
- 시간 순서: `video_index → timestamp` 순

### 4.2 그리드 생성 로직
- **규칙**: 오른쪽으로 갈수록 시간 증가, 아래로 갈수록 시간 증가
```
┌───────┬───────┬───────┐
│  T1   │  T2   │  T3   │  ← 시간 순서
├───────┼───────┼───────┤
│  T4   │  T5   │  T6   │
├───────┼───────┼───────┤
│  T7   │  T8   │  T9   │
└───────┴───────┴───────┘
```

- 1000장 → 약 112개 그리드 (1000 ÷ 9 = 111.11...)
- 각 그리드 이미지 크기: 1920×1080 × 3 = 5760×3240 (또는 적절히 조정)

### 4.3 그리드 생성 구현
```python
def create_grid(frames_list, grid_size=(3, 3)):
    grids = []
    for i in range(0, len(frames_list), 9):
        batch = frames_list[i:i+9]
        if len(batch) == 9:
            grid_image = arrange_in_grid(batch, 3, 3)
            grids.append({
                'grid_id': i // 9,
                'frame_indices': [f['index'] for f in batch],
                'timestamps': [f['timestamp'] for f in batch],
                'image': grid_image
            })
    return grids
```

### 4.4 그리드 저장
- `grids/` 디렉토리에 저장
- 파일명: `grid_{index}_frames_{start_idx}_to_{end_idx}.jpg`
- 메타데이터: `grid_metadata.json`

---

## Phase 5: ChatGPT 평가 (Playwright 자동화)

### 5.1 평가 프롬프트 작성
**핵심 규칙 명시**:
```
"이 이미지는 3x3 그리드로 구성되어 있으며, 시간순으로 배치되어 있습니다.
왼쪽 위부터 오른쪽으로, 다음 줄로 넘어가며 시간이 진행됩니다.

위치 순서:
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 4 │ 5 │ 6 │
├───┼───┼───┤
│ 7 │ 8 │ 9 │
└───┴───┴───┘
```

**평가 기준**:

1. **독특한 촬영기법 점수 (1000점 또는 0점)**
   - 이 프레임이 독특한 촬영 기법을 보여주는가?
   - 편집자가 시간을 많이 들인 것 같은 인상이 드는가?
   - 특별한 앵글, 조명, 구도, 특수효과가 있는가?
   - **YES**: 1000점, **NO**: 0점

2. **인물/사물 배치 및 선명도 (0~100점)**
   - 인물이나 주요 사물이 화면 중앙에 배치되어 있는가? (30점)
   - 얼굴이 선명하게 나오는가? (30점)
   - 표정이 자연스러운가? (20점)
   - 초점이 정확한가? (20점)

3. **제품 스토리텔링 (0~100점)**
   - 광고의 제품/서비스가 명확히 보이는가? (30점)
   - 제품의 기능/장점이 시각적으로 전달되는가? (30점)
   - 스토리텔링 맥락이 담겨있는가? (40점)

4. **영상 품질 (0~50점)**
   - 선명도, 해상도, 노이즈 수준
   - 블러, 흔들림, 압축 아티팩트

**총점 계산**:
- 총점 = 독특한 촬영기법 점수 + 인물/사물 점수 + 스토리텔링 점수 + 영상 품질 점수
- 최대 점수: 1000 + 100 + 100 + 50 = 1250점

### 5.2 출력 형식
```json
{
  "grid_id": 1,
  "evaluations": [
    {
      "position": 1,
      "original_frame_index": 123,
      "timestamp": 15.5,
      "unique_shooting_score": 1000,
      "composition_score": 85,
      "storytelling_score": 90,
      "quality_score": 45,
      "total_score": 1220,
      "reasons": {
        "unique_shooting": "특별한 로우앵글과 조명 효과",
        "composition": "주인공이 중앙, 얼굴 선명",
        "storytelling": "제품 사용 장면이 명확",
        "quality": "고해상도, 선명함"
      }
    },
    ...
  ]
}
```

### 5.3 Playwright 자동화 프로세스

**Step 1: ChatGPT 로그인 확인**
- 이미 로그인되어 있는지 확인
- 필요시 로그인 (이미 완료됨)

**Step 2: 배치 처리**
- 112개 그리드를 10개씩 묶음으로 처리
- 각 묶음마다:
  1. 새 대화 시작 또는 기존 대화 계속
  2. 그리드 이미지 10장 업로드
  3. 평가 프롬프트 전송
  4. 응답 대기 및 JSON 파싱
  5. 결과 저장

**Step 3: 결과 수집**
- 모든 평가 결과를 하나의 JSON 파일로 병합
- `evaluation_results.json`

---

## Phase 6: 최종 선정 및 시간 분산

### 6.1 점수 기반 정렬
- 모든 프레임의 총점을 기준으로 내림차순 정렬
- 상위 100개 후보 선정 (여유있게)

### 6.2 시간 분산 알고리즘
**목표**: 특정 시간대에 몰리지 않도록 분산

**알고리즘**:
1. 전체 영상 길이를 시간대로 구간 분할 (예: 10초 구간)
2. 각 구간별로 최대 1~2장만 선택
3. 상위 점수 순으로 선택하되, 이미 선택된 구간은 스킵
4. 최종 20장 확보

**구현 예시**:
```python
def select_with_time_distribution(candidates, top_n=20, time_window=10):
    # 시간 구간별로 그룹화
    time_slots = {}
    for frame in candidates:
        slot = frame['timestamp'] // time_window
        if slot not in time_slots:
            time_slots[slot] = []
        time_slots[slot].append(frame)
    
    # 각 구간에서 최고 점수만 선택
    selected = []
    for slot in sorted(time_slots.keys()):
        slot_frames = sorted(time_slots[slot], key=lambda x: x['total_score'], reverse=True)
        selected.extend(slot_frames[:2])  # 구간당 최대 2장
    
    # 최종 상위 20장
    final = sorted(selected, key=lambda x: x['total_score'], reverse=True)[:top_n]
    return final
```

### 6.3 최종 프레임 저장
- 상위 20장을 `final_selection/` 디렉토리에 저장
- 파일명: `rank_{rank}_score_{score}_time_{timestamp}.jpg`
- 최종 리포트: `final_report.json` (순위, 점수, 시간, 이유 포함)

---

## Phase 7: 결과 리포트 생성

### 7.1 리포트 내용
- 전체 통계 (평균 점수, 분포 등)
- 상위 20장 상세 정보
- 시간 분포 그래프
- 점수 분포 그래프
- 각 평가 기준별 평균 점수

### 7.2 출력 형식
- JSON 리포트
- CSV 요약
- 시각화 그래프 (선택사항)

---

## 기술 스택 요약

### 사용 도구
1. **YouTube MCP**: URL 수집, 메타데이터
2. **yt-dlp**: 영상 다운로드
3. **OpenCV / PIL**: 프레임 추출, 유사도 계산, 그리드 생성
4. **scikit-learn / numpy**: 유사도 계산, 클러스터링
5. **Playwright MCP**: ChatGPT 웹 자동화
6. **Python**: 전체 파이프라인 스크립팅

### 주요 라이브러리
- `opencv-python`: 영상 처리, 프레임 추출
- `numpy`: 수치 계산
- `scikit-learn`: 유사도, 클러스터링
- `PIL/Pillow`: 이미지 처리, 그리드 생성
- `yt-dlp`: YouTube 다운로드
- `playwright`: 브라우저 자동화

---

## 예상 처리 시간

- Phase 1 (URL 수집): ~30분
- Phase 2 (다운로드): ~2-3시간 (인터넷 속도에 따라)
- Phase 3 (필터링): ~1-2시간 (60,000 프레임 처리)
- Phase 4 (그리드 생성): ~10분
- Phase 5 (ChatGPT 평가): ~2-3시간 (112개 그리드, 자동화 속도에 따라)
- Phase 6-7 (선정 및 리포트): ~5분

**총 예상 시간**: 약 6-9시간

---

## 주의사항 및 개선 가능한 부분

### 주의사항
1. YouTube 다운로드 제한 (ToS 확인 필요)
2. ChatGPT API 제한 (너무 빠른 요청 시 제한 가능)
3. 저장 공간 필요 (1000개 영상 + 프레임)
4. 네트워크 안정성

### 개선 가능
1. 병렬 처리 (다운로드, 프레임 추출)
2. 실패 재시도 로직 강화
3. 중간 체크포인트 저장 (실패 시 재시작)
4. 진행 상황 모니터링 대시보드

---

## 파일 구조

```
project/
├── youtube_ads_1000.csv
├── downloads/
│   ├── video_0.mp4
│   └── ...
├── filtered_frames/
│   ├── video_0_frame_15.5_850.jpg
│   └── ...
├── filtered_frames_metadata.json
├── grids/
│   ├── grid_0_frames_0_to_8.jpg
│   └── ...
├── grid_metadata.json
├── evaluation_results.json
├── final_selection/
│   ├── rank_1_score_1220_time_15.5.jpg
│   └── ...
└── final_report.json
```

---

이 계획으로 진행하면 됩니다. 각 Phase별로 순차적으로 구현하면 됩니다.


