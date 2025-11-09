'''
Scoring functions for Algorithm 1

The evaluation criteria for each frame are as follows:

- Sharpness: 
Measures how crisp and clear an image is,
 typically indicating good focus and detail. 
 Higher sharpness usually means a better quality frame.

- Subject Visibility: 
Evaluates how prominently the main subject 
(e.g., a person or object of interest) is visible 
and distinguishable within the frame, especially near the center.
Uses YOLO if available, otherwise falls back to edge-based center detection.

- Aesthetic Score: 
Assesses the overall visual appeal of the frame, 
including aspects like color balance (saturation), lighting (contrast), 
and exposure quality (penalty for over/under-exposure).

'''
import numpy as np
import cv2

try:
    # ultralytics는 선택적 의존성입니다. 설치되어 있지 않으면 YOLO 경로는 자동으로 우회됩니다.
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - 선택 의존성
    YOLO = None

def sharpness(frame: np.ndarray, norm_factor: float = 500.0) -> float:
    '''
    Measures how crisp and clear an image is,
    typically indicating good focus and detail.
    Higher sharpness usually means a better quality frame.

    cvtColor - only for grayscale image
    laplacian - high laplacian -> high sharpness
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(laplacian.var())

    # 0~1 normalized with tanh function.
    var_norm = float(np.tanh(var / norm_factor))

    return var_norm

def subject_visibility(
        frame: np.ndarray,
        yolo_model: "YOLO | None" = None,
        center_ratio: float = 0.4,
    ) -> float:
    '''
    Evaluates how prominently the main subject is visible near the center.

    - 기본: YOLO가 있으면 탐지 박스와 중앙 박스 겹침(중심성)×신뢰도를 점수화(0~1)
    - 보조: YOLO가 없으면 중앙 영역의 엣지 에너지 대비 전체 엣지 에너지 비율(0~1)

    Args:
        frame: BGR image
        yolo_model: ultralytics.YOLO 객체(선택)
        center_ratio: 중앙 박스 한 변 비율(0.4 => 이미지 중앙 40%)
    '''
    h, w = frame.shape[:2]
    cx0 = int((1 - center_ratio) / 2 * w)
    cy0 = int((1 - center_ratio) / 2 * h)
    cx1 = int((1 + center_ratio) / 2 * w)
    cy1 = int((1 + center_ratio) / 2 * h)

    if yolo_model is not None and YOLO is not None:
        try:
            # YOLO 추론(빠른 설정 권장: imgsz 축소, conf 상향)
            results = yolo_model.predict(source=frame, imgsz=640, verbose=False, conf=0.25)
            best = 0.0
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for b in r.boxes:
                    # xyxy, confidence
                    x0, y0, x1, y1 = map(float, b.xyxy[0].tolist())
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                    # 중심성과 면적을 함께 고려
                    inter_x0 = max(cx0, int(x0)); inter_y0 = max(cy0, int(y0))
                    inter_x1 = min(cx1, int(x1)); inter_y1 = min(cy1, int(y1))
                    inter_w = max(0, inter_x1 - inter_x0)
                    inter_h = max(0, inter_y1 - inter_y0)
                    inter_area = inter_w * inter_h
                    box_area = max(1.0, (x1 - x0) * (y1 - y0))
                    center_overlap = inter_area / box_area
                    # 너무 작은 박스 패널티(프레임 대비 비율)
                    size_ratio = box_area / float(w * h)
                    size_factor = float(np.tanh(size_ratio * 5.0))  # 0~1
                    score = center_overlap * conf * max(0.5, size_factor)
                    best = max(best, score)
            return float(np.clip(best, 0.0, 1.0))
        except Exception:
            pass  # YOLO 실패 시 폴백 사용

    # 폴백: 중앙 영역의 엣지 에너지 비율
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    total = float(np.sum(edges > 0)) + 1e-6
    center = float(np.sum(edges[cy0:cy1, cx0:cx1] > 0))
    return float(np.clip(center / total, 0.0, 1.0))

def aesthetic_score(frame: np.ndarray, norm_factor: float = 30.0, saturation_factor: float = 100.0) -> float:
    '''
    Lightweight proxy for aesthetic appeal (0~1):
    - contrast: 그레이 표준편차
    - saturation: HSV S 채널 평균
    - exposure penalty: 밝기 분포 테일
    
    Args:
        frame: BGR image
        norm_factor: contrast normalization factor, default 30.0
        saturation_factor: saturation normalization factor, default 100.0
    
    Returns:
        Aesthetic score (0~1)
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # first criteria - contrast, standard deviation of gray image
    contrast = float(np.std(gray))

    # 2nd criteria - average of saturation(채도)
    saturation = float(np.mean(hsv[:, :, 1]))
    
    # 3rd criteria - exposure penalty: 너무 어둡거나 밝은 픽셀 비율 측정(패널티 부과)
    dark_ratio = float(np.mean(gray < 20))
    bright_ratio = float(np.mean(gray > 235))
    exposure_penalty = float(np.clip(dark_ratio + bright_ratio, 0.0, 1.0))

    # normalize the scores with tanh function
    contrast_n = float(np.tanh(contrast / norm_factor))
    saturation_n = float(np.tanh(saturation / saturation_factor))
    score = 0.6 * contrast_n + 0.6 * saturation_n - 0.4 * exposure_penalty
    return float(np.clip(score, 0.0, 1.0))  # 0~1 normalized score

def combine_scores(
        sharp: float,
        subj: float,
        aesth: float,
        weights=(0.4, 0.4, 0.2),
    ) -> float:
    '''
    Weighted sum of sharpness, subject visibility, and aesthetic scores.
    
    Args:
        sharp: sharpness score (0~1)
        subj: subject visibility score (0~1)
        aesth: aesthetic score (0~1)
        weights: tuple of (sharpness_weight, subject_weight, aesthetic_weight), default (0.4, 0.4, 0.2)
    
    Returns:
        Combined score (0~1)
    '''
    w1, w2, w3 = weights
    s = w1 * sharp + w2 * subj + w3 * aesth
    return float(np.clip(s, 0.0, 1.0))