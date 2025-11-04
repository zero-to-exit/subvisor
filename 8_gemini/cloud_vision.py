"""
Google Cloud Video Intelligence API를 사용하여 동영상을 통째로 분석합니다.
"""
import os
from pathlib import Path
from google.cloud import videointelligence
from google.cloud.videointelligence import Feature
from google.oauth2 import service_account

def analyze_video(video_path, credentials_path=None):
    """Google Cloud Video Intelligence API를 사용하여 동영상을 통째로 분석합니다."""
    # 인증 설정
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
    elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        # 환경 변수로 설정된 경우
        client = videointelligence.VideoIntelligenceServiceClient()
    else:
        # 기본 인증 시도
        client = videointelligence.VideoIntelligenceServiceClient()
    
    # 동영상 파일 읽기
    with open(video_path, 'rb') as video_file:
        input_content = video_file.read()
    
    # 분석할 기능 설정
    features = [
        Feature.LABEL_DETECTION,      # 객체/장면 라벨 감지
        Feature.SHOT_CHANGE_DETECTION,  # 씬 전환 감지
        Feature.SPEECH_TRANSCRIPTION,   # 음성 인식 (선택사항)
        Feature.TEXT_DETECTION,         # 텍스트 감지 (선택사항)
    ]
    
    # 동영상 분석 요청
    print(f"동영상 분석 시작: {video_path}")
    print("이 작업은 몇 분 정도 걸릴 수 있습니다...")
    
    operation = client.annotate_video(
        request={
            'input_content': input_content,
            'features': features,
        }
    )
    
    # 비동기 작업 완료 대기
    print("분석 진행 중...")
    result = operation.result(timeout=600)  # 최대 10분 대기
    
    return result

def print_analysis_results(result):
    """분석 결과를 보기 좋게 출력합니다."""
    print("\n" + "="*80)
    print("동영상 분석 결과")
    print("="*80)
    
    # 라벨 감지 결과
    if result.annotation_results:
        for annotation_result in result.annotation_results:
            # 장면별 라벨
            print("\n[장면별 라벨 감지]")
            for segment_label in annotation_result.segment_label_annotations:
                print(f"\n카테고리: {segment_label.entity.description}")
                print(f"신뢰도: {segment_label.entity.score:.2%}")
                
                # 시간대별 세그먼트
                for segment in segment_label.segments:
                    start_time = segment.segment.start_time_offset.total_seconds()
                    end_time = segment.segment.end_time_offset.total_seconds()
                    confidence = segment.confidence
                    print(f"  시간대: {start_time:.2f}s - {end_time:.2f}s (신뢰도: {confidence:.2%})")
            
            # 씬 전환 감지
            print("\n[씬 전환 감지]")
            for shot in annotation_result.shot_annotations:
                start_time = shot.start_time_offset.total_seconds()
                end_time = shot.end_time_offset.total_seconds()
                print(f"씬: {start_time:.2f}s - {end_time:.2f}s")
            
            # 텍스트 감지 (있는 경우)
            if annotation_result.text_annotations:
                print("\n[텍스트 감지]")
                for text_annotation in annotation_result.text_annotations:
                    print(f"텍스트: {text_annotation.text}")
                    for segment in text_annotation.segments:
                        start_time = segment.segment.start_time_offset.total_seconds()
                        confidence = segment.confidence
                        print(f"  시간: {start_time:.2f}s (신뢰도: {confidence:.2%})")

def main():
    """메인 함수"""
    import sys
    
    # 동영상 파일 경로
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("분석할 동영상 파일 경로를 입력하세요: ")
    
    if not os.path.exists(video_path):
        print(f"오류: 파일을 찾을 수 없습니다: {video_path}")
        return
    
    # 인증 키 파일 경로 (선택사항)
    credentials_path = None
    if len(sys.argv) > 2:
        credentials_path = sys.argv[2]
    elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        print(f"인증 정보: {credentials_path} 사용")
    
    if not credentials_path and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("\n⚠️  인증 정보가 설정되지 않았습니다.")
        print("다음 방법 중 하나를 사용하세요:")
        print("1. 환경 변수 설정: export GOOGLE_APPLICATION_CREDENTIALS='경로/키파일.json'")
        print("2. 인증 키 파일 경로를 두 번째 인자로 전달: python cloud_vision.py video.mp4 key.json")
        print("\nGoogle Cloud 인증 설정: https://cloud.google.com/docs/authentication")
        return
    
    # 동영상 분석
    try:
        result = analyze_video(video_path, credentials_path)
        print_analysis_results(result)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()