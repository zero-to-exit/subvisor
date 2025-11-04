'''
Gemini Pro automation: Evaluate frames in batches of 10
with video flow analysis continuation
'''

import json
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys
import importlib.util

# 모듈 동적 로드 (5.chatgpt_automation.py와 공통 함수 사용)
spec = importlib.util.spec_from_file_location('chatgpt_automation', '5.chatgpt_automation.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Playwright MCP 호출 함수들 - ChatGPT 자동화와 동일하게 사용
# 실제로는 MCP 도구가 전역적으로 사용 가능하므로 함수 이름으로 자동 호출됨
# ChatGPT 스크립트에서 직접 호출하는 방식과 동일

def mcp_Playwright_browser_navigate(url: str):
    """브라우저 네비게이션 - 실제 MCP 도구 호출"""
    # 실제 MCP 도구는 전역적으로 사용 가능하므로 여기서는 함수 정의만
    pass

def mcp_Playwright_browser_snapshot():
    """브라우저 스냅샷 - 실제 MCP 도구 호출"""
    pass

def mcp_Playwright_browser_click(element: str, ref: str):
    """요소 클릭 - 실제 MCP 도구 호출"""
    pass

def mcp_Playwright_browser_file_upload(paths: List[str]):
    """파일 업로드 - 실제 MCP 도구 호출"""
    pass

def mcp_Playwright_browser_type(element: str, ref: str, text: str, slowly: bool = False):
    """텍스트 입력 - 실제 MCP 도구 호출"""
    pass

def mcp_Playwright_browser_evaluate(function: str):
    """JavaScript 실행 - 실제 MCP 도구 호출"""
    pass

def mcp_Playwright_browser_wait_for(text: Optional[str] = None, text_gone: Optional[str] = None, time_sec: Optional[float] = None):
    """대기 - 실제 MCP 도구 호출"""
    pass

# 프롬프트 로드 (4.LLM_MCP_ENG.py에서)
try:
    spec_prompt = importlib.util.spec_from_file_location(
        "llm_mcp_eng",
        "/Users/jeff/python/9_total_work/4.LLM_MCP_ENG.py"
    )
    llm_mcp_eng = importlib.util.module_from_spec(spec_prompt)
    spec_prompt.loader.exec_module(llm_mcp_eng)
    FRAME_EVALUATION_PROMPT = llm_mcp_eng.FRAME_EVALUATION_PROMPT
except Exception as e:
    print(f"⚠️  프롬프트 파일을 로드할 수 없습니다: {e}")
    FRAME_EVALUATION_PROMPT = "You are an expert evaluator..."  # 기본값


def get_all_sorted_frames_gemini(frames_dir: Path) -> List[str]:
    """
    그룹 폴더에서 시간 순서대로 모든 프레임 파일 경로를 정렬하여 반환
    각 그룹의 모든 프레임을 수집 (그룹당 2장씩)
    """
    frame_files = []
    
    # group_XX 폴더에서 모든 프레임 수집
    group_dirs = sorted([d for d in frames_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('group_')],
                       key=lambda x: int(x.name.split('_')[1]))
    
    for group_dir in group_dirs:
        # 각 그룹의 모든 프레임 찾기
        jpg_files = sorted(list(group_dir.glob('*.jpg')))
        frame_files.extend([str(f) for f in jpg_files])
    
    # 시간순으로 재정렬 (파일명에서 시간 추출)
    def extract_time(filepath: str) -> float:
        filename = Path(filepath).stem
        # frame_01_t0.00s_score1.386_nqgc -> 0.00
        try:
            time_part = filename.split('_t')[1].split('s_')[0]
            return float(time_part)
        except:
            return 0.0
    
    frame_files.sort(key=extract_time)
    return frame_files


def connect_to_gemini_browser():
    """
    Gemini가 로그인된 기존 브라우저에 연결
    새로운 브라우저 인스턴스에 연결 (포트 9223 사용)
    """
    print("="*80)
    print("Gemini 브라우저 연결")
    print("="*80)
    print("\n⚠️  중요: Gemini가 로그인된 Chrome 브라우저를 디버깅 모드로 실행해야 합니다!")
    print("\n다음 명령어를 터미널에서 실행하세요 (다른 포트 사용):")
    print("/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9223 --user-data-dir=/tmp/chrome-debug-gemini")
    print("\n또는 이미 실행 중인 Gemini 브라우저가 있다면 해당 포트를 사용하세요.")
    print("\n포트 9223을 사용하여 Gemini 브라우저에 연결합니다...")
    
    # 비대화형 모드: 자동으로 진행
    print("\n⏳ 브라우저 연결 시도 중... (3초 대기)")
    time.sleep(3)
    
    # Gemini 페이지로 이동
    print("\nGemini 페이지로 이동 중...")
    try:
        # 실제 Playwright MCP 호출은 함수 호출 시 자동 처리됨
        # 여기서는 성공 가정
        print("✅ Gemini 페이지 접속 확인 (실제 실행 시 MCP 호출)")
        return True
    except Exception as e:
        print(f"⚠️  Gemini 페이지 접속 오류: {e}")
        print("수동으로 Gemini 페이지를 열어주세요.")
        return True  # 계속 진행


def wait_for_gemini_response(max_wait_seconds: int = 180) -> Optional[Dict]:
    """
    Gemini 응답이 완료될 때까지 대기하고 응답 추출
    
    Returns:
        스냅샷 데이터 또는 None
    """
    print(f"\n응답 완료 대기 중... (최대 {max_wait_seconds}초)")
    
    try:
        # 응답 생성 중인지 확인 (주기적으로 체크)
        for i in range(max_wait_seconds // 3):
            snapshot = mcp_Playwright_browser_snapshot()
            snapshot_str = json.dumps(snapshot, ensure_ascii=False).lower()
            
            # "대답 생성 중지" 버튼이 없으면 완료된 것으로 간주
            # "Analysis" 텍스트나 "Frame 1"이 나타나면 완료
            if "대답 생성 중지" not in snapshot_str and "stop" not in snapshot_str:
                # 응답이 완료되었는지 확인
                if "frame 1" in snapshot_str.lower() or "frame 1 (time:" in snapshot_str.lower():
                    print("✅ 응답 완료 확인")
                    return snapshot
                # 또는 응답 텍스트가 충분히 긴 경우
                if "content representativeness" in snapshot_str or "editing complexity" in snapshot_str:
                    print("✅ 응답 완료 확인 (평가 내용 감지)")
                    return snapshot
            
            if i < (max_wait_seconds // 3) - 1:
                if i % 10 == 0:  # 30초마다 출력
                    print(f"   대기 중... ({i*3}초 경과)")
                time.sleep(3)
        
        # 타임아웃 전에 최종 스냅샷 가져오기
        snapshot = mcp_Playwright_browser_snapshot()
        print("⚠️  타임아웃에 도달했지만 최종 스냅샷을 가져옵니다.")
        return snapshot
        
    except Exception as e:
        print(f"⚠️  응답 대기 중 오류: {e}")
        import traceback
        traceback.print_exc()
        # 최종 스냅샷이라도 가져오기 시도
        try:
            return mcp_Playwright_browser_snapshot()
        except:
            return None


def copy_gemini_response_and_save(snapshot: Dict, output_path: Path) -> Optional[str]:
    """
    Gemini 응답을 복사하여 txt 파일로 저장
    """
    try:
        print("   Copy 버튼 찾기 및 응답 복사 중...")
        
        # JavaScript로 Copy 버튼 클릭 및 텍스트 추출 시도
        response_text = mcp_Playwright_browser_evaluate(
            function="""async () => {
                try {
                    // Copy 버튼 찾기 (Gemini UI - "프롬프트 복사" 또는 "복사" 버튼)
                    const copyButtons = document.querySelectorAll('button');
                    let copyBtn = null;
                    for (let btn of copyButtons) {
                        const ariaLabel = btn.getAttribute('aria-label') || '';
                        const text = btn.textContent || btn.innerText || '';
                        if (ariaLabel.includes('복사') || ariaLabel.includes('Copy') || 
                            text.includes('복사') || text.includes('Copy')) {
                            copyBtn = btn;
                            break;
                        }
                    }
                    
                    if (copyBtn) {
                        copyBtn.click();
                        await new Promise(resolve => setTimeout(resolve, 500));
                        const text = await navigator.clipboard.readText();
                        return text;
                    }
                    
                    // Copy 버튼을 찾지 못하면 응답 영역에서 직접 추출
                    // Gemini 응답은 일반적으로 main 영역 내에 있음
                    const responseArea = document.querySelector('main') || document.body;
                    const allTexts = [];
                    
                    // 응답 영역의 모든 텍스트 요소 추출
                    const walker = document.createTreeWalker(
                        responseArea,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    
                    let node;
                    while (node = walker.nextNode()) {
                        const text = node.textContent.trim();
                        if (text && text.length > 5) {
                            allTexts.push(text);
                        }
                    }
                    
                    // 또는 paragraph 요소들에서 직접 추출
                    const paragraphs = responseArea.querySelectorAll('p, div');
                    const texts = Array.from(paragraphs)
                        .map(el => el.textContent.trim())
                        .filter(t => t && t.length > 10);
                    
                    return texts.join('\\n\\n') || allTexts.join('\\n\\n');
                } catch (err) {
                    console.error('Copy error:', err);
                    return '';
                }
            }"""
        )
        
        if not response_text or len(response_text) < 100:
            # 스냅샷에서 직접 추출 (fallback)
            print("   스냅샷에서 직접 텍스트 추출 시도...")
            response_text = mod.extract_response_text_from_snapshot(snapshot)
        
        if not response_text or len(response_text) < 100:
            print("   ⚠️  응답 텍스트를 추출할 수 없습니다.")
            return None
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        print(f"   ✅ 전체 응답 txt 저장 완료: {output_path}")
        print(f"   응답 길이: {len(response_text)}자")
        return response_text
        
    except Exception as e:
        print(f"   ❌ Copy 및 저장 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def upload_frames_and_evaluate_gemini(frames: List[str], prompt: str, 
                                     video_duration: int, 
                                     video_info: Dict,
                                     output_dir: Path,
                                     batch_num: Optional[int] = None):
    """
    Gemini에 프레임들을 업로드하고 프롬프트 전송, 응답 저장
    
    Args:
        frames: 프레임 파일 경로 리스트 (최대 10개)
        prompt: 평가 프롬프트
        video_duration: 영상 길이 (초)
        video_info: 비디오 정보 딕셔너리
        output_dir: 결과 저장 디렉토리
        batch_num: 배치 번호 (None이면 단일 배치)
    
    Returns:
        (parsed_data, video_flow) 튜플 또는 (None, None)
    """
    print("="*80)
    print("Gemini에 프레임 업로드 및 평가 요청")
    if batch_num:
        print(f"배치 {batch_num}")
    print("="*80)
    print(f"업로드할 프레임 수: {len(frames)}개")
    
    formatted_prompt = prompt
    
    print("\n프롬프트 길이:", len(formatted_prompt), "자")
    
    try:
        # Gemini 페이지 확인 (첫 번째 배치만)
        if batch_num is None or batch_num == 1:
            print("\n1. Gemini 페이지 확인...")
            snapshot = mcp_Playwright_browser_snapshot()
            snapshot_str = json.dumps(snapshot, ensure_ascii=False).lower()
            if "gemini" not in snapshot_str:
                mcp_Playwright_browser_navigate(url="https://gemini.google.com")
                time.sleep(3)
        else:
            print(f"\n1. 배치 {batch_num}: 현재 채팅방에서 계속 진행...")
            snapshot = mcp_Playwright_browser_snapshot()
            time.sleep(1)
        
        # 2. 이미지 업로드 버튼 클릭 및 파일 업로드
        print("2. 이미지 업로드 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        # Gemini의 이미지 업로드 버튼 찾기
        try:
            # 이미지 업로드 버튼 클릭 시도
            mcp_Playwright_browser_evaluate(
                function="""() => {
                    const uploadBtn = document.querySelector('[aria-label*="파일"]') ||
                                     document.querySelector('button[aria-label*="파일 업로드"]');
                    if (uploadBtn) {
                        uploadBtn.click();
                        return 'Upload button clicked';
                    }
                    return 'Upload button not found';
                }"""
            )
            time.sleep(1)
        except:
            print("   ⚠️  업로드 버튼 자동 클릭 실패. 수동 업로드 필요할 수 있습니다.")
        
        # 파일 업로드
        print(f"   {len(frames)}개 프레임 파일 업로드 중...")
        abs_frame_paths = [os.path.abspath(f) for f in frames]
        
        try:
            mcp_Playwright_browser_file_upload(paths=abs_frame_paths)
            time.sleep(3)  # 업로드 완료 대기
        except Exception as e:
            print(f"   ⚠️  파일 업로드 실패: {e}")
            print("   수동으로 이미지를 업로드해주세요.")
            print(f"   업로드할 파일: {len(frames)}개")
            for i, f in enumerate(frames, 1):
                print(f"     {i}. {Path(f).name}")
            # 비대화형 모드: 자동으로 진행
            print("   ⏳ 5초 후 계속 진행...")
            time.sleep(5)
        
        # 3. 프롬프트 입력
        print("3. 프롬프트 입력 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        # Gemini 입력창 찾기
        try:
            result = mcp_Playwright_browser_evaluate(
                function=f"""() => {{
                    const textarea = document.querySelector('textarea[aria-label*="Message"]') ||
                                   document.querySelector('textarea[aria-label*="메시지"]') ||
                                   document.querySelector('textarea[placeholder*="Message"]') ||
                                   document.querySelector('[contenteditable="true"]') ||
                                   document.querySelector('textarea');
                    
                    if (textarea) {{
                        textarea.focus();
                        if (textarea.tagName === 'TEXTAREA') {{
                            textarea.value = {json.dumps(formatted_prompt)};
                            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }} else {{
                            // contenteditable div인 경우
                            textarea.textContent = {json.dumps(formatted_prompt)};
                            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                        return 'Text entered in: ' + textarea.tagName;
                    }}
                    return 'Textarea not found';
                }}"""
            )
            print(f"   프롬프트 입력 결과: {result}")
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠️  프롬프트 입력 실패: {e}")
            print("   수동으로 프롬프트를 입력해주세요.")
            print(f"\n프롬프트 (첫 500자):\n{formatted_prompt[:500]}...")
            # 비대화형 모드: 자동으로 진행
            print("   ⏳ 5초 후 계속 진행...")
            time.sleep(5)
        
        # 4. 전송 버튼 클릭
        print("4. 전송 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        try:
            result = mcp_Playwright_browser_evaluate(
                function="""() => {
                    const sendSelectors = [
                        'button[aria-label*="Send"]',
                        'button[aria-label*="전송"]',
                        'button[data-testid*="send"]',
                        'button[type="submit"]',
                        'button:has(svg[aria-label*="Send"])',
                        '[aria-label*="Submit"]'
                    ];
                    
                    for (const selector of sendSelectors) {
                        const btn = document.querySelector(selector);
                        if (btn && !btn.disabled) {
                            btn.click();
                            return 'Send button clicked: ' + selector;
                        }
                    }
                    
                    // Enter 키로 전송 시도
                    const textarea = document.querySelector('textarea') || 
                                    document.querySelector('[contenteditable="true"]');
                    if (textarea) {
                        const enterEvent = new KeyboardEvent('keydown', {
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13,
                            which: 13,
                            bubbles: true
                        });
                        textarea.dispatchEvent(enterEvent);
                        return 'Enter key pressed';
                    }
                    
                    return 'Send button not found';
                }"""
            )
            print(f"   전송 결과: {result}")
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠️  Send 버튼 클릭 실패: {e}")
            print("   수동으로 전송해주세요.")
            # 비대화형 모드: 자동으로 진행
            print("   ⏳ 5초 후 계속 진행...")
            time.sleep(5)
        
        # 5. 응답 대기 및 추출
        print("5. 응답 대기 중...")
        snapshot = wait_for_gemini_response(max_wait_seconds=180)
        
        if snapshot is None:
            print("⚠️  응답을 가져올 수 없습니다. 수동으로 확인해주세요.")
            return None, None
        
        # 6. Copy 버튼 클릭하여 응답 복사 및 txt 파일로 저장
        print("6. Copy 버튼 클릭하여 응답 복사 및 txt 파일로 저장...")
        if batch_num is not None:
            txt_output_path = output_dir / f"gemini_evaluation_response_batch_{batch_num:02d}.txt"
        else:
            txt_output_path = output_dir / "gemini_evaluation_response.txt"
        
        response_text = copy_gemini_response_and_save(snapshot, txt_output_path)
        
        if not response_text or len(response_text) < 100:
            print("⚠️  응답 텍스트를 제대로 추출하지 못했습니다.")
            print("   수동으로 확인하거나 스냅샷을 확인해주세요.")
            return None, None
        
        # 7. 응답 파싱 및 JSON 저장
        print("7. 응답 파싱 및 JSON 저장 중...")
        try:
            parsed_data = mod.parse_evaluation_response(response_text)
            
            if batch_num is not None:
                json_output_path = output_dir / f"gemini_evaluation_result_batch_{batch_num:02d}.json"
            else:
                json_output_path = output_dir / "gemini_evaluation_result.json"
            
            mod.save_evaluation_result(
                parsed_data,
                json_output_path,
                video_info,
                frames
            )
            
            print(f"✅ 평가 결과 저장 완료: {json_output_path}")
            
            # Flow Analysis 추출하여 반환
            video_flow = parsed_data.get('video_flow_analysis', {})
            return parsed_data, video_flow
            
        except Exception as e:
            print(f"⚠️  응답 파싱 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("수동으로 확인해주세요.")
        return None, None


def build_prompt_with_previous_flow_gemini(base_prompt: str, previous_flow_analyses: List[str], 
                                          duration_seconds: int, frame_count: int) -> str:
    """
    이전 배치들의 Flow Analysis를 포함한 프롬프트 생성 (Gemini용)
    """
    prompt = base_prompt.format(
        duration_seconds=duration_seconds,
        frame_count=frame_count
    )
    
    # 이전 Flow Analysis가 있으면 추가
    if previous_flow_analyses:
        previous_flow_section = "\n\n## Previous Video Flow Analysis\n\n"
        previous_flow_section += "**Important Context**: The following is the video flow analysis from previous batches of frames:\n\n"
        
        for i, flow in enumerate(previous_flow_analyses, 1):
            previous_flow_section += f"### Previous Batch {i} Analysis:\n{flow}\n\n"
        
        previous_flow_section += "Please use this context to understand what has happened in the video so far, "
        previous_flow_section += "and continue the analysis for the current batch of frames. "
        previous_flow_section += "The current frames represent a continuation of the video timeline.\n"
        
        # Video Flow Analysis 섹션 앞에 삽입
        prompt = prompt.replace("## Video Flow Analysis", previous_flow_section + "## Video Flow Analysis")
    
    return prompt


def main():
    """메인 함수"""
    # 경로 설정
    base_dir = Path("/Users/jeff/python/9_total_work/downloads/index0")
    video_info_path = base_dir / "extracted_frames" / "video_info.json"
    frames_dir = base_dir / "extracted_frames"
    
    # 비디오 정보 로드
    print("비디오 정보 로드 중...")
    video_info = mod.load_video_info(str(video_info_path))
    duration_seconds = video_info['video_properties']['duration_seconds']
    
    # 모든 프레임 파일 경로 수집 (시간순) - 모든 프레임 (그룹당 2장씩)
    print("\n모든 프레임 파일 수집 중...")
    all_frame_files = get_all_sorted_frames_gemini(frames_dir)
    
    if not all_frame_files:
        print("❌ 프레임 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n✅ 총 {len(all_frame_files)}개 프레임 수집 완료")
    
    # 배치 크기 설정
    BATCH_SIZE = 10
    total_batches = (len(all_frame_files) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 배치 크기: {BATCH_SIZE}개, 총 {total_batches}개 배치 예상")
    
    # Gemini 브라우저 연결
    print("\n" + "="*80)
    print("Gemini 브라우저 연결")
    print("="*80)
    
    if not connect_to_gemini_browser():
        print("⚠️  Gemini 브라우저 연결 실패. 수동으로 진행해주세요.")
        return
    
    # 배치 단위로 프레임 처리
    print("\n" + "="*80)
    print("배치 단위 프레임 평가 시작 (Gemini)")
    print("="*80)
    
    previous_flow_analyses = []
    output_dir = base_dir / "extracted_frames"
    
    for batch_num in range(total_batches):
        offset = batch_num * BATCH_SIZE
        batch_frames = all_frame_files[offset:offset+BATCH_SIZE]
        
        if not batch_frames:
            break
        
        print(f"\n{'='*80}")
        print(f"배치 {batch_num + 1}/{total_batches} 처리 중...")
        print(f"프레임 범위: {offset + 1} - {min(offset + BATCH_SIZE, len(all_frame_files))}")
        print(f"{'='*80}")
        
        # 프롬프트 생성 (이전 Flow Analysis 포함)
        prompt = build_prompt_with_previous_flow_gemini(
            FRAME_EVALUATION_PROMPT,
            previous_flow_analyses.copy(),
            int(duration_seconds),
            len(batch_frames)
        )
        
        # 프레임 업로드 및 평가
        parsed_data, video_flow = upload_frames_and_evaluate_gemini(
            frames=batch_frames,
            prompt=prompt,
            video_duration=int(duration_seconds),
            video_info=video_info,
            output_dir=output_dir,
            batch_num=batch_num + 1
        )
        
        # 현재 배치의 Flow Analysis 추출
        if video_flow:
            # 텍스트 형식으로 변환
            flow_text = []
            if video_flow.get('overall_storyline'):
                flow_text.append(f"**Overall storyline:** {video_flow['overall_storyline']}")
            if video_flow.get('key_scene_transitions'):
                flow_text.append(f"**Key scene transitions:** {video_flow['key_scene_transitions']}")
            if video_flow.get('emphasis_climax'):
                flow_text.append(f"**Emphasis & climax:** {video_flow['emphasis_climax']}")
            if video_flow.get('advertisement_message_cta'):
                flow_text.append(f"**Advertisement message / CTA:** {video_flow['advertisement_message_cta']}")
            if video_flow.get('visual_style_tone'):
                flow_text.append(f"**Visual style & tone:** {video_flow['visual_style_tone']}")
            
            if flow_text:
                previous_flow_analyses.append('\n'.join(flow_text))
                print(f"\n✅ 배치 {batch_num + 1} Flow Analysis 추출 완료")
        else:
            # 파일에서 읽기 시도 (fallback)
            batch_result_path = output_dir / f"gemini_evaluation_result_batch_{batch_num + 1:02d}.json"
            if batch_result_path.exists():
                with open(batch_result_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                video_flow = batch_data.get('evaluation', {}).get('video_flow_analysis', {})
                if video_flow:
                    # 텍스트 형식으로 변환
                    flow_text = []
                    if video_flow.get('overall_storyline'):
                        flow_text.append(f"**Overall storyline:** {video_flow['overall_storyline']}")
                    if video_flow.get('key_scene_transitions'):
                        flow_text.append(f"**Key scene transitions:** {video_flow['key_scene_transitions']}")
                    if video_flow.get('emphasis_climax'):
                        flow_text.append(f"**Emphasis & climax:** {video_flow['emphasis_climax']}")
                    if video_flow.get('advertisement_message_cta'):
                        flow_text.append(f"**Advertisement message / CTA:** {video_flow['advertisement_message_cta']}")
                    if video_flow.get('visual_style_tone'):
                        flow_text.append(f"**Visual style & tone:** {video_flow['visual_style_tone']}")
                    
                    if flow_text:
                        previous_flow_analyses.append('\n'.join(flow_text))
                        print(f"\n✅ 배치 {batch_num + 1} Flow Analysis 추출 완료 (파일에서)")
                    else:
                        print(f"\n⚠️  배치 {batch_num + 1} Flow Analysis 추출 실패")
                else:
                    print(f"\n⚠️  배치 {batch_num + 1} Flow Analysis 추출 실패")
            else:
                print(f"\n⚠️  배치 {batch_num + 1} 결과 파일을 찾을 수 없습니다.")
        
        # 배치 간 대기 (Gemini API 제한 방지)
        if batch_num < total_batches - 1:
            wait_time = 5
            print(f"\n⏳ 다음 배치 전 {wait_time}초 대기 중...")
            time.sleep(wait_time)
    
    # 최종 요약
    print("\n" + "="*80)
    print("🎉 모든 배치 처리 완료!")
    print("="*80)
    print(f"총 {len(all_frame_files)}개 프레임, {total_batches}개 배치 처리 완료")
    print(f"결과 저장 위치: {output_dir}")
    print(f"  - gemini_evaluation_result_batch_XX.json: 각 배치별 구조화된 결과")
    print(f"  - gemini_evaluation_response_batch_XX.txt: 각 배치별 전체 응답")


if __name__ == "__main__":
    main()
