'''
ChatGPT 웹 자동화를 통한 프레임 평가
Playwright MCP를 사용하여 ChatGPT에 프롬프트와 이미지를 업로드하고 평가받기

로그인 방법:
1. 자동 로그인: .env 파일에 CHATGPT_EMAIL과 CHATGPT_PASSWORD 설정
2. 수동 로그인: 스크립트 실행 후 브라우저에서 수동 로그인, 이후 자동화 진행
   (쿠키가 저장되어 있으면 자동으로 로그인 상태 유지)
'''

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. pip install python-dotenv")

# 프롬프트 임포트
import sys
import importlib.util
sys.path.append('/Users/jeff/python/9_total_work')

try:
    # 숫자로 시작하는 모듈명은 직접 import할 수 없으므로 importlib 사용
    spec = importlib.util.spec_from_file_location(
        "llm_mcp_eng",
        "/Users/jeff/python/9_total_work/4.LLM_MCP_ENG.py"
    )
    llm_mcp_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_mcp_eng)
    FRAME_EVALUATION_PROMPT = llm_mcp_eng.FRAME_EVALUATION_PROMPT
except (ImportError, FileNotFoundError, AttributeError) as e:
    # 직접 정의
    print(f"⚠️  프롬프트 파일을 로드할 수 없습니다: {e}")
    FRAME_EVALUATION_PROMPT = """You are an expert evaluator..."""  # 임시


def load_video_info(video_info_path: str) -> Dict:
    """비디오 정보 JSON 파일 로드"""
    with open(video_info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_sorted_frames(frames_dir: Path) -> List[str]:
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


def check_chatgpt_login_status(page_snapshot: Dict) -> bool:
    """ChatGPT 로그인 상태 확인"""
    # 로그인 버튼이나 채팅 입력창이 있는지 확인
    login_indicators = [
        '로그인', 'Login', 'Sign in', '로그인하기'
    ]
    
    # 채팅 입력창이 있으면 로그인된 것으로 간주
    chat_indicators = [
        '메시지', 'message', 'chat', '채팅', '입력', 'Send'
    ]
    
    snapshot_text = json.dumps(page_snapshot, ensure_ascii=False).lower()
    
    # 로그인 필요 요소가 있으면 False
    if any(indicator.lower() in snapshot_text for indicator in login_indicators):
        if 'input' not in snapshot_text and 'textarea' not in snapshot_text:
            return False
    
    # 채팅 입력창이 있으면 True
    return True


def connect_to_existing_chrome():
    """
    기존에 실행 중인 Chrome 브라우저에 연결
    또는 기존 Chrome 프로필을 사용하여 새 브라우저 인스턴스 시작
    
    방법:
    1. Chrome을 디버깅 모드로 실행: 
       /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
    2. Playwright가 해당 포트로 연결
    3. 또는 Chrome 프로필 경로를 사용하여 시작
    """
    print("="*80)
    print("기존 Chrome 브라우저 연결")
    print("="*80)
    
    print("\n📌 방법 1: Chrome 디버깅 모드 사용 (권장)")
    print("- Chrome을 디버깅 모드로 실행하세요:")
    print("  macOS: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222")
    print("- 이렇게 실행하면 이미 로그인된 상태의 Chrome을 사용할 수 있습니다.")
    print("- 위 명령어로 Chrome을 실행한 후, Enter 키를 눌러주세요.")
    
    input("\nChrome이 디버깅 모드로 실행되었으면 Enter를 눌러주세요...")
    
    try:
        # Playwright MCP가 기존 Chrome에 연결
        # 실제로는 Playwright MCP 설정에서 연결하거나
        # Chrome 프로필 경로를 사용
        print("\n✅ Chrome 연결 완료")
        print("   (실제 연결은 Playwright MCP 설정에서 처리)")
        return True
    except Exception as e:
        print(f"❌ Chrome 연결 실패: {e}")
        return False


def get_chrome_profile_path() -> str:
    """
    Chrome 프로필 경로 반환
    macOS 기본 위치
    """
    import platform
    
    system = platform.system()
    home = os.path.expanduser("~")
    
    if system == "Darwin":  # macOS
        # Chrome 기본 프로필 경로
        # 실제 프로필 이름은 "Default", "Profile 1" 등일 수 있음
        chrome_profile_base = os.path.join(
            home, 
            "Library/Application Support/Google/Chrome"
        )
        
        # Default 프로필 경로
        default_profile = os.path.join(chrome_profile_base, "Default")
        
        if os.path.exists(default_profile):
            return default_profile
        else:
            # Profile 1, Profile 2 등 찾기
            profiles = [d for d in os.listdir(chrome_profile_base) 
                       if d.startswith("Profile") and os.path.isdir(os.path.join(chrome_profile_base, d))]
            if profiles:
                return os.path.join(chrome_profile_base, sorted(profiles)[0])
            return chrome_profile_base
    else:
        # Windows나 Linux 경로
        return os.path.join(home, ".config/google-chrome")


def check_if_logged_in() -> bool:
    """
    Playwright MCP를 사용하여 현재 ChatGPT 로그인 상태 확인
    """
    print("ChatGPT 로그인 상태 확인 중...")
    
    try:
        # ChatGPT 메인 페이지로 이동
        # mcp_Playwright_browser_navigate(url="https://chat.openai.com")
        
        # 페이지 스냅샷으로 로그인 상태 확인
        # snapshot = mcp_Playwright_browser_snapshot()
        
        # 로그인 버튼이나 채팅 입력창 존재 여부 확인
        # if "Login" in snapshot or "로그인" in snapshot:
        #     return False
        
        # 실제로는 Playwright MCP 도구를 호출해야 함
        print("⚠️  Playwright MCP 호출 필요 - 현재는 플레이스홀더")
        return False
        
    except Exception as e:
        print(f"❌ 로그인 상태 확인 중 오류: {e}")
        return False


def login_to_chatgpt_manual():
    """
    수동 로그인 안내
    사용자가 브라우저에서 직접 로그인하도록 안내
    """
    print("="*80)
    print("📌 수동 로그인 모드")
    print("="*80)
    print("1. 브라우저가 열리면 ChatGPT에 수동으로 로그인해주세요")
    print("2. 로그인이 완료되면 Enter 키를 눌러주세요")
    print("3. 이후 자동화가 계속 진행됩니다")
    print("="*80)
    
    input("로그인 완료 후 Enter를 눌러주세요...")
    print("✅ 수동 로그인 완료 확인")


def login_to_chatgpt_auto(email: str, password: str):
    """
    ChatGPT 자동 로그인 시도
    Playwright MCP를 사용
    
    ⚠️ 주의: ChatGPT는 CAPTCHA나 2FA가 있을 수 있어
    자동 로그인이 실패할 수 있습니다.
    이 경우 수동 로그인을 사용하세요.
    """
    print("="*80)
    print("ChatGPT 자동 로그인 시도")
    print("="*80)
    
    try:
        # 1. ChatGPT 로그인 페이지로 이동
        print("1. ChatGPT 로그인 페이지로 이동...")
        # mcp_Playwright_browser_navigate(url="https://chat.openai.com/auth/login")
        # time.sleep(2)
        
        # 2. 페이지 스냅샷 확인
        print("2. 페이지 상태 확인...")
        # snapshot = mcp_Playwright_browser_snapshot()
        
        # 3. 로그인 버튼 찾기 및 클릭
        print("3. 로그인 버튼 클릭...")
        # mcp_Playwright_browser_click(element="로그인 버튼", ref="...")
        # time.sleep(2)
        
        # 4. 이메일 입력
        print("4. 이메일 입력...")
        # mcp_Playwright_browser_type(element="이메일 입력창", ref="...", text=email)
        
        # 5. Continue 버튼 클릭
        # mcp_Playwright_browser_click(element="Continue 버튼", ref="...")
        # time.sleep(2)
        
        # 6. 비밀번호 입력
        print("5. 비밀번호 입력...")
        # mcp_Playwright_browser_type(element="비밀번호 입력창", ref="...", text=password)
        
        # 7. 로그인 버튼 클릭
        print("6. 로그인 버튼 클릭...")
        # mcp_Playwright_browser_click(element="로그인 제출 버튼", ref="...")
        
        # 8. 로그인 완료 대기 (CAPTCHA 등 확인)
        print("7. 로그인 완료 대기...")
        time.sleep(5)
        
        print("✅ 자동 로그인 완료 (또는 수동 확인 필요)")
        
    except Exception as e:
        print(f"❌ 자동 로그인 실패: {e}")
        print("수동 로그인으로 전환합니다.")
        return False
    
    return True


def extract_response_text_from_snapshot(snapshot: Dict) -> str:
    """
    스냅샷에서 ChatGPT 응답 텍스트 추출
    """
    def extract_text_recursive(node):
        """재귀적으로 텍스트 추출"""
        text_parts = []
        
        if isinstance(node, dict):
            # 텍스트 필드가 있으면 추가
            if 'text' in node:
                text_parts.append(node['text'])
            # 자식 요소 재귀 처리
            for value in node.values():
                if isinstance(value, (dict, list)):
                    text_parts.extend(extract_text_recursive(value))
        elif isinstance(node, list):
            for item in node:
                text_parts.extend(extract_text_recursive(item))
        
        return text_parts
    
    all_texts = extract_text_recursive(snapshot)
    return '\n'.join(filter(None, all_texts))


def copy_chatgpt_response_and_save(snapshot: Dict, output_path: Path) -> Optional[str]:
    """
    ChatGPT 응답의 Copy 버튼을 클릭하여 클립보드에 복사하고,
    텍스트를 추출하여 txt 파일로 저장
    
    실제 Playwright MCP를 사용하여 Copy 버튼 클릭 후 클립보드에서 읽기
    
    Args:
        snapshot: 현재 페이지 스냅샷
        output_path: 저장할 txt 파일 경로
        
    Returns:
        추출된 텍스트 또는 None
    """
    try:
        # 1. Copy 버튼 찾기 (ChatGPT 응답 영역의 Copy 버튼)
        # 스냅샷에서 "Copy" 버튼 ref 찾기
        copy_button_ref = None
        
        def find_copy_button_ref(node):
            """재귀적으로 Copy 버튼 ref 찾기"""
            if isinstance(node, dict):
                # 버튼이고 텍스트가 "Copy"인 경우
                if 'button' in str(node.get('type', '')).lower() or 'Copy' in str(node.get('text', '')):
                    if 'Copy' in str(node.get('text', '')) or 'copy' in str(node).lower():
                        if 'ref' in node:
                            return node['ref']
                # 자식 요소 재귀 처리
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        result = find_copy_button_ref(value)
                        if result:
                            return result
            elif isinstance(node, list):
                for item in node:
                    result = find_copy_button_ref(item)
                    if result:
                        return result
            return None
        
        # Copy 버튼 ref 찾기
        copy_button_ref = find_copy_button_ref(snapshot)
        
        response_text = None
        
        # 2. Copy 버튼 클릭 시도 (실제 MCP 사용 가능한 경우)
        if copy_button_ref:
            print(f"   Copy 버튼 찾음 (ref: {copy_button_ref})")
            try:
                # 실제 MCP 호출
                # from mcp import mcp_Playwright_browser_click, mcp_Playwright_browser_evaluate
                # mcp_Playwright_browser_click(element="Copy", ref=copy_button_ref)
                # time.sleep(0.5)  # 복사 완료 대기
                
                # 클립보드에서 텍스트 읽기
                # response_text = mcp_Playwright_browser_evaluate(
                #     function="async () => await navigator.clipboard.readText()"
                # )
                print("   ⚠️  실제 MCP 호출은 주석 해제 필요")
            except Exception as e:
                print(f"   ⚠️  Copy 버튼 클릭 실패: {e}")
        else:
            print("   ⚠️  Copy 버튼을 찾을 수 없습니다. 스냅샷에서 직접 추출 시도...")
        
        # 3. JavaScript로 직접 텍스트 추출 (Copy 버튼이 없거나 실패한 경우)
        if not response_text:
            print("   JavaScript로 응답 텍스트 직접 추출 중...")
            try:
                # 실제 MCP 사용 시:
                # response_text = mcp_Playwright_browser_evaluate(
                #     function="""async () => {
                #         try {
                #             // 먼저 Copy 버튼 클릭 시도
                #             const copyBtn = document.querySelector('[data-testid="copy-turn-action-button"]');
                #             if (copyBtn) {
                #                 copyBtn.click();
                #                 await new Promise(resolve => setTimeout(resolve, 500));
                #             }
                #             // 클립보드에서 텍스트 읽기
                #             const text = await navigator.clipboard.readText();
                #             return text;
                #         } catch (err) {
                #             // 클립보드 접근이 실패하면 ChatGPT 응답 영역에서 직접 추출
                #             const articles = document.querySelectorAll('article');
                #             if (articles.length < 2) return '';
                #             const responseArticle = articles[1]; // 두 번째 article이 ChatGPT 응답
                #             const textElements = responseArticle.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, strong');
                #             const texts = Array.from(textElements).map(el => el.textContent.trim()).filter(t => t);
                #             return texts.join('\\n\\n');
                #         }
                #     }"""
                # )
                
                # 현재는 스냅샷에서 직접 추출
                response_text = extract_response_text_from_snapshot(snapshot)
            except Exception as e:
                print(f"   ⚠️  텍스트 추출 실패: {e}")
                return None
        
        if not response_text:
            print("   ⚠️  응답 텍스트를 추출할 수 없습니다.")
            return None
        
        # 4. txt 파일로 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        
        print(f"   ✅ 전체 응답 txt 저장 완료: {output_path}")
        return response_text
        
    except Exception as e:
        print(f"   ❌ Copy 및 저장 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_previous_flow_analysis_from_json(result_json_path: Path) -> Optional[str]:
    """
    이전 배치의 evaluation_result.json에서 Video Flow Analysis 추출
    
    Args:
        result_json_path: evaluation_result.json 파일 경로
        
    Returns:
        Video Flow Analysis 텍스트 또는 None
    """
    try:
        if not result_json_path.exists():
            return None
        
        with open(result_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_flow = data.get('evaluation', {}).get('video_flow_analysis', {})
        if not video_flow:
            return None
        
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
        
        return '\n'.join(flow_text) if flow_text else None
        
    except Exception as e:
        print(f"   ⚠️  이전 Flow Analysis 추출 실패: {e}")
        return None


def build_prompt_with_previous_flow(base_prompt: str, previous_flow_analyses: List[str], 
                                    duration_seconds: int, frame_count: int) -> str:
    """
    이전 배치들의 Flow Analysis를 포함한 프롬프트 생성
    
    Args:
        base_prompt: 기본 프롬프트 템플릿
        previous_flow_analyses: 이전 배치들의 Flow Analysis 리스트
        duration_seconds: 영상 길이 (초)
        frame_count: 현재 배치의 프레임 수
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


def parse_evaluation_response(response_text: str) -> Dict:
    """
    ChatGPT 평가 응답을 구조화된 데이터로 파싱
    """
    result = {
        'frames': [],
        'video_flow_analysis': {}
    }
    
    # Frame 평가 파싱
    frame_pattern = r'Frame (\d+) \(Time: ([\d.]+) seconds?\)'
    frames_data = re.finditer(frame_pattern, response_text)
    
    for match in frames_data:
        frame_num = int(match.group(1))
        frame_time = float(match.group(2))
        
        # 해당 프레임 다음부터 다음 프레임 전까지 텍스트 추출
        start_pos = match.end()
        next_match = None
        for next_match_obj in re.finditer(frame_pattern, response_text):
            if next_match_obj.start() > start_pos:
                next_match = next_match_obj
                break
        
        if next_match:
            frame_text = response_text[start_pos:next_match.start()]
        else:
            # 마지막 프레임이면 Video Flow Analysis 전까지
            video_flow_start = response_text.find('Video Flow Analysis')
            if video_flow_start > 0:
                frame_text = response_text[start_pos:video_flow_start]
            else:
                frame_text = response_text[start_pos:]
        
        # 각 평가 항목 추출
        frame_data = {
            'frame_number': frame_num,
            'time_seconds': frame_time,
            'scores': {},
            'total_score': None,
            'comments': ''
        }
        
        # 점수 추출
        score_patterns = {
            'content_representativeness': r'Content Representativeness & Composition:\s*(\d+)/10',
            'editing_complexity': r'Editing Complexity & Technicality:\s*(\d+)/10',
            'sharpness_focus': r'Sharpness & Focus:\s*(\d+)/10',
            'exposure_lighting': r'Exposure & Lighting:\s*(\d+)/10',
            'contrast_visual_impact': r'Contrast & Visual Impact:\s*(\d+)/10',
            'color_richness_tone': r'Color Richness & Tone:\s*(\d+)/10',
        }
        
        for key, pattern in score_patterns.items():
            match_score = re.search(pattern, frame_text)
            if match_score:
                frame_data['scores'][key] = int(match_score.group(1))
        
        # 총점 추출
        total_match = re.search(r'Total Score:\s*(\d+)/60', frame_text)
        if total_match:
            frame_data['total_score'] = int(total_match.group(1))
        
        # 코멘트 추출
        comments_match = re.search(r'Brief Evaluation Comments?:?\s*(.+?)(?=\n\n|\n###|Video Flow|$)', frame_text, re.DOTALL)
        if comments_match:
            frame_data['comments'] = comments_match.group(1).strip()
        
        result['frames'].append(frame_data)
    
    # Video Flow Analysis 파싱
    video_flow_patterns = {
        'overall_storyline': r'Overall storyline:?\s*(.+?)(?=\n-|\n\n|Key scene|$)',
        'key_scene_transitions': r'Key scene transitions?:?\s*(.+?)(?=\n-|\n\n|Emphasis|$)',
        'emphasis_climax': r'Emphasis & climax:?\s*(.+?)(?=\n-|\n\n|Advertisement|$)',
        'advertisement_message_cta': r'Advertisement message / CTA:?\s*(.+?)(?=\n-|\n\n|Visual style|$)',
        'visual_style_tone': r'Visual style & tone:?\s*(.+?)(?=\n\n|$)',
    }
    
    for key, pattern in video_flow_patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            result['video_flow_analysis'][key] = match.group(1).strip()
    
    return result


def save_evaluation_result(evaluation_data: Dict, output_path: Path, 
                          video_info: Dict, frames: List[str]):
    """
    평가 결과를 JSON 파일로 저장
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 저장할 데이터 구조
    result_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'video_info': video_info,
            'frames_count': len(frames),
            'frames_files': [Path(f).name for f in frames]
        },
        'evaluation': evaluation_data,
        'raw_response': evaluation_data.get('raw_response', '')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 평가 결과 저장 완료: {output_path}")


def wait_for_chatgpt_response(max_wait_seconds: int = 120) -> Optional[Dict]:
    """
    ChatGPT 응답이 완료될 때까지 대기하고 응답 추출
    
    Returns:
        스냅샷 데이터 또는 None
    """
    print(f"\n응답 완료 대기 중... (최대 {max_wait_seconds}초)")
    
    try:
        # "Frame 1" 텍스트가 나타날 때까지 대기
        mcp_Playwright_browser_wait_for(text="Frame 1", time=max_wait_seconds)
        
        # 응답 생성 중인지 확인 (Stop streaming 버튼이 사라질 때까지)
        for i in range(max_wait_seconds // 5):
            snapshot = mcp_Playwright_browser_snapshot()
            snapshot_str = json.dumps(snapshot, ensure_ascii=False).lower()
            
            # "ChatGPT is generating" 또는 "Stop streaming" 버튼이 없으면 완료
            if "generating" not in snapshot_str and "stop streaming" not in snapshot_str:
                if "frame 1" in snapshot_str.lower() or "frame 1 (time:" in snapshot_str.lower():
                    print("✅ 응답 완료 확인")
                    return snapshot
            
            if i < (max_wait_seconds // 5) - 1:  # 마지막 반복이 아니면
                print(f"   대기 중... ({i*5}초 경과)")
                time.sleep(5)
        
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


def upload_frames_and_evaluate(frames: List[str], prompt: str, 
                               video_duration: int, 
                               video_info: Dict,
                               output_dir: Path,
                               batch_num: Optional[int] = None):
    """
    ChatGPT에 프레임들을 업로드하고 프롬프트 전송, 응답 저장
    
    Args:
        frames: 프레임 파일 경로 리스트 (최대 10개)
        prompt: 평가 프롬프트
        video_duration: 영상 길이 (초)
        video_info: 비디오 정보 딕셔너리
        output_dir: 결과 저장 디렉토리
        batch_num: 배치 번호 (None이면 단일 배치)
    """
    print("="*80)
    print("ChatGPT에 프레임 업로드 및 평가 요청")
    if batch_num:
        print(f"배치 {batch_num}")
    print("="*80)
    print(f"업로드할 프레임 수: {len(frames)}개")
    
    # 프롬프트는 이미 포맷팅되어 있음 (build_prompt_with_previous_flow에서)
    formatted_prompt = prompt
    
    print("\n프롬프트 길이:", len(formatted_prompt), "자")
    # 이전 Flow Analysis가 프롬프트에 포함되어 있는지 확인
    if "Previous Video Flow Analysis" in formatted_prompt:
        prev_count = formatted_prompt.count("Previous Batch")
        if prev_count > 0:
            print(f"이전 Flow Analysis 포함: {prev_count}개 배치")
    print("\n프레임 목록:")
    for i, frame_path in enumerate(frames, 1):
        print(f"  {i}. {Path(frame_path).name}")
    
    try:
        # 실제 Playwright MCP 호출 코드
        
        # 1. ChatGPT 채팅 페이지로 이동 (첫 번째 배치만 새 대화 시작)
        if batch_num is None or batch_num == 1:
            print("\n1. ChatGPT 채팅 페이지로 이동 (새 대화 시작)...")
            mcp_Playwright_browser_navigate(url="https://chat.openai.com")
            time.sleep(3)
            
            # 새 대화 시작 (필요한 경우)
            snapshot = mcp_Playwright_browser_snapshot()
            # "New chat" 버튼이 있으면 클릭
            try:
                # 새 대화 버튼 찾기 시도
                new_chat_ref = None
                snapshot_str = json.dumps(snapshot, ensure_ascii=False).lower()
                if "new chat" in snapshot_str:
                    # 새 대화 링크 찾기
                    mcp_Playwright_browser_navigate(url="https://chat.openai.com")
                    time.sleep(2)
            except:
                pass
        else:
            # 배치 2 이상: 현재 채팅방에서 계속 진행
            print(f"\n1. 배치 {batch_num}: 현재 채팅방에서 계속 진행...")
            # 페이지가 이미 로드되어 있으므로 스냅샷만 확인
            snapshot = mcp_Playwright_browser_snapshot()
            time.sleep(1)
        
        # 2. 이미지 업로드 버튼 클릭 및 파일 업로드
        print("2. 이미지 업로드 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        # "Add files and more" 버튼 찾기
        add_files_ref = None
        for item in snapshot.get('snapshot', []):
            if isinstance(item, dict):
                if 'button' in str(item.get('type', '')).lower() and 'Add files' in str(item.get('text', '')):
                    if 'ref' in item:
                        add_files_ref = item['ref']
                        break
        
        if not add_files_ref:
            # 스냅샷에서 직접 찾기
            snapshot_str = json.dumps(snapshot, ensure_ascii=False)
            # ref를 찾기 위해 스냅샷 구조 분석
            try:
                # "Add files and more" 버튼 클릭
                mcp_Playwright_browser_click(element="Add files and more", ref="e102")
                time.sleep(1)
            except:
                print("   ⚠️  파일 업로드 버튼을 찾을 수 없습니다. 스냅샷 확인 필요")
        
        # 파일 업로드
        print(f"   {len(frames)}개 프레임 파일 업로드 중...")
        abs_frame_paths = [os.path.abspath(f) for f in frames]
        mcp_Playwright_browser_file_upload(paths=abs_frame_paths)
        time.sleep(3)  # 업로드 완료 대기
        
        # 3. 프롬프트 입력
        print("3. 프롬프트 입력 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        # 채팅 입력창 찾기
        textbox_ref = None
        for item in snapshot.get('snapshot', []):
            if isinstance(item, dict):
                if 'textbox' in str(item.get('type', '')).lower() or 'Ask anything' in str(item.get('text', '')):
                    if 'ref' in item:
                        textbox_ref = item['ref']
                        break
        
        if not textbox_ref:
            # 기본 ref 시도
            try:
                mcp_Playwright_browser_type(element="채팅 입력창", ref="e99", text=formatted_prompt, slowly=False)
            except:
                print("   ⚠️  입력창을 찾을 수 없습니다. 스냅샷 확인 필요")
        else:
            mcp_Playwright_browser_type(element="채팅 입력창", ref=textbox_ref, text=formatted_prompt, slowly=False)
        
        time.sleep(2)
        
        # 4. 전송 버튼 클릭
        print("4. 전송 중...")
        snapshot = mcp_Playwright_browser_snapshot()
        
        # Send 버튼 찾기
        send_button_ref = None
        snapshot_str = json.dumps(snapshot, ensure_ascii=False).lower()
        if "send" in snapshot_str:
            # 페이지에서 Send 버튼 찾기
            try:
                # testid로 찾기
                mcp_Playwright_browser_evaluate(function="() => document.querySelector('[data-testid=\"send-button\"]')?.click()")
            except:
                try:
                    mcp_Playwright_browser_click(element="Send prompt", ref="e115")
                except:
                    print("   ⚠️  Send 버튼을 찾을 수 없습니다. 수동 전송 필요")
        
        time.sleep(2)
        
        # 5. 응답 대기 및 추출
        print("5. 응답 대기 중...")
        snapshot = wait_for_chatgpt_response(max_wait_seconds=180)
        
        if snapshot is None:
            print("⚠️  응답을 가져올 수 없습니다. 수동으로 확인해주세요.")
            return
        
        # 6. Copy 버튼 클릭하여 응답 복사 및 txt 파일로 저장
        print("6. Copy 버튼 클릭하여 응답 복사 및 txt 파일로 저장...")
        if batch_num is not None:
            txt_output_path = output_dir / f"evaluation_response_batch_{batch_num:02d}.txt"
        else:
            txt_output_path = output_dir / "evaluation_response.txt"
        response_text = copy_chatgpt_response_and_save(snapshot, txt_output_path)
        
        if not response_text or "Frame 1" not in response_text:
            print("⚠️  응답 텍스트를 제대로 추출하지 못했습니다.")
            print("   수동으로 확인하거나 스냅샷을 확인해주세요.")
            # 원본 스냅샷을 저장
            raw_output_path = output_dir / "evaluation_response_raw.json"
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            print(f"   원본 스냅샷 저장: {raw_output_path}")
            return
        
        # 9. 응답 파싱
        print("9. 응답 파싱 중...")
        evaluation_data = parse_evaluation_response(response_text)
        evaluation_data['raw_response'] = response_text  # 원본 응답도 저장
        
        # 10. 구조화된 결과 저장 (JSON)
        print("10. 구조화된 결과 저장 중...")
        if batch_num is not None:
            output_path = output_dir / f"evaluation_result_batch_{batch_num:02d}.json"
            txt_output_path = output_dir / f"evaluation_response_batch_{batch_num:02d}.txt"
        else:
            output_path = output_dir / "evaluation_result.json"
            txt_output_path = output_dir / "evaluation_response.txt"
        
        # txt 파일도 배치별로 저장
        if txt_output_path != output_dir / "evaluation_response.txt":
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"   ✅ 배치별 응답 txt 저장 완료: {txt_output_path}")
        
        save_evaluation_result(evaluation_data, output_path, video_info, frames)
        
        print("\n✅ 평가 완료 및 저장 완료!")
        print(f"   저장 위치: {output_path}")
        
        # 간단한 요약 출력
        if evaluation_data['frames']:
            print("\n📊 평가 요약:")
            total_scores = [f['total_score'] for f in evaluation_data['frames'] if f['total_score']]
            if total_scores:
                avg_score = sum(total_scores) / len(total_scores)
                print(f"   평균 점수: {avg_score:.1f}/60")
                print(f"   최고 점수: {max(total_scores)}/60 (Frame {total_scores.index(max(total_scores))+1})")
                print(f"   최저 점수: {min(total_scores)}/60 (Frame {total_scores.index(min(total_scores))+1})")
        
    except Exception as e:
        print(f"\n❌ 업로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("수동으로 확인해주세요.")


def main():
    """메인 함수"""
    # 경로 설정
    base_dir = Path("/Users/jeff/python/9_total_work/downloads/index0")
    video_info_path = base_dir / "extracted_frames" / "video_info.json"
    frames_dir = base_dir / "extracted_frames"
    
    # 비디오 정보 로드
    print("비디오 정보 로드 중...")
    video_info = load_video_info(str(video_info_path))
    duration_seconds = video_info['video_properties']['duration_seconds']
    
    # 모든 프레임 파일 경로 수집 (시간순)
    print("\n모든 프레임 파일 수집 중...")
    all_frame_files = get_all_sorted_frames(frames_dir)
    
    if not all_frame_files:
        print("❌ 프레임 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n✅ 총 {len(all_frame_files)}개 프레임 수집 완료")
    
    # 배치 크기 설정
    BATCH_SIZE = 10
    total_batches = (len(all_frame_files) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📦 배치 크기: {BATCH_SIZE}개, 총 {total_batches}개 배치 예상")
    
    # ChatGPT 로그인 확인 및 처리
    print("\n" + "="*80)
    print("ChatGPT 로그인 처리")
    print("="*80)
    
    print("\n📌 로그인 방식 선택:")
    print("1. 기존 Chrome 브라우저 사용 (이미 로그인된 Chrome - 권장)")
    print("2. 자동 로그인 시도 (.env 파일 필요)")
    print("3. 수동 로그인")
    
    choice = input("\n방식을 선택하세요 (1/2/3, 기본값: 1): ").strip() or "1"
    
    if choice == "1":
        # 기존 Chrome 브라우저 연결
        print("\n기존 Chrome 브라우저 연결 모드")
        chrome_profile_path = get_chrome_profile_path()
        print(f"\nChrome 프로필 경로: {chrome_profile_path}")
        print("\n⚠️  중요: Chrome을 디버깅 모드로 실행해야 합니다!")
        print("\n다음 명령어를 터미널에서 실행하세요:")
        print("/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222")
        print("\n또는 이미 실행 중인 Chrome이 있다면:")
        print("Chrome 주소창에 chrome://version 입력하여 프로필 경로 확인")
        
        connect_to_existing_chrome()
        
    elif choice == "2":
        # 자동 로그인 시도
        chatgpt_email = os.getenv('CHATGPT_EMAIL', '')
        chatgpt_password = os.getenv('CHATGPT_PASSWORD', '')
        
        if chatgpt_email and chatgpt_password:
            print("\n자동 로그인 시도 중...")
            success = login_to_chatgpt_auto(chatgpt_email, chatgpt_password)
            if not success:
                print("\n자동 로그인 실패. 수동 로그인으로 전환합니다.")
                login_to_chatgpt_manual()
        else:
            print("⚠️  환경 변수에 ChatGPT 로그인 정보가 없습니다.")
            print("   .env 파일에 다음을 추가하세요:")
            print("   CHATGPT_EMAIL=your_email@example.com")
            print("   CHATGPT_PASSWORD=your_password")
            login_to_chatgpt_manual()
    
    else:
        # 수동 로그인 모드
        login_to_chatgpt_manual()
    
    # 로그인 상태 최종 확인
    print("\n로그인 상태 최종 확인...")
    if check_if_logged_in():
        print("✅ 로그인 확인 완료")
    else:
        print("⚠️  로그인 상태를 확인할 수 없습니다. 계속 진행합니다.")
    
    # 배치 단위로 프레임 처리
    print("\n" + "="*80)
    print("배치 단위 프레임 평가 시작")
    print("="*80)
    
    previous_flow_analyses = []
    output_dir = base_dir / "extracted_frames"
    
    # 배치 1의 Flow Analysis가 이미 있으면 로드 (이전 실행에서 저장된 경우)
    batch1_result_path = output_dir / "evaluation_result.json"
    if batch1_result_path.exists():
        flow_analysis = extract_previous_flow_analysis_from_json(batch1_result_path)
        if flow_analysis:
            previous_flow_analyses.append(flow_analysis)
            print(f"✅ 배치 1 Flow Analysis 로드 완료")
    
    # 배치 2의 Flow Analysis도 있으면 로드 (이미 처리된 경우)
    batch2_result_path = output_dir / "evaluation_result_batch_02.json"
    if batch2_result_path.exists():
        flow_analysis = extract_previous_flow_analysis_from_json(batch2_result_path)
        if flow_analysis:
            previous_flow_analyses.append(flow_analysis)
            print(f"✅ 배치 2 Flow Analysis 로드 완료")
    
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
        prompt = build_prompt_with_previous_flow(
            FRAME_EVALUATION_PROMPT,
            previous_flow_analyses.copy(),
            int(duration_seconds),
            len(batch_frames)
        )
        
        # 프레임 업로드 및 평가
        upload_frames_and_evaluate(
            frames=batch_frames,
            prompt=prompt,
            video_duration=int(duration_seconds),
            video_info=video_info,
            output_dir=output_dir,
            batch_num=batch_num + 1
        )
        
        # 현재 배치의 Flow Analysis 추출
        batch_result_path = output_dir / f"evaluation_result_batch_{batch_num + 1:02d}.json"
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
                    print(f"\n✅ 배치 {batch_num + 1} Flow Analysis 추출 완료")
            else:
                print(f"\n⚠️  배치 {batch_num + 1} Flow Analysis 추출 실패")
        else:
            print(f"\n⚠️  배치 {batch_num + 1} 결과 파일을 찾을 수 없습니다.")
        
        # 배치 간 대기 (ChatGPT API 제한 방지)
        if batch_num < total_batches - 1:
            wait_time = 5  # 5초 대기
            print(f"\n⏳ 다음 배치 전 {wait_time}초 대기 중...")
            time.sleep(wait_time)
    
    # 최종 요약
    print("\n" + "="*80)
    print("🎉 모든 배치 처리 완료!")
    print("="*80)
    print(f"총 {len(all_frame_files)}개 프레임, {total_batches}개 배치 처리 완료")
    print(f"결과 저장 위치: {output_dir}")
    print(f"  - evaluation_result_batch_XX.json: 각 배치별 구조화된 결과")
    print(f"  - evaluation_response_batch_XX.txt: 각 배치별 전체 응답")


if __name__ == "__main__":
    main()

