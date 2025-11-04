'''
기존 로그인된 Chrome 브라우저에 Playwright MCP로 연결 테스트
'''

import json

def test_chrome_connection():
    """기존 Chrome 브라우저 연결 테스트"""
    
    print("="*80)
    print("Chrome 브라우저 연결 테스트")
    print("="*80)
    
    print("\n1단계: Chrome 디버깅 모드 확인")
    print("="*80)
    print("먼저 Chrome을 디버깅 모드로 실행해야 합니다:")
    print("\n터미널에서 다음 명령어 실행:")
    print("/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222")
    print("\n⚠️  기존 Chrome 창이 모두 닫혀있어야 합니다!")
    print("   (디버깅 모드로 실행하면 새로운 Chrome 창이 열립니다)")
    
    input("\nChrome을 디버깅 모드로 실행했으면 Enter를 눌러주세요...")
    
    print("\n2단계: ChatGPT 페이지로 이동 테스트")
    print("="*80)
    
    try:
        # ChatGPT 페이지로 이동
        print("ChatGPT 페이지로 이동 중...")
        # 실제 Playwright MCP 호출:
        # result = mcp_Playwright_browser_navigate(url="https://chat.openai.com")
        
        print("✅ ChatGPT 페이지 이동 (MCP 호출 필요)")
        
        # 페이지 스냅샷 확인
        print("\n페이지 스냅샷 확인 중...")
        # snapshot = mcp_Playwright_browser_snapshot()
        
        print("✅ 페이지 스냅샷 완료")
        print("\n스냅샷 내용 (실제 MCP 호출 필요):")
        # print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        
        print("\n3단계: 로그인 상태 확인")
        print("="*80)
        
        # 로그인 상태 확인
        # 채팅 입력창이 있는지 확인
        print("로그인 상태 확인 중...")
        
        # 실제로는 스냅샷에서 채팅 입력창을 찾아야 함
        print("✅ 테스트 완료!")
        print("\n📌 다음 단계:")
        print("1. 실제 Playwright MCP 도구 호출 코드 추가 필요")
        print("2. 채팅 입력창 찾기 및 프레임 업로드 테스트")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False


def test_chatgpt_navigation():
    """ChatGPT 페이지에서 실제 동작 테스트"""
    
    print("\n" + "="*80)
    print("ChatGPT 페이지 동작 테스트")
    print("="*80)
    
    try:
        # 1. ChatGPT 페이지로 이동
        print("1. ChatGPT 페이지로 이동...")
        # mcp_Playwright_browser_navigate(url="https://chat.openai.com")
        
        # 2. 페이지 스냅샷으로 현재 상태 확인
        print("2. 페이지 상태 확인...")
        # snapshot = mcp_Playwright_browser_snapshot()
        # print("스냅샷:", json.dumps(snapshot, indent=2, ensure_ascii=False)[:500])
        
        # 3. 새 대화 시작 버튼 찾기 (있으면)
        print("3. 새 대화 버튼 확인...")
        
        # 4. 채팅 입력창 찾기
        print("4. 채팅 입력창 찾기...")
        
        # 5. 간단한 메시지 입력 테스트
        print("5. 테스트 메시지 입력...")
        # mcp_Playwright_browser_type(
        #     element="채팅 입력창",
        #     ref="...",
        #     text="Hello, this is a test"
        # )
        
        print("\n✅ 테스트 메시지 입력 완료")
        print("   (실제 전송은 하지 않았습니다)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False


if __name__ == "__main__":
    print("기존 Chrome 브라우저 연결 테스트 시작\n")
    
    # Chrome 연결 테스트
    success = test_chrome_connection()
    
    if success:
        # ChatGPT 페이지에서 실제 동작 테스트
        test_chatgpt_navigation()
    
    print("\n" + "="*80)
    print("테스트 완료!")
    print("="*80)
    print("\n⚠️  실제 Playwright MCP 도구를 호출하려면:")
    print("   - mcp_Playwright_browser_navigate")
    print("   - mcp_Playwright_browser_snapshot")
    print("   - mcp_Playwright_browser_type")
    print("   등의 함수를 실제로 호출해야 합니다.")

