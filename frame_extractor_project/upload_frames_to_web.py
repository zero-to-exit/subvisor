"""
선택된 프레임들을 웹에 업로드하고 표시하는 서버
FastAPI를 사용하여 프레임 이미지와 선택 이유를 함께 표시
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
from typing import Dict, Optional

# 프레임 디렉토리 설정
FRAMES_DIR: Optional[Path] = None
REASONS_JSON: Optional[Path] = None

app = FastAPI(title="Selected Frames Display Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def set_frames_dir(frames_dir: Path):
    """
    프레임 디렉토리 설정
    
    Args:
        frames_dir: final_selected_frames 디렉토리 경로
    """
    global FRAMES_DIR, REASONS_JSON
    FRAMES_DIR = Path(frames_dir)
    REASONS_JSON = FRAMES_DIR / "reasons.json"
    
    if not FRAMES_DIR.exists():
        raise ValueError(f"Frames directory does not exist: {FRAMES_DIR}")


def load_reasons() -> Dict[str, str]:
    """
    reasons.json 파일을 로드
    
    Returns:
        프레임 번호를 키로, 이유를 값으로 하는 딕셔너리
    """
    if REASONS_JSON is None or not REASONS_JSON.exists():
        return {}
    
    try:
        with open(REASONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load reasons.json: {e}")
        return {}


def get_frame_files() -> list:
    """
    프레임 디렉토리에서 이미지 파일 목록 가져오기
    
    Returns:
        프레임 파일 경로 리스트 (정렬됨)
    """
    if FRAMES_DIR is None:
        return []
    
    frame_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        frame_files.extend(FRAMES_DIR.glob(f"*{ext}"))
    
    # 파일명의 숫자로 정렬
    frame_files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    return frame_files


@app.get("/", response_class=HTMLResponse)
async def frames_page():
    """프레임 표시 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>선택된 프레임</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                color: white;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .stats {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stats span {
                font-size: 1.2em;
                color: #667eea;
                font-weight: bold;
            }
            .frames-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }
            .frame-card {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .frame-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .frame-image {
                width: 100%;
                height: 300px;
                object-fit: cover;
                display: block;
            }
            .frame-info {
                padding: 20px;
            }
            .frame-number {
                font-size: 1.1em;
                color: #667eea;
                font-weight: bold;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            }
            .frame-number::before {
                content: "🎬";
                margin-right: 8px;
            }
            .frame-reason {
                color: #555;
                line-height: 1.6;
                font-size: 0.95em;
                white-space: pre-wrap;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .loading {
                text-align: center;
                color: white;
                font-size: 1.2em;
                padding: 50px;
            }
            .error {
                background: #ff6b6b;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }
            @media (max-width: 768px) {
                .frames-grid {
                    grid-template-columns: 1fr;
                }
                h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📸 선택된 프레임</h1>
            
            <div class="stats" id="stats">
                <span>로딩 중...</span>
            </div>
            
            <div id="framesContainer" class="loading">
                프레임을 불러오는 중...
            </div>
        </div>

        <script>
            async function loadFrames() {
                try {
                    const response = await fetch('/api/frames');
                    if (!response.ok) {
                        throw new Error('프레임을 불러오는데 실패했습니다.');
                    }
                    
                    const data = await response.json();
                    
                    // 통계 업데이트
                    document.getElementById('stats').innerHTML = 
                        `<span>총 ${data.frames.length}개의 프레임이 선택되었습니다</span>`;
                    
                    // 프레임 표시
                    const container = document.getElementById('framesContainer');
                    container.className = 'frames-grid';
                    
                    if (data.frames.length === 0) {
                        container.innerHTML = '<div class="error">표시할 프레임이 없습니다.</div>';
                        return;
                    }
                    
                    container.innerHTML = data.frames.map(frame => `
                        <div class="frame-card">
                            <img src="/api/frame/${frame.frame_number}" 
                                 alt="Frame ${frame.frame_number}" 
                                 class="frame-image"
                                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'300\\'%3E%3Crect fill=\\'%23ddd\\' width=\\'400\\' height=\\'300\\'/%3E%3Ctext fill=\\'%23999\\' font-family=\\'sans-serif\\' font-size=\\'18\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dominant-baseline=\\'middle\\'%3E이미지를 불러올 수 없습니다%3C/text%3E%3C/svg%3E';">
                            <div class="frame-info">
                                <div class="frame-number">Frame ${frame.frame_number}</div>
                                <div class="frame-reason">${frame.reason || '선택 이유가 없습니다.'}</div>
                            </div>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('framesContainer').innerHTML = 
                        `<div class="error">오류: ${error.message}</div>`;
                }
            }
            
            // 페이지 로드 시 프레임 불러오기
            loadFrames();
            
            // 5초마다 자동 새로고침 (선택사항)
            // setInterval(loadFrames, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/frames")
async def get_frames():
    """프레임 목록과 이유를 반환하는 API"""
    if FRAMES_DIR is None:
        raise HTTPException(status_code=500, detail="Frames directory not set")
    
    frame_files = get_frame_files()
    reasons = load_reasons()
    
    frames_data = []
    for frame_path in frame_files:
        frame_number = frame_path.stem
        reason = reasons.get(frame_number, "선택 이유가 제공되지 않았습니다.")
        
        frames_data.append({
            "frame_number": frame_number,
            "filename": frame_path.name,
            "reason": reason
        })
    
    return JSONResponse(content={
        "frames": frames_data,
        "total": len(frames_data)
    })


@app.get("/api/frame/{frame_number}")
async def get_frame_image(frame_number: str):
    """특정 프레임 이미지를 반환"""
    if FRAMES_DIR is None:
        raise HTTPException(status_code=500, detail="Frames directory not set")
    
    # 여러 확장자 시도
    for ext in ['.jpg', '.jpeg', '.png']:
        frame_path = FRAMES_DIR / f"{frame_number}{ext}"
        if frame_path.exists():
            from fastapi.responses import FileResponse
            return FileResponse(frame_path)
    
    raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found")


@app.get("/api/reasons")
async def get_reasons():
    """모든 선택 이유를 반환하는 API"""
    reasons = load_reasons()
    return JSONResponse(content=reasons)


@app.get("/api/health")
async def health_check():
    """서버 상태 확인"""
    return JSONResponse(content={
        "status": "ok",
        "frames_dir": str(FRAMES_DIR) if FRAMES_DIR else None,
        "reasons_json": str(REASONS_JSON) if REASONS_JSON else None,
        "frames_count": len(get_frame_files())
    })


def main(frames_dir: Path, port: int = 8002):
    """
    프레임 표시 서버 시작
    
    Args:
        frames_dir: final_selected_frames 디렉토리 경로
        port: 서버 포트 (기본값: 8002)
    """
    set_frames_dir(frames_dir)
    
    print(f"🚀 Selected Frames Display Server is starting")
    print(f"📁 Frames Directory: {FRAMES_DIR}")
    print(f"📄 Reasons JSON: {REASONS_JSON}")
    print(f"🌐 Please access http://localhost:{port} in your browser")
    
    uvicorn.run(
        "upload_frames_to_web:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python upload_frames_to_web.py <frames_dir> [port]")
        print("Example: python upload_frames_to_web.py frames_1214/final_selected_frames 8002")
        sys.exit(1)
    
    frames_dir = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8002
    
    main(frames_dir, port)
