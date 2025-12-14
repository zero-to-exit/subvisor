"""
웹에서 영상 업로드 및 로컬 저장 서버
FastAPI를 사용하여 간단한 웹 인터페이스 제공
프레임 표시 기능 포함
"""
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import json
from typing import Dict, Optional

# 저장 디렉토리 설정
UPLOAD_DIR = Path(__file__).parent / "uploaded_videos"
UPLOAD_DIR.mkdir(exist_ok=True)

# 작업 디렉토리 설정 (프레임 표시용)
WORKING_DIR: Optional[Path] = None


def set_upload_dir(upload_dir: Path):
    """
    업로드 디렉토리 설정
    
    Args:
        upload_dir: 업로드 디렉토리 경로
    """
    global UPLOAD_DIR
    UPLOAD_DIR = Path(upload_dir)
    UPLOAD_DIR.mkdir(exist_ok=True)


def set_working_dir(working_dir: Path):
    """
    작업 디렉토리 설정 (프레임 표시용)
    
    Args:
        working_dir: 작업 디렉토리 경로
    """
    global WORKING_DIR
    WORKING_DIR = Path(working_dir)

app = FastAPI(title="Video Upload Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_reasons() -> Dict[str, str]:
    """reasons.json 파일을 로드"""
    if WORKING_DIR is None:
        return {}
    
    reasons_json = WORKING_DIR / "final_selected_frames" / "reasons.json"
    if not reasons_json.exists():
        return {}
    
    try:
        with open(reasons_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load reasons.json: {e}")
        return {}


def get_frame_files() -> list:
    """프레임 디렉토리에서 이미지 파일 목록 가져오기"""
    if WORKING_DIR is None:
        return []
    
    frames_dir = WORKING_DIR / "final_selected_frames"
    if not frames_dir.exists():
        return []
    
    frame_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        frame_files.extend(frames_dir.glob(f"*{ext}"))
    
    # 파일명의 숫자로 정렬
    frame_files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    return frame_files


@app.get("/", response_class=HTMLResponse)
async def upload_page():
    """영상 업로드 페이지 (메인 페이지)"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>영상 업로드</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 3px dashed #4CAF50;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background-color: #f9f9f9;
                margin-bottom: 20px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background-color: #f0f0f0;
                border-color: #45a049;
            }
            .upload-area.dragover {
                background-color: #e8f5e9;
                border-color: #2e7d32;
            }
            input[type="file"] {
                display: none;
            }
            .file-label {
                display: inline-block;
                padding: 12px 24px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .file-label:hover {
                background-color: #45a049;
            }
            .upload-btn {
                width: 100%;
                padding: 15px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 18px;
                cursor: pointer;
                margin-top: 20px;
                transition: background-color 0.3s;
            }
            .upload-btn:hover {
                background-color: #0b7dda;
            }
            .upload-btn:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .status.success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .file-info {
                margin-top: 15px;
                padding: 10px;
                background-color: #e3f2fd;
                border-radius: 5px;
                display: none;
            }
            .progress-bar {
                width: 100%;
                height: 30px;
                background-color: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin-top: 15px;
                display: none;
            }
            .progress-fill {
                height: 100%;
                background-color: #4CAF50;
                width: 0%;
                transition: width 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                border-bottom: 2px solid #e0e0e0;
            }
            .tab {
                padding: 12px 24px;
                background: none;
                border: none;
                border-bottom: 3px solid transparent;
                cursor: pointer;
                font-size: 16px;
                color: #666;
                transition: all 0.3s;
            }
            .tab:hover {
                color: #2196F3;
            }
            .tab.active {
                color: #2196F3;
                border-bottom-color: #2196F3;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 영상 처리 시스템</h1>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('upload')">📤 영상 업로드</button>
                <button class="tab" onclick="showTab('frames')">📸 선택된 프레임</button>
            </div>
            
            <div id="uploadTab" class="tab-content active">
                <h2 style="text-align: center; margin-bottom: 20px;">영상 업로드</h2>
            
            <div class="upload-area" id="uploadArea">
                <p style="font-size: 18px; margin-bottom: 20px;">영상 파일을 드래그하거나 클릭하여 선택하세요</p>
                <label for="fileInput" class="file-label">파일 선택</label>
                <input type="file" id="fileInput" accept="video/*" />
            </div>
            
            <div class="file-info" id="fileInfo">
                <strong>선택된 파일:</strong> <span id="fileName"></span><br>
                <strong>파일 크기:</strong> <span id="fileSize"></span>
            </div>
            
            <button class="upload-btn" id="uploadBtn" disabled>업로드</button>
            
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
            
            <div class="status" id="status"></div>
            </div>
            
            <div id="framesTab" class="tab-content">
                <h2 style="text-align: center; margin-bottom: 20px;">선택된 프레임</h2>
                <div class="stats" id="stats" style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
                    <span>로딩 중...</span>
                </div>
                <div id="framesContainer" style="min-height: 200px;">
                    <div style="text-align: center; padding: 50px; color: #666;">
                        프레임을 불러오는 중...
                    </div>
                </div>
            </div>
        </div>

        <script>
            function showTab(tabName) {
                // 모든 탭과 콘텐츠 숨기기
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // 선택된 탭과 콘텐츠 표시
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach((tab, index) => {
                    if ((index === 0 && tabName === 'upload') || (index === 1 && tabName === 'frames')) {
                        tab.classList.add('active');
                    }
                });
                document.getElementById(tabName + 'Tab').classList.add('active');
                
                // 프레임 탭이면 프레임 로드
                if (tabName === 'frames') {
                    loadFrames();
                }
            }
            
            async function loadFrames() {
                try {
                    const response = await fetch('/api/frames');
                    if (!response.ok) {
                        throw new Error('프레임을 불러오는데 실패했습니다.');
                    }
                    
                    const data = await response.json();
                    
                    // 통계 업데이트
                    document.getElementById('stats').innerHTML = 
                        `<span style="font-size: 1.1em; color: #2196F3; font-weight: bold;">총 ${data.frames.length}개의 프레임이 선택되었습니다</span>`;
                    
                    // 프레임 표시
                    const container = document.getElementById('framesContainer');
                    
                    if (data.frames.length === 0) {
                        container.innerHTML = '<div style="background: #ffebee; color: #c62828; padding: 20px; border-radius: 8px; text-align: center;">표시할 프레임이 없습니다. 영상을 업로드하고 처리해주세요.</div>';
                        return;
                    }
                    
                    container.innerHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px;">' +
                        data.frames.map(frame => `
                            <div style="background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform='translateY(0)'">
                                <img src="/api/frame/${frame.frame_number}" 
                                     alt="Frame ${frame.frame_number}" 
                                     style="width: 100%; height: 250px; object-fit: cover; display: block;"
                                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'350\\' height=\\'250\\'%3E%3Crect fill=\\'%23ddd\\' width=\\'350\\' height=\\'250\\'/%3E%3Ctext fill=\\'%23999\\' font-family=\\'sans-serif\\' font-size=\\'16\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dominant-baseline=\\'middle\\'%3E이미지를 불러올 수 없습니다%3C/text%3E%3C/svg%3E';">
                                <div style="padding: 15px;">
                                    <div style="font-size: 1em; color: #2196F3; font-weight: bold; margin-bottom: 10px;">
                                        🎬 Frame ${frame.frame_number}
                                    </div>
                                    <div style="color: #555; line-height: 1.6; font-size: 0.9em; background: #f8f9fa; padding: 12px; border-radius: 6px; border-left: 3px solid #2196F3; white-space: pre-wrap;">
                                        ${frame.reason || '선택 이유가 없습니다.'}
                                    </div>
                                </div>
                            </div>
                        `).join('') + '</div>';
                    
                } catch (error) {
                    document.getElementById('framesContainer').innerHTML = 
                        `<div style="background: #ffebee; color: #c62828; padding: 20px; border-radius: 8px; text-align: center;">오류: ${error.message}</div>`;
                }
            }
            
            // 페이지 로드 시 프레임 탭이 활성화되어 있으면 프레임 로드
            window.addEventListener('load', () => {
                if (document.getElementById('framesTab').classList.contains('active')) {
                    loadFrames();
                }
            });
            
        </script>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('uploadArea');
            const uploadBtn = document.getElementById('uploadBtn');
            const fileInfo = document.getElementById('fileInfo');
            const fileName = document.getElementById('fileName');
            const fileSize = document.getElementById('fileSize');
            const status = document.getElementById('status');
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');
            
            let selectedFile = null;

            // 파일 선택
            fileInput.addEventListener('change', (e) => {
                handleFileSelect(e.target.files[0]);
            });

            // 드래그 앤 드롭
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    handleFileSelect(e.dataTransfer.files[0]);
                }
            });

            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });

            function handleFileSelect(file) {
                if (!file) return;
                
                // 비디오 파일인지 확인
                if (!file.type.startsWith('video/')) {
                    showStatus('비디오 파일만 업로드 가능합니다.', 'error');
                    return;
                }

                selectedFile = file;
                fileName.textContent = file.name;
                fileSize.textContent = formatFileSize(file.size);
                fileInfo.style.display = 'block';
                uploadBtn.disabled = false;
                status.style.display = 'none';
            }

            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
            }

            // 업로드 버튼 클릭
            uploadBtn.addEventListener('click', async () => {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('file', selectedFile);

                uploadBtn.disabled = true;
                progressBar.style.display = 'block';
                status.style.display = 'none';

                try {
                    const xhr = new XMLHttpRequest();

                    // 진행률 업데이트
                    xhr.upload.addEventListener('progress', (e) => {
                        if (e.lengthComputable) {
                            const percentComplete = (e.loaded / e.total) * 100;
                            progressFill.style.width = percentComplete + '%';
                            progressFill.textContent = Math.round(percentComplete) + '%';
                        }
                    });

                    xhr.addEventListener('load', () => {
                        progressBar.style.display = 'none';
                        if (xhr.status === 200) {
                            const response = JSON.parse(xhr.responseText);
                            showStatus(`✅ 업로드 성공!<br>파일명: ${response.filename}<br>저장 경로: ${response.saved_path}`, 'success');
                            uploadBtn.disabled = false;
                            fileInput.value = '';
                            selectedFile = null;
                            fileInfo.style.display = 'none';
                        } else {
                            const response = JSON.parse(xhr.responseText);
                            showStatus('❌ 업로드 실패: ' + response.detail, 'error');
                            uploadBtn.disabled = false;
                        }
                    });

                    xhr.addEventListener('error', () => {
                        progressBar.style.display = 'none';
                        showStatus('❌ 업로드 중 오류가 발생했습니다.', 'error');
                        uploadBtn.disabled = false;
                    });

                    xhr.open('POST', '/upload');
                    xhr.send(formData);

                } catch (error) {
                    progressBar.style.display = 'none';
                    showStatus('❌ 오류: ' + error.message, 'error');
                    uploadBtn.disabled = false;
                }
            });

            function showStatus(message, type) {
                status.innerHTML = message;
                status.className = 'status ' + type;
                status.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    영상 파일 업로드 및 로컬 저장
    """
    # 파일 확장자 확인
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(allowed_extensions)}"
        )
    
    # 파일명 생성 (타임스탬프 추가로 중복 방지)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem
    safe_filename = f"{timestamp}_{original_name}{file_ext}"
    save_path = UPLOAD_DIR / safe_filename
    
    try:
        # 파일 저장
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size = save_path.stat().st_size / (1024 * 1024)  # MB
        
        return JSONResponse({
            "message": "파일 업로드 성공",
            "filename": safe_filename,
            "original_filename": file.filename,
            "saved_path": str(save_path),
            "file_size_mb": round(file_size, 2),
            "upload_time": timestamp
        })
    
    except Exception as e:
        # 저장 실패 시 파일 삭제 시도
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")


@app.get("/list")
async def list_uploaded_videos():
    """업로드된 영상 목록 조회"""
    videos = []
    for video_file in sorted(UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if video_file.is_file():
            file_size = video_file.stat().st_size / (1024 * 1024)  # MB
            videos.append({
                "filename": video_file.name,
                "path": str(video_file),
                "size_mb": round(file_size, 2),
                "modified_time": datetime.fromtimestamp(video_file.stat().st_mtime).isoformat()
            })
    
    return JSONResponse({
        "count": len(videos),
        "videos": videos,
        "upload_dir": str(UPLOAD_DIR)
    })


@app.get("/frames", response_class=HTMLResponse)
async def frames_page():
    """프레임 표시 페이지 (리다이렉트용)"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/?tab=frames">
    </head>
    <body>
        <p>리다이렉트 중...</p>
    </body>
    </html>
    """)


@app.get("/api/frames")
async def get_frames():
    """프레임 목록과 이유를 반환하는 API"""
    if WORKING_DIR is None:
        return JSONResponse(content={"frames": [], "total": 0})
    
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
    if WORKING_DIR is None:
        raise HTTPException(status_code=404, detail="Working directory not set")
    
    frames_dir = WORKING_DIR / "final_selected_frames"
    
    # 여러 확장자 시도
    for ext in ['.jpg', '.jpeg', '.png']:
        frame_path = frames_dir / f"{frame_number}{ext}"
        if frame_path.exists():
            return FileResponse(frame_path)
    
    raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found")


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    frames_dir = None
    if WORKING_DIR:
        frames_dir = WORKING_DIR / "final_selected_frames"
    
    return {
        "status": "healthy",
        "upload_dir": str(UPLOAD_DIR),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "working_dir": str(WORKING_DIR) if WORKING_DIR else None,
        "frames_dir": str(frames_dir) if frames_dir else None,
        "frames_count": len(get_frame_files())
    }


def run_server(upload_dir: Path = None, host: str = "0.0.0.0", port: int = 8001):
    """
    비디오 업로드 서버 실행
    
    Args:
        upload_dir: 업로드 디렉토리 경로 (None이면 기본 경로 사용)
        host: 서버 호스트
        port: 서버 포트
    
    Returns:
        uvicorn 서버 인스턴스
    """
    global UPLOAD_DIR
    if upload_dir:
        UPLOAD_DIR = Path(upload_dir)
        UPLOAD_DIR.mkdir(exist_ok=True)
    
    print(f"🚀 영상 업로드 서버 시작")
    print(f"📁 저장 디렉토리: {UPLOAD_DIR}")
    print(f"🌐 웹 브라우저에서 http://localhost:{port} 접속하세요")
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=False
    )
    server = uvicorn.Server(config)
    return server


def get_latest_uploaded_video(upload_dir: Path = None) -> Path:
    """
    가장 최근에 업로드된 비디오 경로 반환
    
    Args:
        upload_dir: 업로드 디렉토리 경로 (None이면 기본 경로 사용)
    
    Returns:
        가장 최근 비디오 파일 경로, 없으면 None
    """
    if upload_dir:
        target_dir = Path(upload_dir)
    else:
        target_dir = UPLOAD_DIR
    
    if not target_dir.exists():
        return None
    
    videos = list(target_dir.glob("*"))
    if not videos:
        return None
    
    # 가장 최근 수정된 파일 반환
    latest = max(videos, key=lambda p: p.stat().st_mtime if p.is_file() else 0)
    return latest if latest.is_file() else None

