from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session
import shutil
import os
import time
import bcrypt
from datetime import datetime

# Database aur Models
from .database import get_db
from . import models

# AI Pipeline
from core_ai.inference_pipeline import analyze_video

router = APIRouter()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../storage/uploads/"))

# --- 1. SIGNUP & LOGIN ---

@router.post("/api/signup")
def register_user(user_data: dict, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.email == user_data['email']).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Identity already exists.")

    try:
        hashed_password = bcrypt.hashpw(user_data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = models.User(
            full_name=user_data['full_name'], email=user_data['email'],
            password_hash=hashed_password, role="operator", dept_id=1, clearance_level=1
        )
        db.add(new_user)
        db.commit()
        return {"status": "success", "message": "Identity Created!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/login")
def login_user(login_data: dict, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == login_data['email']).first()
    if not user or not bcrypt.checkpw(login_data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid Credentials.")
    return {"user_id": user.user_id, "full_name": user.full_name, "role": user.role}

# --- 2. FORENSIC ANALYSIS (Atomic Transaction) ---

@router.post("/api/analyze")
async def analyze_uploaded_video(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    form_data = await request.form()
    user_id = int(form_data.get("user_id", 1))

    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unauthorized format.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    client_ip = request.client.host 
    start_time = time.time()

    try:
        # Save File
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run AI Analysis
        ai_result = analyze_video(file_path) 
        duration = round(time.time() - start_time, 2)

        # 🚀 STEP 1: Add Parent (AnalysisHistory) and FLUSH
        new_analysis = models.AnalysisHistory(
            user_id=user_id, filename=file.filename,
            verdict=ai_result.get("verdict", "UNKNOWN"),
            confidence_score=ai_result.get("confidence", 0.0),
            spatial_score=ai_result.get("branch_scores", {}).get("spatial", 0.0),
            physics_score=ai_result.get("branch_scores", {}).get("physics", 0.0),
            forensics_score=ai_result.get("branch_scores", {}).get("forensics", 0.0),
            audio_score=ai_result.get("branch_scores", {}).get("audio", 0.0),
            processing_time_sec=duration, client_ip=client_ip
        )
        db.add(new_analysis)
        db.flush() # SQL Server se ID maang lo bina commit kiye

        # 🚀 STEP 2: Add Children (Metadata & Verification)
        new_meta = models.VideoMetadata(
            analysis_id=new_analysis.analysis_id,
            file_size_mb=round(os.path.getsize(file_path) / (1024*1024), 2),
            resolution="1080p", codec="H.264"
        )
        db.add(new_meta)

        if new_analysis.verdict == "FAKE":
            new_verification = models.ResultVerification(
                analysis_id=new_analysis.analysis_id,
                verification_status="PENDING",
                comments="System Flagged for review."
            )
            db.add(new_verification)

        # Audit Log
        db.add(models.AuditLog(user_id=user_id, action_type="SCAN", description=f"Analyzed {file.filename}"))

        # 🚀 STEP 3: Final Commit
        db.commit()

        return {
            "status": "success", "verdict": new_analysis.verdict,
            "confidence": new_analysis.confidence_score, "time": duration
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ DB ERROR: {e}")
        raise HTTPException(status_code=500, detail="Database integrity failure.")

@router.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(models.AnalysisHistory).order_by(models.AnalysisHistory.timestamp.desc()).all()