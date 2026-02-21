from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# Database aur Models import karein
import app.models as models
from app.database import engine

# Routes import
from app.api_routes import router as api_router

app = FastAPI(title="DeepGuard Enterprise API", version="1.0")

# --- DATABASE INITIALIZATION ---
# Server start hote hi ye line SQL Server mein saare tables bana degi
print("📡 Connecting to SQL Server and Initializing Governance Tables...")
models.Base.metadata.create_all(bind=engine)

# Security Bypass for React (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect API Routes
app.include_router(api_router)

# Storage folders creation
os.makedirs("../storage/uploads", exist_ok=True)
os.makedirs("../storage/reports", exist_ok=True)

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "DeepGuard Enterprise Governance",
        "database": "SQL Server Connected ✅"
    }

if __name__ == "__main__":
    print("🚀 Starting DeepGuard Backend Server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)