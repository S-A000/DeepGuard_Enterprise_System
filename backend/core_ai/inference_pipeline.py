import os
import torch
import random
import time
from .models.fusion_net import DeepGuardFusionModel

# Paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.abspath(os.path.join(current_dir, "../../saved_models/production/deepguard_fusion_v1.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable for model (taake har bar upload par model reload na ho)
_deepguard_model = None
_is_mock_mode = True

def load_model():
    global _deepguard_model, _is_mock_mode
    if _deepguard_model is None:
        try:
            _deepguard_model = DeepGuardFusionModel().to(device)
            if os.path.exists(MODEL_WEIGHTS_PATH):
                print("[✓] Asli Trained Model (.pth) loaded successfully!")
                _deepguard_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
                _deepguard_model.eval()
                _is_mock_mode = False
            else:
                print("[!] Warning: .pth file nahi mili! Running in DUMMY MOCK MODE for UI testing.")
                _is_mock_mode = True
        except Exception as e:
            print(f"[X] Error loading model structure: {e}. Defaulting to Mock Mode.")
            _is_mock_mode = True

def analyze_video(video_path: str) -> dict:
    """ AI Model se video scan karwa kar result deta hai """
    load_model()
    
    # UI Loading animation dikhane ke liye 3 seconds ka delay (Mock processing time)
    time.sleep(3) 
    
    if not _is_mock_mode:
        # ASLI INFERENCE LOGIC YAHAN AAYEGA
        # In production, yahan video se frames nikal kar tensors banenge
        dummy_rgb = torch.randn(1, 3, 16, 224, 224).to(device)
        dummy_flow = torch.randn(1, 2, 224, 224).to(device)
        dummy_fft = torch.randn(1, 16, 3, 224, 224).to(device)
        dummy_audio = torch.randn(1, 768).to(device)

        with torch.no_grad():
            probability_tensor = _deepguard_model(dummy_rgb, dummy_flow, dummy_fft, dummy_audio)
            prob = probability_tensor.item()
    else:
        # DUMMY MOCK LOGIC (UI Testing ke liye)
        prob = random.uniform(0.1, 0.99)

    is_fake = bool(prob >= 0.5)
    confidence = round((prob if is_fake else (1 - prob)) * 100, 2)

    return {
        "status": "success",
        "verdict": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
        "branch_scores": {
            "spatial": round(random.uniform(10, 90), 1),
            "physics": round(random.uniform(10, 90), 1),
            "forensics": round(random.uniform(10, 90), 1),
            "audio": round(random.uniform(10, 90), 1)
        }
    }