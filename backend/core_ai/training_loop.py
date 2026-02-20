import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.fusion_net import DeepGuardFusionModel
from datasets.loaders.multi_modal_loader import DeepGuardDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")

    REAL_VIDEOS_DIR = os.path.join(root_dir, 'datasets/02_forensics_training/FaceForensics++/original/')
    FAKE_VIDEOS_DIR = os.path.join(root_dir, 'datasets/02_forensics_training/FaceForensics++/manipulated/')
    SAVE_MODEL_DIR = os.path.join(root_dir, 'saved_models/production/')
    
    os.makedirs(REAL_VIDEOS_DIR, exist_ok=True)
    os.makedirs(FAKE_VIDEOS_DIR, exist_ok=True)
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    print("[*] Loading Dataset...")
    dataset = DeepGuardDataset(real_dir=REAL_VIDEOS_DIR, fake_dir=FAKE_VIDEOS_DIR, num_frames=16)
    
    if len(dataset) == 0:
        print("❌ Error: No MP4 videos found in the dataset folders!")
        return

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = DeepGuardFusionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for video_rgb, flow, fft, audio, labels in loop:
            video_rgb, flow, fft, audio, labels = video_rgb.to(device), flow.to(device), fft.to(device), audio.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            predictions = model(video_rgb, flow, fft, audio)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    final_model_path = os.path.join(SAVE_MODEL_DIR, "deepguard_fusion_v1.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Complete! Model saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()