import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class DeepGuardDataset(Dataset):
    def __init__(self, real_dir, fake_dir, num_frames=16):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        
        if os.path.exists(real_dir):
            for vid in os.listdir(real_dir):
                if vid.endswith(('.mp4', '.avi')):
                    self.video_paths.append(os.path.join(real_dir, vid))
                    self.labels.append(0.0) # REAL
                    
        if os.path.exists(fake_dir):
            for vid in os.listdir(fake_dir):
                if vid.endswith(('.mp4', '.avi')):
                    self.video_paths.append(os.path.join(fake_dir, vid))
                    self.labels.append(1.0) # FAKE

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
        return np.array(frames)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames_np = self.extract_frames(video_path)
        
        video_rgb = torch.tensor(frames_np).permute(3, 0, 1, 2).float() / 255.0
        flow_frames = torch.randn(2, 224, 224)
        forensics_frames = torch.randn(self.num_frames, 3, 224, 224)
        audio_features = torch.randn(768)

        return video_rgb, flow_frames, forensics_frames, audio_features, torch.tensor([label])