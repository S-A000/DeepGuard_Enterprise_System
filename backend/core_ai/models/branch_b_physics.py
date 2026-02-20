import torch
import torch.nn as nn

class BranchB_PhysicsMotion(nn.Module):
    def __init__(self):
        super(BranchB_PhysicsMotion, self).__init__()
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_physics = nn.Linear(128, 512)

    def forward(self, flow_frames):
        features = self.flow_encoder(flow_frames)
        return self.fc_physics(features)