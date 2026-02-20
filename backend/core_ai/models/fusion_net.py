import torch
import torch.nn as nn
from .branch_a_spatial import BranchA_TimeSformer
from .branch_b_physics import BranchB_PhysicsMotion
from .branch_c_forensics import BranchC_DigitalForensics
from .branch_d_audio import BranchD_Audio

class DeepGuardFusionModel(nn.Module):
    def __init__(self):
        super(DeepGuardFusionModel, self).__init__()
        self.branch_a = BranchA_TimeSformer()
        self.branch_b = BranchB_PhysicsMotion()
        self.branch_c = BranchC_DigitalForensics()
        self.branch_d = BranchD_Audio()
        
        self.fusion_dim = 2816
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, video_rgb, flow_frames, forensics_frames, audio_input):
        feat_a = self.branch_a(video_rgb)
        feat_b = self.branch_b(flow_frames)
        feat_c = self.branch_c(forensics_frames)
        feat_d = self.branch_d(audio_input)
        
        fused_features = torch.cat((feat_a, feat_b, feat_c, feat_d), dim=1)
        return self.classifier(fused_features)