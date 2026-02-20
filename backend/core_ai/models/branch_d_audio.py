import torch
import torch.nn as nn

class BranchD_Audio(nn.Module):
    def __init__(self):
        super(BranchD_Audio, self).__init__()
        self.audio_fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, audio_features):
        return self.audio_fc(audio_features)