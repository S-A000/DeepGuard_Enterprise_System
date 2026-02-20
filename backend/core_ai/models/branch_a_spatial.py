import torch
import torch.nn as nn
import torchvision.models as models

class BranchA_TimeSformer(nn.Module):
    def __init__(self):
        super(BranchA_TimeSformer, self).__init__()
        self.backbone = models.video.r3d_18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1024)

    def forward(self, x):
        return self.backbone(x)