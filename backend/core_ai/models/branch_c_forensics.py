import torch
import torch.nn as nn
import torchvision.models as models

class BranchC_DigitalForensics(nn.Module):
    def __init__(self):
        super(BranchC_DigitalForensics, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn_features(c_in)
        features = features.view(batch_size, seq_len, 512)
        lstm_out, (hidden_state, cell_state) = self.lstm(features)
        return hidden_state[-1]