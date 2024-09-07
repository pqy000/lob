# from tsai.models import InceptionTimePlus
from tsai.models.InceptionTimePlus import InceptionTimePlus
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, args, data, num_classes=5, pretrained=True):
        super(Model, self).__init__()

        # 使用 InceptionTimePlus 预训练模型
        input_dim = data.m
        self.feature_extractor = InceptionTimePlus(c_in=input_dim, c_out=num_classes)
        self.fc = nn.Linear(num_classes, num_classes)  # 这里假设 num_classes 一致

    def forward(self, x):
        # x的输入维度为 (batch_size, window_size, feature_dim)
        x = x.transpose(1, 2)
        out = self.feature_extractor(x)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        return out