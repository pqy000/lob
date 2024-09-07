import torch
import torch.nn as nn
import torch.nn.functional as F

# InceptionTime 模块的实现
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1) if in_channels > 1 else nn.Identity()
        self.conv1 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels, n_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(4 * n_filters)

    def forward(self, x):
        y1 = self.conv1(self.bottleneck(x))
        y2 = self.conv2(self.bottleneck(x))
        y3 = self.conv3(self.bottleneck(x))
        y4 = self.conv4(self.pool(x))
        out = torch.cat([y1, y2, y3, y4], dim=1)
        return F.relu(self.bn(out))

class InceptionTime(nn.Module):
    def __init__(self, input_dim, n_filters, n_blocks=6, bottleneck_channels=32, kernel_sizes=[9, 19, 39]):
        super(InceptionTime, self).__init__()
        blocks = []
        for i in range(n_blocks):
            in_channels = input_dim if i == 0 else 4 * n_filters
            blocks.append(InceptionBlock(in_channels, n_filters, kernel_sizes, bottleneck_channels))
        self.network = nn.Sequential(*blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.network(x)
        x = self.global_avg_pool(x).squeeze(-1)  # Global average pooling to reduce feature dim
        return x

# 改进后的 FuturesNet 模型：InceptionTime + LSTM + Skip Connection
class Model(nn.Module):
    def __init__(self, args, data, num_classes=5):
        super(Model, self).__init__()
        feature_dim = data.m  # 原始输入特征的维度 (10 in your case)
        hidden_dim = args.hidRNN

        # 使用 InceptionTime 进行特征提取
        self.feature_extractor = InceptionTime(input_dim=feature_dim, n_filters=32)

        # LSTM for temporal modeling
        lstm_input_size = 8  # InceptionTime 输出的维度
        other_dimension = 8
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # Skip Connection
        self.skip_connection = nn.Linear(other_dimension, hidden_dim)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, window_size, feature_dim = x.size()

        # 使用 InceptionTime 提取特征
        x = x.transpose(1, 2)  # 转换为 (batch_size, feature_dim, window_size)，以适配1D卷积
        x = self.feature_extractor(x)  # 提取时间序列特征
        x = x.view(batch_size, window_size, -1)  # 调整维度为 (batch_size, window_size, lstm_input_size)

        # 使用 LSTM 进行时间序列建模
        lstm_out, _ = self.lstm(x)
        # Skip Connection on hidden state
        skip_out = self.skip_connection(x)

        # LSTM 输出加上 Skip Connection
        lstm_out = lstm_out + skip_out

        # 输出分类结果
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时刻的输出 (batch_size, num_classes)
        out = torch.softmax(out, dim=1)
        return out
