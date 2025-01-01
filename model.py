import torch
import torch.nn as nn
import torch.utils.data as Data
from adabelief_pytorch import AdaBelief
import numpy as np

class DualSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DualSEBlock, self).__init__()

        self.fc1_se = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2_se = nn.Linear(channel // reduction, channel, bias=False)

        self.fc1_dual = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2_dual = nn.Linear(channel // reduction, channel // 2, bias=False)
        self.fc3_dual = nn.Linear(channel // 2, channel, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()


        y_se = torch.mean(x, dim=2)
        y_se = self.fc1_se(y_se)
        y_se = self.relu(y_se)
        y_se = self.fc2_se(y_se)
        y_se = self.sigmoid(y_se).unsqueeze(2)
        y_dual = torch.mean(x, dim=2)
        y_dual = self.fc1_dual(y_dual)
        y_dual = self.relu(y_dual)
        y_dual = self.fc2_dual(y_dual)
        y_dual = self.relu(y_dual)
        y_dual = self.fc3_dual(y_dual)
        y_dual = self.sigmoid(y_dual).unsqueeze(2)

        out_se = x * y_se
        out_dual = x * y_dual
        return out_se + out_dual

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction=16):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.se_block = DualSEBlock(out_channels, se_reduction)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se_block(out)
        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(residual)

        out += residual
        return out



class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, se_reduction=16):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = BasicBlock(64, 64, se_reduction)
        self.layer2 = BasicBlock(64, 128, se_reduction)
        self.layer3 = BasicBlock(128, 256, se_reduction)
        self.fc_resnet = nn.Linear(256, 256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=2)
        x = self.fc_resnet(x)
        return x



class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class ResNetBiGRU(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super(ResNetBiGRU, self).__init__()
        self.resnet = ResNet(input_channels, num_classes)
        self.bi_gru = BiGRU(256, hidden_size, num_classes)

    def forward(self, x):
        x_resnet = self.resnet(x)
        x_gru = self.bi_gru(x_resnet.unsqueeze(1))

        return x_gru
