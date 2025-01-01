import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from adabelief_pytorch import AdaBelief
import pandas as pd


csv_file = 'all_features.csv'  # 请根据实际情况修改文件路径

# 使用pandas加载CSV文件
data_df = pd.read_csv(csv_file)

# 假设最后一列是标签，其余列是特征
X = data_df.iloc[:, :-1].values  # 获取所有特征列
y = data_df.iloc[:, -1].values   # 获取标签列
data = np.random.rand(119808, 10)  # 119808个样本，每个样本10个特征
labels = np.random.randint(0, 2, 119808)  # 二分类标签示例

# 训练集与测试集划分
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class DualSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DualSEBlock, self).__init__()
        # 第一个通道：标准 SE 注意力机制
        self.fc1_se = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2_se = nn.Linear(channel // reduction, channel, bias=False)

        # 第二个通道：改进的双通道 SE 注意力机制
        self.fc1_dual = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2_dual = nn.Linear(channel // reduction, channel // 2, bias=False)
        self.fc3_dual = nn.Linear(channel // 2, channel, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()

        # 第一个通道（SE）
        y_se = torch.mean(x, dim=2)  # (B, C)
        y_se = self.fc1_se(y_se)           # (B, C//reduction)
        y_se = self.relu(y_se)
        y_se = self.fc2_se(y_se)           # (B, C)
        y_se = self.sigmoid(y_se).unsqueeze(2)  # (B, C, 1)

        # 第二个通道（双通道 SE）
        y_dual = torch.mean(x, dim=2)  # (B, C)
        y_dual = self.fc1_dual(y_dual)  # (B, C//reduction)
        y_dual = self.relu(y_dual)
        y_dual = self.fc2_dual(y_dual)  # (B, C//2)
        y_dual = self.relu(y_dual)
        y_dual = self.fc3_dual(y_dual)  # (B, C)
        y_dual = self.sigmoid(y_dual).unsqueeze(2)  # (B, C, 1)

        # 两个通道的加权结果
        out_se = x * y_se
        out_dual = x * y_dual
        return out_se + out_dual  # 非对称加权


# 更新 BasicBlock 以使用 DualSEBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction=16):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 使用 DualSEBlock 代替原 SEBlock
        self.se_block = DualSEBlock(out_channels, se_reduction)

        # 如果输入通道数与输出通道数不一致，使用卷积调整残差通道数
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # 使用 DualSEBlock 进行通道加权
        out = self.se_block(out)

        # 如果输入和输出通道数不同，调整残差的通道数
        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(residual)

        out += residual
        return out

# 定义ResNet模型 (加入SENet模块)
class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, se_reduction=16):
        super(ResNet, self).__init__()
        # 输入层：卷积操作，之后是批标准化和ReLU激活
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = BasicBlock(64, 64, se_reduction)
        self.layer2 = BasicBlock(64, 128, se_reduction)
        self.layer3 = BasicBlock(128, 256, se_reduction)
        self.fc_resnet = nn.Linear(256, 256)  # 作为 ResNet 到 GRU 的过渡

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



class CrossAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, hidden_size):
        super(CrossAttention, self).__init__()
        self.query_projection = nn.Linear(query_size, hidden_size)
        self.key_projection = nn.Linear(key_size, hidden_size)
        self.value_projection = nn.Linear(value_size, hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 256)

    def forward(self, query, key, value):
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Cross Attention Mechanism
        attn_output, _ = self.attn(query, key, value)

        return self.fc(attn_output)


# 定义整体模型（ResNet + CrossAttention + BiGRU + SENet）
class ResNetBiGRUWithAttention(nn.Module):
    def __init__(self, input_channels, num_classes, gru_hidden_size, se_reduction=16, attention_hidden_size=128):
        super(ResNetBiGRUWithAttention, self).__init__()
        self.resnet = ResNet(input_channels, num_classes, se_reduction)
        self.cross_attention = CrossAttention(query_size=256, key_size=256, value_size=256, hidden_size=attention_hidden_size)
        self.bigru = BiGRU(input_size=256, hidden_size=gru_hidden_size, num_classes=num_classes)

    def forward(self, x):
        resnet_features = self.resnet(x)  # 获取 ResNet 特征
        resnet_features = resnet_features.unsqueeze(1)  # 增加时间维度，使其适配GRU的输入形状

        # 通过CrossAttention融合特征
        attention_output = self.cross_attention(resnet_features, resnet_features, resnet_features)

        # 传递给 BiGRU
        output = self.bigru(attention_output)
        return output


# 初始化模型，定义损失函数和优化器
model = ResNetBiGRUWithAttention(input_channels=10, num_classes=2, gru_hidden_size=128, se_reduction=16)  # 假设是二分类任务
criterion = nn.CrossEntropyLoss()

# 使用AdaBelief优化器
optimizer = AdaBelief(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-16, weight_decouple=True)
# 定义保存模型路径
last_model_path = "last_model.pth"
best_model_path = "best_model.pth"

# 初始化变量来跟踪最佳准确率
best_accuracy = 0.0

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 调整输入的形状
        inputs = inputs.unsqueeze(2)  # 适配Conv1d输入形状 (batch_size, channels, sequence_length)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(2)  # 使得形状为 (batch_size, 10, 1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # 保存最后一个模型
    torch.save(model.state_dict(), last_model_path)

    # 如果当前测试准确率更高，保存为最佳模型
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

print(f"Final Best Test Accuracy: {best_accuracy:.2f}%")









