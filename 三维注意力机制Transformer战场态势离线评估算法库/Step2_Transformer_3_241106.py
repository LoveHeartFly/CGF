import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from timesformer_pytorch_3_241105 import TimeSformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
input_channels = 5  # 每个单位的5个属性
batch_size = 1  # 批次
num_epochs = 10
learning_rate = 0.0001

# 设置数据集根目录
root_dir = '.\M1'


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, subset):
        self.data_paths = []
        self.label_paths = []
        # 遍历文件夹结构
        for exp_folder in sorted(os.listdir(root_dir)):
            exp_path = os.path.join(root_dir, exp_folder)
            if os.path.isdir(exp_path):
                for sub_folder in sorted(os.listdir(exp_path)):
                    if len(sub_folder) > 2:
                        # 忽略读到的txt文件
                        continue
                    folder_index = int(sub_folder)  # 将文件夹名转换为数字，方便选择
                    if (subset == 'train' and 1 <= folder_index <= 20) or \
                            (subset == 'val' and 20 < folder_index <= 30) or \
                            (subset == 'test' and 30 < folder_index <= 35):
                        sub_path = os.path.join(exp_path, sub_folder)
                        if os.path.isdir(sub_path):
                            # data_file = os.path.join(sub_path, '1.txt')
                            files = os.listdir(sub_path)
                            data_file = os.path.join(sub_path, files[0])
                            label_file = os.path.join(sub_path, 'winner.txt')
                        if os.path.exists(data_file) and os.path.exists(label_file):
                            self.data_paths.append(data_file)
                            self.label_paths.append(label_file)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 读取数据和标签
        data = torch.tensor(self._load_txt(self.data_paths[idx]), dtype=torch.float32)
        with open(self.label_paths[idx]) as file:
            label = torch.tensor(float(file.read().strip()), dtype=torch.float32)
        # label = torch.tensor(self._load_txt(self.label_paths[idx]), dtype=torch.float32)
        return data, label

    def _load_txt(self, file_path):
        sequences = []
        previous_sequence = None  # 用于存储上一个16x16x5的单元
        current_sequence = torch.empty(0, 16, 5).to(device)  # 当前帧，初始化为空的 torch 张量

        with open(file_path, 'r') as f:
            current_sequence = []
            for line in f:
                line = line.strip()
                if line:  # 非空行
                    # 将每一行按照空格分割成16个5维的向量
                    units = [list(map(float, unit.split(','))) for unit in line.split()]
                    current_sequence.append(units)
                else:
                    if current_sequence != previous_sequence:  # 检查是否与前一个相同
                        sequences.append(current_sequence)  # 如果不同，添加到序列中
                        previous_sequence = current_sequence  # 更新为最新单元
                    if len(sequences) == 500:
                        break
                    current_sequence = []

            # 如果数据少于500条，则进行填充
            while len(sequences) < 500:
                sequences.append([[[0.0] * 5 for _ in range(16)] for _ in range(16)])  # 16x16x5全为0的填充
            # 将数据转成形状为 [500, 5, 16, 16]，即重新排列顺序
        sequences = torch.tensor(sequences).permute(0, 3, 1, 2)
        return sequences  # 返回固定长度的时间序列


# 初始化 TimesFormer 模型
model = TimeSformer(
    dim=128,
    num_frames=500,  # 模型的维度
    image_size=16,  # 每一帧的尺寸
    patch_size=4,
    channels=5,  # 图像划分的patch大小
    num_classes=2,  # 分类任务的类别数量（可以根据你的任务调整）
    depth=8,  # Transformer 层的数量
    heads=8,  # 多头自注意力的头数
    dim_head=32,  # 每个头的维度
    attn_dropout=0.1,  # 注意力层的 dropout 比例
    ff_dropout=0.1,  # 前馈网络的 dropout 比例
).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}, Train"):
            # 将数据移动到GPU或CPU设备上
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels.long())
            # loss = criterion(outputs.view(-1, outputs.size(-1)), labels.repeat_interleave(500))
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 记录损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # 打印每个epoch的损失和准确率
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')


train_dataset = CustomDataset(root_dir, subset='train')
val_dataset = CustomDataset(root_dir, subset='val')
test_dataset = CustomDataset(root_dir, subset='test')

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_model(model, train_loader, criterion, optimizer, device, num_epochs)
torch.save(model.state_dict(), './model_500_3_241106.pth')

torch.cuda.empty_cache()
