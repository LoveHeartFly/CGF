import os
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from timesformer_pytorch_3_241105 import TimeSformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
# time.sleep(5*60*60)


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

            while len(sequences) < 500:
                sequences.append([[[0.0] * 5 for _ in range(16)] for _ in range(16)])  # 16x16x5全为0的填充
            # 将数据转成形状为 [500, 5, 16, 16]，即重新排列顺序
        sequences = torch.tensor(sequences).permute(0, 3, 1, 2)
        return sequences  # 返回固定长度的时间序列


# 加载数据集
root_dir = 'E:\jk\offline\step1_循环赛实验\M1'
test_dataset = CustomDataset(root_dir, subset='test')

# 定义 DataLoader
batch_size = 1  # 设定批次大小
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载 TimeSformer 模型
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
model.load_state_dict(torch.load('./model_500_3_241106.pth'))

# 定义步长大小
step_sizes = range(20, 501, 20)  # 总共n个步长
num_steps = len(step_sizes)  # 100

# 初始化四个列表，用于存储每个步长的指标
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

model.eval()

with torch.no_grad():
    for step_idx, step in tqdm.tqdm(enumerate(step_sizes)):
        all_predictions = []
        all_labels = []

        # print(f"Processing Step {step} ({step_idx + 1}/{num_steps})...")

        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(
                device)  # data shape: [batch_size, 500, 5, 16, 16], labels: [batch_size]

            # 创建填充后的数据
            # 初始化一个与原始数据形状相同的零张量
            input_padded = torch.zeros_like(data)

            # 复制前 'step' 步的数据到填充张量中
            input_padded[:, :step, :, :, :] = data[:, :step, :, :, :]

            # 将填充后的数据输入模型进行预测
            outputs = model(input_padded)  # outputs shape: [batch_size, num_classes]

            # 获取预测类别（取最大值的索引）
            _, predicted = torch.max(outputs, dim=1)  # predicted shape: [batch_size]

            # 将预测结果和真实标签添加到列表中
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算当前步长的指标
        acc = accuracy_score(all_labels, all_predictions)
        prec = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        # 将计算结果添加到对应的列表中
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        # 打印当前步长的指标
        print(f"Step {step}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}\n")


import pandas as pd

metrics_df = pd.DataFrame({
    'Step': list(step_sizes),
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1-Score': f1_list
})

print(metrics_df)

