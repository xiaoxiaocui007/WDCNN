import os

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
from model_2 import *
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, file_dir):
        data_list = []
        labels_list = []

        files = os.listdir(file_dir)

        for i, file in enumerate(files):
            file_path = os.path.join(file_dir, file)
            data = pd.read_csv(file_path)
            for j in range(len(data)):
                labels_list.append(i)  # 添加标签
                row_data = data.iloc[j].astype(np.float32)  # 预处理
                data_list.append(row_data / 1000)

        self.data_numpy = np.array(data_list)
        self.label_numpy = np.array(labels_list)

    def __len__(self):
        return len(self.label_numpy)

    def __getitem__(self, index):
        data = self.data_numpy[index]
        label = self.label_numpy[index]

        return data, label


batch_size = 32

# 创建数据集
dataset = CustomDataset("./data")

# 部分数据
# 指定数据集的总大小
total_size = len(dataset)

# 计算训练集和验证集的大小
train_size = int(0.8 * total_size)  # 80% 用于训练
val_size = total_size - train_size  # 20% 用于验证

# 将数据集划分为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

## 完整数据
# test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

classes_names = ['a',
                 'b',
                 'c',
                 'd',
                 ]  # 自定义

# 加载预训练模型权重
model_name = "MyModel"
model = MyModel(num_class=len(os.listdir("./data")))  # MyModel  MyModel_2 MyModel_3

model_weights_path = f'./weights/{model_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()

# 在测试集上进行预测
model.to(device)

all_labels = []
all_predictions = []
all_signals = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Validation', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_signals.extend(outputs.cpu().numpy())
###########################################################################################
# t-SNE
signals = np.array(all_signals)
labels = np.array(all_labels)
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(signals)

plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(x=embedded_data[:, 0], y=embedded_data[:, 1], hue=labels, palette="tab10")
plt.title('t-SNE Visualization of 1D Signals')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles, labels, title="Classes")
plt.savefig(f"./images/t-SNE_{model_name}.png", dpi=300)
plt.show()
###########################################################################################
# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes_names,
            yticklabels=classes_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f"./images/Confusion_{model_name}.png", dpi=300)
plt.show()
###########################################################################################
# 输出分类报告
class_report = classification_report(all_labels, all_predictions, target_names=classes_names)
print("Classification Report:\n", class_report)

# 计算精度
accuracy = sum([1 for i, j in zip(all_labels, all_predictions) if i == j]) / len(all_labels)
print("Accuracy:", accuracy)

"""
Classification Report:
               precision    recall  f1-score   support

           a       1.00      1.00      1.00        60
           b       1.00      1.00      1.00        30

    accuracy                           1.00        90
   macro avg       1.00      1.00      1.00        90
weighted avg       1.00      1.00      1.00        90

Accuracy: 1.0
"""
