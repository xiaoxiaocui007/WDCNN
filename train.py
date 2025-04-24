import os

import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
from model_2 import *
import time
from torch.utils.data import Dataset
import numpy as np


# 3，6，9是b类

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


batch_size = 1

# 创建数据集
dataset = CustomDataset("./data")

# 指定数据集的总大小
total_size = len(dataset)

# 计算训练集和验证集的大小
train_size = int(0.6 * total_size)  # 80% 用于训练
val_size = total_size - train_size  # 20% 用于验证

# 将数据集划分为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# 创建训练集和验证集的DataLoader实例
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_name = "MyModel"
model = MyModel(num_class=len(os.listdir("./data")))  # MyModel  MyModel_2 MyModel_3

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
start_time = time.time()

# 最高acc
epoch_acc_best = 0
for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions / len(train_dataset)

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    print(f'Epoch Train {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = correct_predictions / len(val_dataset)

        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        print(f'Epoch Val {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        if epoch_acc > epoch_acc_best:
            epoch_acc_best = epoch_acc
            model_state_dict_best = model.state_dict()
            print(f"更新了模型，{epoch_acc_best:.4f}")

print(f"训练时间：{time.time() - start_time}")

# 保存模型
torch.save(model_state_dict_best, f'./weights/{model_name}.pth')

# 绘制acc和loss曲线
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"./images/Loss_{model_name}.png", dpi=300)
plt.show()
