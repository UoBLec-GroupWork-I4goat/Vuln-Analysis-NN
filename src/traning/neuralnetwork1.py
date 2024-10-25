import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import csv
import pickle

# 加载数据
data = pd.read_csv('./grounddata/_sample.csv')

# 分离输入和输出
categorical_input = data[['B']].values  # B列为类别输入
numerical_input = data[['A', 'C', 'D']].values  # A、C、D列为数值输入
labels = data[['Vulnerability_Truth']].values  # 输出标签

# 对类别输入进行one-hot编码
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(categorical_input)

# 将类别输入与数值输入进行合并
inputs = np.hstack((encoded_categorical, numerical_input))

# 使用SMOTE平衡数据
smote = SMOTE(sampling_strategy=1.0, random_state=42)  # 将少数类样本增加到与多数类相同的数量
inputs_resampled, labels_resampled = smote.fit_resample(inputs, labels)


# 转换为Tensor
#inputs_resampled = torch.tensor(inputs_resampled, dtype=torch.float32)
#labels_resampled = torch.tensor(labels_resampled, dtype=torch.float32)
# 转换为Tensor
inputs_resampled = torch.tensor(inputs_resampled, dtype=torch.float32)
labels_resampled = torch.tensor(labels_resampled, dtype=torch.float32).unsqueeze(1)  # 调整 labels 的形状


# 定义神经网络模型
class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        
        # 定义网络的层次结构
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(hidden_size2, output_size)  # 第二隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数relu
        x = torch.relu(self.fc2(x))  # 激活函数relu
        x = torch.sigmoid(self.fc3(x))  # 最终输出二分类，使用sigmoid
        return x

# 定义模型的输入输出维度，隐藏层大小
input_size = inputs_resampled.shape[1]  # 输入的特征数量
hidden_size1 = 8  # 第一隐藏层的神经元数量
hidden_size2 = 4  # 第二隐藏层的神经元数量
output_size = 1   # 输出层的神经元数量，二分类任务

# 创建神经网络模型
model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化结果文件
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['InputNodes', 'LearningRate', 'HiddenNodes', 'OutputNodes', 'epochs', 
                  'PreBalance:ClassDistributionRatio', 'BalancingType', 'PostBalance:ClassDistributionRatio',
                  'truepositiverate', 'falsepositiverate', 'precision', 'recall', 'accuracy', 'f1score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# 训练模型
epochs = 100
for epoch in range(epochs):
    # 前向传播
    outputs = model(inputs_resampled)
    loss = criterion(outputs, labels_resampled)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算性能指标
    predictions = (outputs.detach().numpy() > 0.5).astype(int)
    true_positives = np.sum((predictions == 1) & (labels_resampled.numpy() == 1))
    false_positives = np.sum((predictions == 1) & (labels_resampled.numpy() == 0))
    false_negatives = np.sum((predictions == 0) & (labels_resampled.numpy() == 1))
    true_negatives = np.sum((predictions == 0) & (labels_resampled.numpy() == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(labels_resampled)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    true_positive_rate = recall
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    # 写入结果
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'InputNodes': input_size,
            'LearningRate': 0.001,
            'HiddenNodes': hidden_size1 + hidden_size2,
            'OutputNodes': output_size,
            'epochs': epoch + 1,
            'PreBalance:ClassDistributionRatio': 'Unbalanced',  # 假设原始数据不平衡
            'BalancingType': 'SMOTE',
            'PostBalance:ClassDistributionRatio': 'Balanced',  # 使用SMOTE平衡后的数据
            'truepositiverate': true_positive_rate,
            'falsepositiverate': false_positive_rate,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1score': f1_score
        })

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 保存模型权重
with open('model_weights_smote.pkl', 'wb') as f:
    pickle.dump(model.state_dict(), f)

print("训练完成，结果已保存至results.csv，模型权重已保存至model_weights_smote.pkl")
