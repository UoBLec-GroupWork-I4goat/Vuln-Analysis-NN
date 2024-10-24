import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
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
inputs_resampled = torch.tensor(inputs_resampled, dtype=torch.float32)
labels_resampled = torch.tensor(labels_resampled, dtype=torch.float32).unsqueeze(1)  # 调整 labels 的形状

# 定义神经网络模型
class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        
        # 定义网络的层次结构
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(hidden_size2, output_size)  # 第二隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 激活函数relu
        x = torch.relu(self.fc2(x))  # 激活函数relu
        x = torch.sigmoid(self.fc3(x))  # 最终输出二分类，使用sigmoid
        return x

# 定义模型训练和评估的函数
def train_and_evaluate(hidden_size1, hidden_size2):
    model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 模型训练部分
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs_resampled)
        loss = criterion(outputs, labels_resampled)
        loss.backward()
        optimizer.step()

    # 模型评估部分
    with torch.no_grad():
        model.eval()
        predictions = (model(inputs_resampled) >= 0.5).float()  # 预测结果
        accuracy = (predictions == labels_resampled).sum().item() / len(labels_resampled)
    
    return accuracy  # 返回准确率

# 设置输入输出维度和训练参数
input_size = inputs_resampled.shape[1]  # 输入的特征数量
output_size = 1   # 输出层的神经元数量，二分类任务
epochs = 100  # 训练轮次

# 定义参数网格
param_grid = {
    'hidden_size1': [8, 16, 32],
    'hidden_size2': [4, 8, 16]
}

# 执行网格搜索
best_accuracy = 0
best_params = None
for hidden_size1 in param_grid['hidden_size1']:
    for hidden_size2 in param_grid['hidden_size2']:
        accuracy = train_and_evaluate(hidden_size1, hidden_size2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2}

print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy}")

# 保存最佳模型权重
model = ComplexNN(input_size, best_params['hidden_size1'], best_params['hidden_size2'], output_size)
torch.save(model.state_dict(), 'best_model_weights.pkl')
print("最佳模型权重已保存至 'best_model_weights.pkl'")

with open('best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
print("最佳模型参数已保存至 'best_params.pkl'")
