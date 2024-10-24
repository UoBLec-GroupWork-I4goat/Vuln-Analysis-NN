# 加载必要的库
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# 定义神经网络模型（与训练时相同）
class ComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 预测方法
def predict_and_save(model, input_data,real_labels, output_file):
    # 模型设置为评估模式
    model.eval()
    
    # 将输入数据转换为Tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(input_tensor)
        # 将预测结果转换为0和1（二分类）
        predicted_labels = (predictions >= 0.5).float().numpy().flatten()
   
    # 将真实标签也进行展平操作，确保为一维数组
    real_labels = real_labels.flatten()
    
    # 计算预测是否正确
    accuracy = (predicted_labels == real_labels)
    accuracy_str = np.where(accuracy, 'true', 'false')
    
    # 将结果保存到DataFrame
    results_df = pd.DataFrame({
        'real_value': real_labels,
        'predict_value': predicted_labels,
        'accuracy': accuracy_str
    })

    # 计算整体的准确率
    overall_accuracy = np.mean(accuracy) * 100  # 转换为百分比
    
    # 写入预测结果到CSV文件
    with open(output_file, 'w') as f:
        results_df.to_csv(f, index=False)
        f.write(f"\nTotal Predictions: {len(real_labels)}, Overall Accuracy: {overall_accuracy:.2f}%\n")
    
    print(f"预测结果已保存到 {output_file}，总预测数量为 {len(real_labels)}，整体准确率为 {overall_accuracy:.2f}%")

# 预测部分的主要流程
def main():
    # 加载数据（与训练时的预处理步骤相同）
    data = pd.read_csv('./grounddata/_sample.csv')
    categorical_input = data[['B']].values  # B列为类别输入
    numerical_input = data[['A', 'C', 'D']].values  # A、C、D列为数值输入
    real_labels = data[['Vulnerability_Truth']].values  # 真实标签

    # 对类别输入进行one-hot编码
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(categorical_input)
    
    # 合并编码后的类别输入与数值输入
    inputs = np.hstack((encoded_categorical, numerical_input))

    # 加载最佳模型参数
    with open('best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)

    input_size = inputs.shape[1]
    hidden_size1 = best_params['hidden_size1']
    hidden_size2 = best_params['hidden_size2']
    output_size = 1  # 二分类输出

    # 创建模型实例并加载保存的最佳模型权重
    model = ComplexNN(input_size, hidden_size1, hidden_size2, output_size)
    model.load_state_dict(torch.load('best_model_weights.pkl',weights_only=False))
    
    # 进行预测并保存结果
    predict_and_save(model, inputs, real_labels, 'predictions.csv')

if __name__ == "__main__":
    main()