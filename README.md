# Vuln-Analysis-NN
Third-Party API Javascript Vulnerability Analysis by Neural Network

```shell
Vuln-Analysis-NN/
│
├── data/                      # 数据存储目录
│   ├── raw/                   # 原始数据集
│   └── processed/             # 预处理后的数据
│
├── src/                       # 源代码文件夹
│   ├── data_preprocessing/     # 数据预处理模块
│   ├── models/                # 模型定义与训练模块
│   ├── training/              # 模型训练脚本
│   ├── testing/               # 模型评估与测试脚本
│   └── utils/                 # 常用工具函数
│
├── notebooks/                 # Jupyter notebooks 进行数据探索和可视化
│
├── config/                    # 配置文件
│   └── config.yaml            # 配置文件 (如超参数、路径、环境配置)
│
├── scripts/                   # 启动、自动化脚本（如运行模型、批处理任务）
│
├── logs/                      # 训练日志和模型保存
│   └── checkpoints/           # 模型检查点存储
│
├── tests/                     # 单元测试
│
├── README.md                  # 项目说明文件
├── requirements.txt           # 项目依赖的 Python 包列表
└── setup.py                   # 项目打包和依赖管理
```