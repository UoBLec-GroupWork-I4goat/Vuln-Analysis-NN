import json

# 递归函数来遍历整个JSON结构，查找graphData
def find_key(data, target_key, path=""):
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}/{key}" if path else key
            if key == target_key:
                print(f"找到了 '{target_key}'，路径为: {new_path}")
                print(f"{target_key} 的内容为: {value}")
            else:
                find_key(value, target_key, new_path)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = f"{path}[{index}]"
            find_key(item, target_key, new_path)

# 读取JSON文件
file_path = 'D:/Cyber security/groupwork2/53/0.json'  # 替换为你的JSON文件路径
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 查找 graphData
    find_key(data, 'graphData')

except FileNotFoundError:
    print(f"文件未找到，请检查路径: {file_path}")
except json.JSONDecodeError:
    print("无法解析JSON文件，可能是文件格式错误")
except Exception as e:
    print(f"发生错误: {e}")
