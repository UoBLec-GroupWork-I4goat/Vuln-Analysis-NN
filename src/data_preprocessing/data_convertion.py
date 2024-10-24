import json
import csv
import os

# 读取JSON文件时指定编码为utf-8
file_path = 'D:/Cyber security/groupwork2/53/0.json'  # 替换为你的JSON文件路径

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取nodedata/graphData中的data数组
    graph_data_arrays = []
    if 'nodedata' in data and 'graphData' in data['nodedata']:
        graph_data = data['nodedata']['graphData']
        if 'depnodes' in graph_data:
            for node in graph_data['depnodes']:
                if 'data' in node:
                    # 提取每个data项
                    item = node['data']
                    # 检查是否存在vulnerabilities字段，并将其拆分
                    if 'vulnerabilities' in item:
                        vul = item.pop('vulnerabilities')  # 移除原来的vulnerabilities字段
                        # 拆分并添加为单独的字段
                        item['vul.info'] = vul.get('info', 0)
                        item['vul.low'] = vul.get('low', 0)
                        item['vul.moderate'] = vul.get('moderate', 0)
                        item['vul.high'] = vul.get('high', 0)
                        item['vul.critical'] = vul.get('critical', 0)
                    graph_data_arrays.append(item)

    # 如果提取到数据，生成CSV
    if graph_data_arrays:
        # 打印提取到的数据，调试使用
        print("提取到的数据: ", graph_data_arrays)

        # 准备写入CSV的字段名（横行）
        fieldnames = graph_data_arrays[0].keys()

        # 指定输出路径
        output_path = os.path.join(os.getcwd(), 'output.csv')  # 当前工作目录下生成CSV文件

        # 写入CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # 写入横行 (字段名)

            # 写入每一行的数据（纵行）
            for data_row in graph_data_arrays:
                writer.writerow(data_row)

        print(f"CSV文件生成成功！文件路径: {output_path}")

    else:
        print("没有提取到任何数据，检查 graphData 结构")

except FileNotFoundError:
    print(f"文件未找到，请检查路径: {file_path}")
except json.JSONDecodeError:
    print("无法解析JSON文件，可能是文件格式错误")
except Exception as e:
    print(f"发生错误: {e}")
