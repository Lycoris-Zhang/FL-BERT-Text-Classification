import pandas as pd

def count_label_samples(csv_files):
    for i, csv_path in enumerate(csv_files):
        # 加载数据
        data = pd.read_csv(csv_path)

        # 计算每个标签的样本数量
        label_counts = data['Label'].value_counts()

        # 计算标签占比
        total_samples = len(data)
        label_proportions = label_counts / total_samples * 100  # 转换为百分比

        # 打印每个文件的标签及其对应的样本数量和占比
        print(f"Label counts and proportions in file {i + 1} ({csv_path}):")
        print("Total number of samples in dataset:", total_samples)
        print(label_counts.to_string(header=None))  # 显示标签及其数量
        print("\nLabel proportions (%):")
        print(label_proportions.to_string(header=None))  # 显示标签及其占比
        print("\n" + "-" * 50 + "\n")

# 使用示例
csv_files = [
    'client1_nonEQ.csv',
    'client2_nonEQ.csv',
    'client3_nonEQ.csv',
    'client4_nonEQ.csv'
]
count_label_samples(csv_files)
