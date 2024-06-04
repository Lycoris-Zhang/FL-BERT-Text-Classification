import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_extreme_proportions(num_labels, random_state=None):
    np.random.seed(random_state)
    # 随机选择几个主导标签，其余标签分配较少的比例
    dominant_labels = np.random.choice(num_labels, size=int(num_labels / 4), replace=False)  # 选择约1/4的标签作为主导
    proportions = np.zeros(num_labels)
    proportions[dominant_labels] = np.random.dirichlet(np.ones(len(dominant_labels)) * 10)  # 主导标签分配较高比例
    other_labels = [i for i in range(num_labels) if i not in dominant_labels]
    proportions[other_labels] = np.random.dirichlet(np.ones(len(other_labels)) * 1) * 0.1  # 其余标签分配较低比例
    return proportions / proportions.sum()  # 归一化确保总和为1


def split_data_extremely(csv_path, output_dir, num_parts=4, random_state=42):
    data = pd.read_csv(csv_path)
    labels = data['Label'].unique()
    num_labels = len(labels)

    # 先平均分成四份
    data_parts = np.array_split(data, num_parts)

    # 为每份数据生成极端的标签比例并重新抽样
    for i in range(num_parts):
        proportions = generate_extreme_proportions(num_labels, random_state + i)
        part_data = pd.DataFrame()
        part_size = len(data_parts[i])

        for idx, label in enumerate(labels):
            label_data = data_parts[i][data_parts[i]['Label'] == label]
            sample_size = int(proportions[idx] * part_size)
            sampled_data = label_data.sample(min(sample_size, len(label_data)), random_state=random_state)
            part_data = pd.concat([part_data, sampled_data])

        # 确保每部分数据量相等
        if len(part_data) < part_size:
            extra_samples = data_parts[i].sample(part_size - len(part_data), replace=True, random_state=random_state)
            part_data = pd.concat([part_data, extra_samples])

        # 保存到文件
        part_data.to_csv(f"{output_dir}/client{i + 1}_nonPP.csv", index=False)
        print(f"Part {i + 1} saved with {len(part_data)} records.")

# 示例使用
csv_path = '../dataset_THUCnews/thucnews_initial_titles.csv'  # 更新为实际路径
output_dir = '../dataset_client/dataset_nonPP'  # 更新为实际路径
split_data_extremely(csv_path, output_dir)
