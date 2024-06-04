import numpy as np
import pandas as pd

def split_data_extremely(csv_path, output_dir, sizes, min_samples_per_label, min_total_samples_per_part, random_state=42):
    np.random.seed(random_state)
    data = pd.read_csv(csv_path)
    labels = data['Label'].unique()
    num_labels = len(labels)
    num_parts = len(sizes)
    assert sum(sizes) == 1, "Sizes must sum to 1"

    # 手动指定每个部分的主要标签
    dominant_labels_per_part = []
    all_labels = list(labels)
    for i in range(num_parts):
        dominant_labels = np.random.choice(all_labels, size=3, replace=False)
        dominant_labels_per_part.append(dominant_labels)
        # 移除已选择的主要标签，以避免重复
        all_labels = [label for label in all_labels if label not in dominant_labels]

    remaining_data = data.copy()
    parts = [pd.DataFrame() for _ in range(num_parts)]

    # 分配主要标签的75%数据
    for i in range(num_parts):
        dominant_labels = dominant_labels_per_part[i]
        for label in dominant_labels:
            label_data = remaining_data[remaining_data['Label'] == label].copy()
            num_label_data = len(label_data)
            if num_label_data > 0:
                num_samples = int(0.75 * num_label_data)  # 主要标签占比75%
                sampled_data = label_data.sample(n=num_samples, random_state=random_state)
                parts[i] = pd.concat([parts[i], sampled_data])
                remaining_data = remaining_data.drop(sampled_data.index)

    # 分配剩余25%的数据，确保每个数据集都有所有标签
    for i in range(num_parts):
        non_dominant_labels = [label for label in labels if label not in dominant_labels_per_part[i]]
        for label in non_dominant_labels:
            label_data = remaining_data[remaining_data['Label'] == label].copy()
            num_label_data = len(label_data)
            if num_label_data > 0:
                # 随机分配到其他数据集，不均匀分配
                num_samples = int(0.25 * num_label_data) // (num_parts - 1)
                for j in range(num_parts):
                    if j != i:
                        sampled_data = label_data.sample(n=num_samples, random_state=random_state)
                        parts[j] = pd.concat([parts[j], sampled_data])
                        label_data = label_data.drop(sampled_data.index)
                        remaining_data = remaining_data.drop(sampled_data.index)

    # 确保每个数据集都有所有标签的最小样本数
    for label in labels:
        label_data = remaining_data[remaining_data['Label'] == label].copy()
        for i in range(num_parts):
            if label not in dominant_labels_per_part[i]:
                sampled_data = label_data.sample(n=min_samples_per_label, random_state=random_state)
                parts[i] = pd.concat([parts[i], sampled_data])
                label_data = label_data.drop(sampled_data.index)
                remaining_data = remaining_data.drop(sampled_data.index)

    # 确保每个数据集的总数据量不低于30000，并增加数据量差异
    remaining_data = data.copy()
    for i in range(num_parts):
        additional_size = int(np.random.uniform(0.1, 0.3) * len(data))  # 随机增加数据量
        while len(parts[i]) < min_total_samples_per_part + additional_size:
            if len(remaining_data) == 0:
                break
            part_data = remaining_data.sample(n=min(1000, len(remaining_data)), random_state=random_state)
            parts[i] = pd.concat([parts[i], part_data])
            remaining_data = remaining_data.drop(part_data.index)

    # 保存分配好的数据
    for i, part in enumerate(parts, 1):
        part.to_csv(f"{output_dir}/client{i}_nonIID.csv", index=False)
        print(f"Part {i} saved with {len(part)} records.")

csv_path = '../dataset_THUCnews/thucnews_initial_titles.csv'
output_dir = '../dataset_client/dataset_nonIID'
sizes = [0.15, 0.20, 0.35, 0.30]
min_samples_per_label = 10
min_total_samples_per_part = 30000
split_data_extremely(csv_path, output_dir, sizes, min_samples_per_label, min_total_samples_per_part)
