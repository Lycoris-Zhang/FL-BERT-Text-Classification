import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_with_same_distribution(csv_path, output_dir, sizes, random_state=42):
    # 加载数据
    data = pd.read_csv(csv_path)

    # 确保提供的尺寸比例总和为1
    assert sum(sizes) == 1, "Sizes must sum to 1"

    # 获取数据的总数量
    total_samples = len(data)

    # 计算每个子集的标签分布
    label_counts = data['Label'].value_counts(normalize=True)

    # 为每个子集计算目标样本数
    part_sizes = [int(total_samples * size) for size in sizes]

    # 分割数据
    parts = []
    remaining_data = data.copy()
    for i, part_size in enumerate(part_sizes):
        if i == len(part_sizes) - 1:
            # 最后一个子集取所有剩余数据
            part = remaining_data
        else:
            # 按标签比例分割数据
            part, remaining_data = train_test_split(remaining_data, train_size=part_size,
                                                    stratify=remaining_data['Label'], random_state=random_state)

        # 保存到文件
        part.to_csv(f"{output_dir}/client{i+1}_nonEQ.csv", index=False)
        print(f"Part {i + 1} saved with {len(part)} records.")
        parts.append(part)

    return parts

# 示例使用
csv_path = '../dataset_THUCnews/thucnews_initial_titles.csv'  # 更新为实际路径
output_dir = '../dataset_client/dataset_nonEQ'  # 更新为实际路径
sizes = [0.1, 0.2, 0.3, 0.4]  # 各子集的数据量比例，总和必须为1
split_data_with_same_distribution(csv_path, output_dir, sizes)
