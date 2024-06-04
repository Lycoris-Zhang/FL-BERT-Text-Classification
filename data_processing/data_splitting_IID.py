import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_path, output_dir):
    # 加载数据
    data = pd.read_csv(csv_path)

    # 输出每个类别的样本数，检查是否有类别只有一个样本
    print("Class distribution before splitting:")
    print(data['Label'].value_counts())

    # 检查每个类的样本数，确保每类至少有五个样本，否则不能用该类别进行分层抽样
    valid_labels = data['Label'].value_counts()[data['Label'].value_counts() >= 5].index
    filtered_data = data[data['Label'].isin(valid_labels)]

    # 分割数据为四份，使用分层抽样
    # 第一次分割出1/4的数据作为part1
    rest_data, part1 = train_test_split(filtered_data, test_size=0.25, stratify=filtered_data['Label'], random_state=50)

    # 从剩余数据中再分割出1/3（即原始数据的1/4）作为part2
    rest_data, part2 = train_test_split(rest_data, test_size=1/3, stratify=rest_data['Label'], random_state=50)

    # 继续分割剩余数据，分割出1/2作为part3
    rest_data, part3 = train_test_split(rest_data, test_size=0.5, stratify=rest_data['Label'], random_state=50)

    # 剩余的作为part4
    part4 = rest_data

    # 保存到文件
    part1.to_csv(f"{output_dir}/client1_IID.csv", index=False, encoding='utf-8')
    part2.to_csv(f"{output_dir}/client2_IID.csv", index=False, encoding='utf-8')
    part3.to_csv(f"{output_dir}/client3_IID.csv", index=False, encoding='utf-8')
    part4.to_csv(f"{output_dir}/client4_IID.csv", index=False, encoding='utf-8')
    print("Data split into four parts and saved to directory:", output_dir)

# 使用示例
csv_path = '../dataset_THUCnews/thucnews_initial_titles.csv'  # CSV文件路径
output_dir = '../dataset_client/dataset_IID'  # 输出目录
split_data(csv_path, output_dir)
