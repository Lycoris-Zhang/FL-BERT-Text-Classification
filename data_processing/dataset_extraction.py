import os
import random
import pandas as pd

# THUCNEWS目录的路径
base_dir = ""

# 子文件夹名称及其对应的标签
categories = {
    "财经": 0, "彩票": 1, "房产": 2, "股票": 3, "家居": 4, "教育": 5,
    "科技": 6, "社会": 7, "时尚": 8, "时政": 9, "体育": 10, "星座": 11,
    "游戏": 12, "娱乐": 13
}

# 用于保存所有样本的列表
data_samples = []

if base_dir != '-1':
    print("Dataset Found.")

# 处理每个分类
for category, label in categories.items():
    folder_path = os.path.join(base_dir, category)
    all_files = os.listdir(folder_path)
    file_count = len(all_files)

    if file_count >= 30000:
        selected_files = random.sample(all_files, 30000)  # 随机选择10000个文件
    else:
        selected_files = all_files  # 文件不足10000，使用所有文件

    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            title = file.readline().strip()  # 读取第一行作为标题
            data_samples.append([label, title])

# 创建DataFrame并保存到CSV
df = pd.DataFrame(data_samples, columns=['Label', 'Title'])
# 随机打乱数据
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('thucnews_initial_titles.csv', index=False, encoding='utf-8')
