import pandas as pd

def check_empty_titles(csv_path):
    try:
        # 尝试使用UTF-8编码读取文件
        data = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试使用ISO-8859-1编码
            data = pd.read_csv(csv_path, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            # 如果ISO-8859-1也失败，尝试使用GBK编码，这对中文字符有用
            data = pd.read_csv(csv_path, encoding='gbk')

    # 检查Title列是否有空值
    empty_titles = data[data['Title'].isna()]

    # 如果有空值，打印相关信息
    if not empty_titles.empty:
        print("Found empty titles at the following rows:")
        for index, row in empty_titles.iterrows():
            print(f"Row {index}, Label: {row['Label']}")
    else:
        print("No empty titles found in the dataset.")

# 使用示例
csv_path = 'client4_IID.csv'  # 替换为您的CSV文件路径
check_empty_titles(csv_path)
