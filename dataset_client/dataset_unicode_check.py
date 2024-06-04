import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        # 读取一定数量的字节用于检测
        raw_data = file.read(5000)
        result = chardet.detect(raw_data)
        print(f"Detected encoding: {result['encoding']} with confidence {result['confidence']}")

# 使用示例
csv_path = 'dataset_nonPP/client1_nonPP.csv'  # 替换为你的CSV文件路径
detect_encoding(csv_path)
