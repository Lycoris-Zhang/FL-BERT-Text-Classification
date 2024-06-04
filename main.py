import os
import subprocess
import json
from data_processing import test_data_sampling

#不允许tokenizer并行处理以防止客户端进程死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

dataset_path = config["dataset_path"]
record_save_path = config["record_save_path"]
server_file_path = config["server_file_path"]

clients = config["clients"]
epochs = config["epochs"]
rounds = config["rounds"]

top_client_count = config["top_client_count"]
clusters_count = config["clusters_count"]

client_datasets = [
    config["client1_dataset"],
    config["client2_dataset"],
    config["client3_dataset"],
    config["client4_dataset"]
]
if __name__ == '__main__':

    port = 5002

    test_data_sampling()

    client_file = "client.py"  # name of the file to run

    processes = []
    print(server_file_path)
    # 启动服务器进程
    processes.append(subprocess.Popen(["python", server_file_path] + [str(port), str(clusters_count), str(top_client_count), str(rounds)]))

    # 根据需要模拟的数量启动客户端进程
    for i in range(clients):
        dataset_path = client_datasets[i]  # 每个客户端使用其对应的数据集
        processes.append(subprocess.Popen(["python", client_file] + [str(port), dataset_path, str(i + 1), str(epochs)]))

    for process in processes:
        process.wait()


