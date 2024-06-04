import numpy as np
import sys
import json
import flwr as fl
from data_processing import test_data_sampling

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

proximal_mu = config["proximal_mu"]
fraction_fit = config["fraction_fit"]
fraction_evaluate = config["fraction_evaluate"]
min_fit_clients = config["min_fit_clients"]
min_evaluate_clients = config["min_evaluate_clients"]
min_available_clients = config["min_available_clients"]

class AggregationStrategy(fl.server.strategy.FedProx):

    def __init__(self, num_clusters: int = 2, top_n: int = 2):
        super().__init__(proximal_mu=proximal_mu,
                         fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients,
                         min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients
                        )
        self.num_clusters = num_clusters  # 聚类的数量
        self.top_n = top_n  # 每个聚类中选出性能最好的模型的数量
        self.lastResult = None  # 保存上一次的聚合结果

    def cluster_clients(self, results):
        # 根据客户端的hamming百分比来评估每个客户端的表现，将客户端分为 `self.num_clusters` 个聚类
        cluster_size = 100 // self.num_clusters
        clusters = [[] for _ in range(self.num_clusters + 1)]
        hamp = []
        for i in range(len(results)):
            hamp.append(results[i][1].metrics['ham'])
        print(hamp, len(clusters))
        minimum = min(hamp)
        maximum = max(hamp)
        diff = maximum - minimum
        for i in range(len(results)):
            cur = results[i][1].metrics['ham'] - minimum
            cur = (cur / diff) * 100
            print(cur, int(cur // cluster_size))
            clusters[int(cur // cluster_size)].append(results[i])
        return clusters

    def select_top_models(self, cluster):
        # 从每个聚类中选择表现最好的 `self.top_n` 个模型
        cluster.sort(key=lambda x: x[1].metrics['accuracy'])
        return cluster[:min(self.top_n, len(cluster))]

    def aggregate_fit(self,
                      rnd: int,
                      results,
                      failures,
                      ):
        # 聚合选择的模型的权重
        aggregated_weights = []
        aggregated_weights = super().aggregate_fit(rnd, results, failures)  # 使用超类的聚合方法
        if aggregated_weights:
            print(f"Round {rnd} - Weights Saving...")
            np.savez(f"results/weights/round-{rnd}-weights.npz", *aggregated_weights)
            print(f"Global Model {rnd} - Weights Saved")
        #完成聚合后重新采样评估数据集
        test_data_sampling()
        return aggregated_weights


# 创建聚合策略
strategy = AggregationStrategy(int(sys.argv[2]), int(sys.argv[3]))

# 启动 Flower 服务器，进行联邦学习
fl.server.start_server(server_address='localhost:' + str(sys.argv[1]),
                       config=fl.server.ServerConfig(num_rounds=int(sys.argv[4])),
                       grpc_max_message_length=1024 * 1024 * 1024,
                       strategy=strategy
                      )


