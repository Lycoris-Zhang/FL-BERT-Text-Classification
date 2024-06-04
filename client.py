import numpy as np
import csv
import sys
import torch
import chardet
import flwr as fl
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from model_config import BERT_FC
from torch.utils.tensorboard import SummaryWriter

#GPU引用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(sys.argv[2], 'rb') as file:
    # 读取一定数量的字节用于检测
    raw_data = file.read(5000)
    result = chardet.detect(raw_data)
    print(f"For Client{sys.argv[3]},Dataset detected encoding: {result['encoding']} with confidence {result['confidence']}")

df = pd.read_csv(sys.argv[2], encoding=result['encoding'])


print(f'For Client{sys.argv[3]},Dataset loaded in path:',sys.argv[2])

# 划分训练，验证，测试集
train_text, temp_text, train_labels, temp_labels = train_test_split(df['Title'], df['Label'],
                                                                    random_state=2024,
                                                                    test_size=0.3,
                                                                    stratify=df['Label']
                                                                   )

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2024,
                                                                test_size=0.5,
                                                                stratify=temp_labels
                                                               )

def tokenize_and_encode(texts, Tokenizer, max_len=25):
    """使用指定的tokenizer对文本序列进行分词和编码。

    Args:
        texts (list): 文本列表。
        Tokenizer (PreTrainedTokenizer): 预训练的分词器。
        max_len (int, optional): 最大序列长度。默认为25。

    Returns:
        dict: 包含输入ID和注意力掩码的字典。
    """
    return Tokenizer.batch_encode_plus(texts,
                                       max_length=max_len,
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors='pt'  # 直接返回PyTorch张量
                                      )

def getData():
    df = pd.read_csv("dataset_THUCnews/thucnews_test_data.csv")

    # 将训练数据集分割为训练、验证和测试集
    train_text, train_labels = df['Title'], df['Label']

    # 加载BERT分词器
    Tokenizer = BertTokenizerFast.from_pretrained("pretrained_BERT")

    # 对训练集中的序列进行分词和编码
    tokens_train = Tokenizer.batch_encode_plus(train_text.tolist(),
                                               max_length=25,
                                               padding=True,
                                               truncation=True
                                              )

    ## 将列表转换为张量
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    # 定义批次大小
    batch_size = 32

    # 封装张量
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # 为训练过程中的采样定义采样器
    train_sampler = RandomSampler(train_data)

    # 为训练集定义数据加载器
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader

# 引入BERT与分词器
BERT = AutoModel.from_pretrained("pretrained_BERT")
Tokenizer = BertTokenizerFast.from_pretrained("pretrained_BERT")

# 使用函数对训练集、验证集和测试集进行处理
tokens_train = tokenize_and_encode(train_text.tolist(), Tokenizer)
tokens_val = tokenize_and_encode(val_text.tolist(), Tokenizer)
tokens_test = tokenize_and_encode(test_text.tolist(), Tokenizer)

## 将列表转换为张量
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids']).clone().detach()
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# 定义一个批次的大小
batch_size = 50

# 封装张量
train_data = TensorDataset(train_seq, train_mask, train_y)

# 训练过程中的采样器
train_sampler = RandomSampler(train_data)

# 训练集的数据加载器
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# 封装张量
val_data = TensorDataset(val_seq, val_mask, val_y)

# 验证过程中的采样器
val_sampler = SequentialSampler(val_data)

# 验证集的数据加载器
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# 冻结所有参数
for param in BERT.parameters():
    param.requires_grad = False

# 将预训练的BERT传递给定义的架构
model = BERT_FC(BERT)
model = model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# 计算类别权重
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)

print(f"For Client{sys.argv[3]}:Dataset Class Weights:", class_weights)

# 将类别权重列表转换为张量
weights = torch.tensor(class_weights, dtype=torch.float)

# 将权重推送到GPU
weights = weights.to(device)

# 定义损失函数
cross_entropy = nn.NLLLoss(weight=weights)


KEY = [i for i in model.state_dict()]

state_dict = model.state_dict()
model_weight_rr = []

# 将状态字典的值转换为ndarrays
for key, value in state_dict.items():
    model_weight_rr.append(value.cpu().numpy())

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        print("\nGetting Params...")
        # 获取模型的状态字典
        state_dict = model.state_dict()
        model_weight = []
        # 将状态字典的值转换为ndarrays
        for key, value in state_dict.items():
            model_weight.append(value.cpu().numpy())
        return model_weight

    def fit(self, parameters, config):
        print("\nFitting Model...")
        new_weight_dict = {}
        # 从参数中加载新的权重到模型
        for i, key in enumerate(KEY):
            new_weight_dict[key] = torch.from_numpy(parameters[i])
        model.load_state_dict(new_weight_dict)

        model.train()

        tbwriter = SummaryWriter(log_dir=f'results/training_records/tensorboard_records/Client_{sys.argv[3]}')

        # 设定训练周期
        epochs = int(sys.argv[4])
        for epoch in range(epochs):
            total_loss, total_accuracy = 0, 0
            total_examples = 0
            total_ham = 0
            itr = 1
            # 遍历数据批次
            for step, batch in enumerate(train_dataloader):
                itr += 1
                # 将批次数据推送到GPU
                batch = [r.to(device) for r in batch]

                sent_id, mask, labels = batch

                # 清除之前的梯度
                model.zero_grad()

                # 获取当前批次的模型预测
                preds = model(sent_id, mask)

                # 计算实际值和预测值之间的损失
                loss = cross_entropy(preds, labels)

                # 累加总损失
                total_loss += loss.item()
                # 累加总的Hamming距离（可选，根据实际需要选择是否实现）
                total_ham += labels.sum().item()
                # 计算当前批次的准确率
                _, predicted = torch.max(preds.data, 1)
                total_accuracy += (predicted == labels).sum().item()

                # 反向传播计算梯度
                loss.backward()

                # 对梯度进行裁剪，以防止梯度爆炸问题
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # 更新参数
                optimizer.step()

                # 更新处理的总样本数
                total_examples += len(sent_id)

            # 计算本轮训练的平均损失和准确率
            avg_loss = total_loss / len(train_dataloader)
            accuracy = total_accuracy / total_examples
            # 打印"模型训练完成"
            row_data = [1, sys.argv[3], "Fit", epoch, avg_loss, accuracy]

            # 记录到TensorBoard
            tbwriter.add_scalar('Loss/train', avg_loss, epoch)
            tbwriter.add_scalar('Accuracy/train', accuracy, epoch)


            # 打印客户端编号、轮次、平均损失和准确率
            print('Client no.', sys.argv[3], 'Epoch :', epoch, 'Average Loss:', avg_loss, 'Accuracy :', accuracy)
        # 获取模型状态字典
        newacc = self.testEval()
        print('Client no.', sys.argv[3], "new Accuracy", newacc)
        print('Client no.', sys.argv[3], 'ham', total_ham / total_examples, 'Accuracy :', accuracy, total_examples)
        state_dict = model.state_dict()
        torch.save(model.state_dict(), f'results/model/pytorch_model_client{sys.argv[3]}.pth')
        print(f'Client{sys.argv[3]} Model Saved')
        model_weight = []
        # 将状态字典的值转换为ndarrays
        for key, value in state_dict.items():
            model_weight.append(value.cpu().numpy())

        tbwriter.close()

        # 返回模型权重、处理的总样本数和其他统计信息
        return model_weight, total_examples, {"loss": avg_loss, "newAccuracy": newacc, "ham": total_ham / total_examples, "oldAccuracy": accuracy}

    @staticmethod
    def testEval():
        # 禁用dropout层
        model.eval()

        total_loss, total_accuracy, total_correct = 0, 0, 0
        val_dataloader_n = getData()  # 获取验证数据集
        # 用于保存模型预测的空列表
        total_preds = []
        total_len = 0
        # 遍历批次
        itr = 1
        for step, batch in enumerate(val_dataloader_n):
            # 打印批次进度
            itr += 1
            # 每处理50个批次报告一次进度
            if step % 50 == 0 and not step == 0:
                # 报告进度
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # 将批次推送到GPU
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch

            # 禁用自动梯度计算
            with torch.no_grad():
                # 获取模型预测结果
                preds = model(sent_id, mask)

                # 计算验证损失
                loss = cross_entropy(preds, labels)
                total_loss += loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

                # 将预测结果转换为标签
                preds_labels = np.argmax(preds, axis=1)
                total_len += len(preds_labels)
                # 计算正确预测的数量
                correct = np.sum(preds_labels == labels.cpu().numpy())

                total_correct += correct

        # 计算本轮验证的平均损失和准确率
        avg_loss = total_loss / total_len
        accuracy = total_correct / total_len
        # 打印验证完成信息
        return accuracy

    def evaluate(self, parameters, config):
        print("\n评估中...")
        # 加载模型权重
        new_weight_dict = {}
        for i, key in enumerate(KEY):
            new_weight_dict[key] = torch.from_numpy(parameters[i])
        # 加载模型权重
        model.load_state_dict(new_weight_dict)

        # 禁用dropout层
        model.eval()

        total_loss, total_accuracy, total_correct = 0, 0, 0

        # 用来保存模型预测的空列表
        total_preds = []
        total_len = 0
        # 遍历批次
        itr = 1
        for step, batch in enumerate(val_dataloader):
            # 打印批次进度
            itr += 1
            # 每处理50个批次报告一次进度
            if step % 50 == 0 and not step == 0:
                # 报告进度
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # 将批次推送到GPU
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # 禁用自动梯度计算
            with torch.no_grad():
                # 获取模型预测结果
                preds = model(sent_id, mask)

                # 计算验证损失
                loss = cross_entropy(preds, labels)

                total_loss += loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

                # 将预测结果转换为标签
                preds_labels = np.argmax(preds, axis=1)
                total_len += len(preds_labels)
                # 计算正确预测的数量
                correct = np.sum(preds_labels == labels.cpu().numpy())

                total_correct += correct

        # 计算本轮验证的平均损失和准确率
        avg_loss = total_loss / total_len
        accuracy = total_correct / total_len

        # 重塑预测结果为(样本数, 类别数)的形状
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, len(val_dataloader), {"accuracy": accuracy}

# 启动Flower客户端
flower_client = FlowerClient()
# 将 NumPy 客户端转换为常规客户端
client = flower_client.to_client()
# 启动客户端
fl.client.start_client(server_address="localhost:" + str(sys.argv[1]), client=client)



