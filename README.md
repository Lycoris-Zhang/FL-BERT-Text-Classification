# 这是一个基于Flower联邦学习框架，BERT预训练模型与Pytorch的联邦学习中文文本分类项目FL-BERT-Text-Classification
Github页面：https://github.com/Lycoris-Zhang/FL-BERT-Text-Classification
该项目是基于samadeep的联邦学习垃圾邮件分类项目完成的。详情可见：https://github.com/samadeep/federated_learning_BERT
如果您需要进行英文文本分类，将BERT模型替换为英文版即可：）
## 项目结构
|-- dataset_client
    |-- dataset_unicode_check.py
    |-- empty_texts_check.py
|-- dataset_THUCnews
    |-- thucnews_initial_titles.csv
    |-- thucnews_test_data.csv
|-- data_processing
    |-- dataset_extraction.py
    |-- data_splitting_IID.py
    |-- data_splitting_nonEQ.py
    |-- data_splitting_nonIID.py
    |-- data_splitting_nonPP.py
|-- pretrained_BERT
    |-- config.json
    |-- pytorch_model.bin
    |-- README.md
    |-- tokenizer.json
    |-- tokenizer_config.json
    |-- vocab.txt
|-- results
    |-- model
    |-- training_records
    |-- weights
|-- data_processing.py
|-- main.py
|-- model_config.py
|-- server_FedAvg.py
|-- server_FedOpt.py
|-- server_FedProx.py
|-- README.md
|-- BERT-FC-demo.py
|-- client.py
|-- config.json

## 您将在列表中看到如下文件夹：
- **dataset_THUCnews** 这是经过第一次预处理的文本数据集的存放位置，我使用的数据集为THUCnews，数据集下载请见：http://thuctc.thunlp.org/。
- **pretrained_BERT** 这是预训练的BERT模型的存放位置，本项目不涉及BERT的预训练工作，您可以选择直接在线使用transformers库中的模型进行训练，那样的话您可以不用关心该文件夹的内容，如果您需要下载BERT模型到本地，请放在这里。
- **results** 这是存放训练过程中的参数（损失函数，正确率）的Tensorboard记录文件，每一轮模型聚合的权重(.npz)，与最终全局模型结构文件(.pth)的位置。该文件夹下有3个子文件夹：**model**, **training_records**和**weight**，用以存放上述文件。
- **dataset_client** 存放不同客户端使用的数据集，以及空字符串检查代码**empty_texts_check.py**，和文件编码检查代码**dataset_unicode_check.py**
- **data_processing** 用于数据集抽取，各个客户端数据集划分的代码存放处，包含从原始THUCnews数据集中抽取数据的**dataset_extraction.py**，以及进行四种不同数据分布划分的**data_splitting_IID.py**，**data_splitting_nonEQ.py**，**data_splitting_nonIID.py**与**data_splitting_nonPP.py**。

## 以及如下的.py文件：
- **BERT-FC_demo** 该文件用于使用最终全局模型进行实际预测。
- **client_config** 进行训练的客户端配置与代码内容。
- **server_FedAvg/FedProx/FedOpt** 进行模型聚合的三种聚合算法文件，包含服务器配置与代码内容。
- **model_config** 训练的模型结构。
- **data_processing** 额外从数据集中抽取验证集的代码，主要用于全局模型评估。
- **main** 项目启动点

## 如何使用它？
所有的模型训练参数与联邦学习模式设定参数都存放在**config.json**中，您可以在该文件中调整所有参数，包括使用的数据集位置，模型训练学习率，模拟的客户端数量等。
如果您需要使用其他Flower支持的聚合算法进行联邦学习，您需要修改server.py中的代码。
完成上述准备后，您就可以进行本地模拟的联邦学习了：），如果您想进行实际的远程联邦学习，您需要修改一下代码。

# This is a federated learning Chinese text classification project called 'FL-BERT-Text-Classification,' which is based on the Flower federated learning framework, BERT pre-trained models, and PyTorch.
GitHub page: https://github.com/Lycoris-Zhang/FL-BERT-Text-Classification
This project is based on samadeep's federated learning spam classification project. For more details, please visit: https://github.com/samadeep/federated_learning_BERT.
If you need to perform English text classification, just replace the BERT model with the English version :)

## Project Structure
|-- dataset_client
    |-- dataset_unicode_check.py
    |-- empty_texts_check.py
|-- dataset_THUCnews
    |-- thucnews_initial_titles.csv
    |-- thucnews_test_data.csv
|-- data_processing
    |-- dataset_extraction.py
    |-- data_splitting_IID.py
    |-- data_splitting_nonEQ.py
    |-- data_splitting_nonIID.py
    |-- data_splitting_nonPP.py
|-- pretrained_BERT
    |-- config.json
    |-- pytorch_model.bin
    |-- README.md
    |-- tokenizer.json
    |-- tokenizer_config.json
    |-- vocab.txt
|-- results
    |-- model
    |-- training_records
    |-- weights
|-- data_processing.py
|-- main.py
|-- model_config.py
|-- server_FedAvg.py
|-- server_FedOpt.py
|-- server_FedProx.py
|-- README.md
|-- BERT-FC-demo.py
|-- client.py
|-- config.json

## You will see the following folders in the list:
- **dataset_THUCnews** This is the storage location of the preprocessed text dataset. I am using the THUCnews dataset. For dataset download, please visit: http://thuctc.thunlp.org/.
- **pretrained_BERT** This is where the pre-trained BERT models are stored. This project doesn't involve BERT pre-training. You can directly use the models in the transformers library for training online, which would mean you don't need to worry about the contents of this folder.If you need to download the BERT model locally, please place it here.
- **results** This directory contains the Tensorboard log files of training process parameters (loss function, accuracy), the weights (.npz) for each round of model aggregation, and the final global model structure file (.pth). The folder includes three subdirectories: **model**, **training_records**, and **weight**, which are used to store the aforementioned files.
- **dataset_client** directory contains the datasets used by different clients, as well as the empty string check code **empty_texts_check.py**, and the file encoding check code **dataset_unicode_check.py**.
- **data_processing**: This directory contains the code for dataset extraction and partitioning of client datasets. It includes **dataset_extraction.py** for extracting data from the original THUCnews dataset, and **data_splitting_IID.py**, **data_splitting_nonEQ.py**, **data_splitting_nonIID.py**, and **data_splitting_nonPP.py** for performing four different types of data distribution partitioning.

## And the following '.py' files:
- **BERT-FC_demo**: This file is used to make actual predictions using the final global model.
- **client_config**: The configuration and code for training clients.
- **server_FedAvg/FedProx/FedOpt**: Files for the three model aggregation algorithms, including server configuration and code content.
- **model_config**: The structure of the training model.
- **data_processing**: Code for extracting a validation set from the dataset, primarily used for evaluating the global model.
- **main**: Project entry point.
 
## How to use it?
All model training parameters and federated learning configuration settings are stored in **config.json**. You can adjust all parameters in this file, including the dataset location, model training learning rate, number of simulated clients, and more.
If you want to use other aggregation algorithms supported by Flower for federated learning, you need to modify the code in **server_FedAvg/FedProx/FedOpt.py**.
After completing the above preparations, you can proceed with local simulated federated learning:). If you wish to conduct actual remote federated learning, you will need to make some additional modifications to the code.
