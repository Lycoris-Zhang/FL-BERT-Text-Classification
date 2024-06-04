import torch
from transformers import BertTokenizer, BertModel
from model_config import BERT_FC

categories={0: "财经",1: "彩票",2: "房产",3: "股票",4: "家居",5: "教育",6: "科技",
            7: "社会",8: "时尚",9: "时政",10: "体育",11: "星座",12: "游戏",13: "娱乐"
           }

print("模型加载中...")
# 加载预训练的 BERT 模型和 tokenizer
bert_model = BertModel.from_pretrained('pretrained_BERT')
tokenizer = BertTokenizer.from_pretrained('pretrained_BERT')

# 初始化自定义的 BERT_FC 模型
model = BERT_FC(bert_model)

# 加载模型的权重
model_path = 'results/model/pytorch_model_client1.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式


input_text = input("输入待分类的标题文本: ")

# 使用 tokenizer 对输入文本进行编码
encoded_input = tokenizer.encode_plus(input_text,
                                      add_special_tokens=True,
                                      max_length=64,  # 根据你的数据调整长度
                                      return_attention_mask=True,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt'
                                     )

# 提取编码后的输入和注意力掩码
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# 使用模型进行预测
with torch.no_grad():
    prediction = model(input_ids, attention_mask)

# 输出预测结果
predicted_class = torch.argmax(prediction, dim=1)
classification_result=categories.get(predicted_class.item())
print("模型预测的分类为:", classification_result)
