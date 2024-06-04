import torch.nn as nn

#在这里定义需要载入的模型架构，这里是BERT+全连接层，BERT的导入与模型引用可见client_config
class BERT_FC(nn.Module):

    def __init__(self, BERT):
        super(BERT_FC, self).__init__()
        self.BERT = BERT
        self.Dropout = nn.Dropout(0.1)
        self.ReLU = nn.ReLU()
        self.FC1 = nn.Linear(768, 512)
        self.FC2 = nn.Linear(512,512)
        self.FC3 = nn.Linear(512, 14)
        self.Softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.BERT(sent_id, attention_mask=mask, return_dict=False)
        x = self.FC1(cls_hs)
        x = self.ReLU(x)
        x = self.Dropout(x)
        x = self.FC2(x)
        x = self.ReLU(x)
        x = self.Dropout(x)
        x = self.FC3(x)
        x = self.Softmax(x)

        return x