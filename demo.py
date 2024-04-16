import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

result = ["两个句子不相似！", "两个句子相似！"]

# 模型参数文件
model_name = "../bert-base-chinese"  # 此处填写预训练模型权重保存地址

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(model_name)

# 加载模型并修改pooler.dense的输出大小
model = BertModel.from_pretrained(model_name)
model.pooler.dense = nn.Linear(in_features=768, out_features=2, bias=True)

model.load_state_dict(torch.load("../weight/model_weights.pth", map_location='cpu'))  # 加载训练好的模型权重

model.eval()

sentence1 = "我最爱吃牛肉拌饭"
sentence2 = "牛肉拌饭是我的最爱"

input_dict = tokenizer.encode_plus(text=sentence1, text_pair=sentence2, add_special_tokens=True, max_length=128,
                                   padding='max_length', truncation=True, return_tensors='pt')

input = input_dict['input_ids']
segment = input_dict['token_type_ids']
mask = input_dict['attention_mask']


with torch.no_grad():
    output = model(input, mask, segment)['pooler_output']

print(f"{result[torch.argmax(output, dim=-1).item()]}")
