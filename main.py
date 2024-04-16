import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from train import train
from eval import evaluate
from transformers import BertTokenizer, BertModel


# 超参数
data_dir = "../lcqmc"  # 数据集存储位置
train_name = 'train.tsv'
dev_name = 'dev.tsv'
batch_size = 16
max_len = 128
lr = 1e-5
epochs = 5
model_name = "../bert-base-chinese"  # 预训练模型存储位置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载分词器
tokenizer = BertTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = BertModel.from_pretrained(model_name)
model.pooler.dense = nn.Linear(in_features=768, out_features=2, bias=True)

# 初始化输出层参数
nn.init.xavier_uniform_(model.pooler.dense.weight)

# 加载数据集
train_loader = load_data(data_dir, train_name, tokenizer, max_len, batch_size, shuffle=True)
dev_loader = load_data(data_dir, dev_name, tokenizer, max_len, batch_size, shuffle=False)

# 优化器
optimizer = optim.AdamW(params=model.parameters(), lr=lr)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 原模型未训练在验证集合上的表现
accuracy = evaluate(model, dev_loader, device)
print(f"accuracy: {accuracy: .2%}")

best_acc = 0.0
for epoch in range(epochs):
    train_acc = train(model, train_loader, optimizer, loss_fn, device, epoch)
    dev_acc = evaluate(model, dev_loader, device)
    print(f"train_acc: {train_acc:.2%} --*-- dev_acc: {dev_acc:.2%}")
    if dev_acc >= best_acc:
        torch.save(model.state_dict(), 'weight/model_weights.pth')
        best_acc = dev_acc
