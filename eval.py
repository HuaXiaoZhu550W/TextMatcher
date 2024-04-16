import torch


# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0.0
    for batch in dataloader:
        inputs, labels = [x.to(device) for x in batch]
        with torch.no_grad():
            outputs = model(inputs)['pooler_output']  # 前向传播
        total_correct += torch.sum(torch.argmax(outputs, dim=-1) == labels)
    return total_correct/len(dataloader.dataset)
