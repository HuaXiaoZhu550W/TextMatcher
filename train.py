import torch
from tqdm import tqdm


# 训练函数
def train(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_correct = 0.0
    total_loss = 0.0
    iterations = len(dataloader)

    # 创建进度条
    pbar = tqdm(desc=f"epoch: {epoch + 1}", total=iterations, postfix=dict, mininterval=0.4)
    for iteration, batch in enumerate(dataloader):
        inputs, labels = [x.to(device) for x in batch]
        outputs = model(inputs)['pooler_output']  # 前向传播
        optimizer.zero_grad()  # 梯度清零
        loss = loss_fn(outputs, labels)  # 模型已经计算了loss, 详情见源码
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(outputs, dim=-1) == labels)

        pbar.set_postfix(**{'loss': f"{total_loss / (iteration + 1):.4f}"})
        pbar.update(1)
    pbar.close()
    return total_correct / len(dataloader.dataset)
