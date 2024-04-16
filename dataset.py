import os
import torch
from torch.utils.data import Dataset, DataLoader


# 读取数据
def read_data(data_dir, file):
    file_name = os.path.join(data_dir, file)
    with open(file_name, 'r', encoding='utf8') as f:
        lines = f.readlines()
        inputs = [line.strip('\n').split('\t')[: -1] for line in lines]
        labels = [int(line.strip('\n').split('\t')[-1]) for line in lines]
    return inputs, labels


class TextPairDataset(Dataset):
    def __init__(self, data_dir, file_name, tokenizer, max_len, **kwargs):
        super(TextPairDataset, self).__init__(**kwargs)
        self.inputs, self.labels = read_data(data_dir=data_dir, file=file_name)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.tokenizer.encode(text=self.inputs[idx][0], text_pair=self.inputs[idx][1],
                                      add_special_tokens=True, padding='max_length',
                                      max_length=self.max_len, return_tensors='pt')
        label = self.labels[idx]

        return input.squeeze(0), torch.tensor(label, dtype=torch.long)


def load_data(data_dir, file_name, tokenizer, max_len, batch_size, shuffle):
    dataset = TextPairDataset(data_dir, file_name, tokenizer=tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader
