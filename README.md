# LCQMC Sentence Pair Matching

## 简介
本项目是针对LCQMC数据集的句子对匹配任务。LCQMC是一个广泛用于中文自然语言处理的数据集，其中包含了大量的中文句子对，每个句子对都标记为相似或不相似。

## 文件结构
- `main.py`: 主文件，用于运行整个项目。
- `train.py`: 训练模型。
- `dataset.py`: 处理和加载数据集。
- `eval.py`: 评估模型性能。
- `demo.py`: 一个简单的演示，展示如何使用训练好的模型。

## 如何运行
1. 确保已安装所有必要的Python库（请参考源代码）。
2. 使用`train.py`训练模型。
3. 使用`eval.py`评估模型性能。
4. 使用`demo.py`进行简单的演示。

## 安装依赖
- numpy  1.23.5
- pandas  1.5.3
- torch   2.1.2
- transformers  4.39.0
