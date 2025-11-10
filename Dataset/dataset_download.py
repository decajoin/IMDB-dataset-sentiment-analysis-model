from datasets import load_dataset

# 下载 IMDB 数据集
dataset = load_dataset("stanfordnlp/imdb")

# 查看数据集结构
print(dataset)

# 查看训练集第一个样本
print(dataset['train'][0])
