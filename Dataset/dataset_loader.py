# dataset_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class IMDBDataset(Dataset):
    """自定义Dataset，用于IMDB文本分类"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_imdb_data(tokenizer_name='bert-base-uncased', batch_size=32, max_length=512):
    """加载IMDB数据集，并返回DataLoader"""
    print("Loading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # tokenier: 把文本转化为token（分词 + 映射到词表）
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size

    return train_loader, test_loader, vocab_size, tokenizer
