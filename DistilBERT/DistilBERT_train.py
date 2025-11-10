import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
import time
import json

# ----------------------------
# 设备设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------
# 数据加载与预处理
# ----------------------------
def load_imdb_data(tokenizer, max_length=512, batch_size=16):
    """
    加载IMDB数据集，并进行tokenize
    """
    dataset = load_dataset("imdb")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    train_dataset = dataset["train"].map(tokenize, batched=True)
    test_dataset = dataset["test"].map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


# ----------------------------
# 训练一个epoch
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # 继续更新模型参数
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(logits, dim=1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), acc


# ----------------------------
# 模型评估
# ----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), acc


# ----------------------------
# 保存模型和配置信息
# ----------------------------
def save_model_and_config(model, tokenizer, filepath_prefix, best_acc, epoch):
    """
    保存模型权重和配置信息为.pth格式

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        filepath_prefix: 文件路径前缀
        best_acc: 最佳准确率
        epoch: 当前epoch
    """
    # 保存模型权重
    model_save_path = f"{filepath_prefix}.pth"
    torch.save(model.state_dict(), model_save_path)

    # 保存模型配置信息
    config_save_path = f"{filepath_prefix}_config.json"
    config_info = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 2,
        "max_length": 512,
        "best_accuracy": best_acc,
        "epoch": epoch,
        "vocab_size": tokenizer.vocab_size,
        "model_type": "DistilBertForSequenceClassification"
    }

    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    # 保存tokenizer
    tokenizer_save_path = f"{filepath_prefix}_tokenizer"
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"Model saved to: {model_save_path}")
    print(f"Config saved to: {config_save_path}")
    print(f"Tokenizer saved to: {tokenizer_save_path}")


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 加载tokenizer和数据
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader, test_loader = load_imdb_data(tokenizer)

    # 加载模型
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        # 在预训练的 DistilBERT 之上添加了一个随机初始化的分类头（全连接层），用于二分类（正/负情感）
        num_labels=2
    ).to(device)

    print("\nModel architecture:")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        # 当测试准确率连续 1 个 epoch（patience=1）不提升时，学习率减半（factor=0.5）。
        optimizer, mode="max", factor=0.5, patience=1
    )

    # 训练循环
    num_epochs = 3
    best_test_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 调整学习率
        scheduler.step(test_acc)

        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s")

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            print(f"  ⭐ New best test accuracy: {best_test_acc:.4f}")

            # 使用新的保存方式
            save_model_and_config(
                model=model,
                tokenizer=tokenizer,
                filepath_prefix="distilbert_imdb_best",
                best_acc=best_test_acc,
                epoch=best_epoch
            )

    # 保存最终模型
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")

    # 保存最终模型权重
    torch.save(model.state_dict(), "distilbert_imdb_final.pth")
    print("Final model saved to: distilbert_imdb_final.pth")


if __name__ == "__main__":
    main()