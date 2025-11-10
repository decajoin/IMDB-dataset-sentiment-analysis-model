import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from Dataset.dataset_loader import load_imdb_data  # 调用通用数据加载模块
from sklearn.metrics import accuracy_score
import time
from TextCNN_model import TextCNN

# ----------------------------
# 设备设置
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------------------
# 训练/验证函数
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy

# ----------------------------
# 主函数
# ----------------------------
def main():
    train_loader, test_loader, vocab_size, tokenizer = load_imdb_data(
        tokenizer_name='bert-base-uncased',
        batch_size=64,
        max_length=512
    )

    # ----------------------------
    # 模型初始化
    # ----------------------------
    model = TextCNN(
        vocab_size=vocab_size,
        embed_size=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=2,
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------
    # 训练循环
    # ----------------------------
    num_epochs = 10
    best_test_acc = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.1f}s")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'textcnn_imdb_best.pth')
            print(f"⭐ New best test accuracy: {best_test_acc:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'textcnn_imdb_final.pth')
    print("Training completed! Model saved.")

if __name__ == "__main__":
    main()
