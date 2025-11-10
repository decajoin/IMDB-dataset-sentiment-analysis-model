import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time
from Dataset.dataset_loader import load_imdb_data
from BiLSTM_model import BiLSTM

# ----------------------------
# 设备设置
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------------------
# 训练一个epoch
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return total_loss / len(dataloader), accuracy


# ----------------------------
# 模型评估
# ----------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
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

    vocab_size = tokenizer.vocab_size

    model = BiLSTM(
        vocab_size=vocab_size,
        embed_size=128,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")

    best_test_acc = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print("  Evaluating...")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_acc)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        epoch_time = time.time() - start_time
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        gap = train_acc - test_acc
        if gap > 0.05:
            print(f"  ⚠️  Overfitting detected! Train-Test gap: {gap:.4f}")
        else:
            print(f"  ✓ Good generalization, gap: {gap:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"  ⭐ New best test accuracy: {best_test_acc:.4f}")
            torch.save(model.state_dict(), 'bilstm_imdb_best.pth')

        print("-" * 60)

    print(f"\nTraining completed! Best test accuracy: {best_test_acc:.4f}")
    torch.save(model.state_dict(), 'bilstm_imdb.pth')
    print("Model saved as 'bilstm_imdb.pth'")


if __name__ == "__main__":
    main()
