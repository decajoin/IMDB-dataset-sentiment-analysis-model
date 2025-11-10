import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
from BiLSTM_model import BiLSTM

warnings.filterwarnings('ignore')

# ----------------------------
# 可解释性分析类
# ----------------------------
class BiLSTMInterpreter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def preprocess_text(self, text, max_length=512):
        """文本预处理"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids']

    def get_token_importance_occlusion(self, text, target_class=None):
        """使用遮挡法计算每个token的重要性"""
        input_ids = self.preprocess_text(text).to(self.device)

        # 原始预测
        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = probs.argmax(dim=1).item()

        original_score = probs[0, target_class].item()

        importance_scores = []
        tokens = input_ids[0].cpu().numpy()

        for i in range(len(tokens)):
            token_id = tokens[i]
            # 忽略特殊符号
            if token_id in [
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id
            ]:
                importance_scores.append(0.0)
                continue

            # 遮挡该token（用UNK token替换）
            masked_ids = input_ids.clone()
            masked_ids[0, i] = self.tokenizer.unk_token_id

            with torch.no_grad():
                masked_logits = self.model(masked_ids)
                masked_probs = torch.softmax(masked_logits, dim=1)
                masked_score = masked_probs[0, target_class].item()

            importance = original_score - masked_score
            importance_scores.append(importance)

        return np.array(importance_scores), target_class, original_score, probs[0].cpu().numpy()

    def visualize_importance(self, text, save_path=None):
        """绘制词重要性可视化"""
        importance_scores, predicted_class, confidence, probs = self.get_token_importance_occlusion(text)

        # 获取tokens
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])

        # 过滤padding部分
        valid_indices = [i for i, t in enumerate(tokens) if t != self.tokenizer.pad_token]
        tokens = [tokens[i] for i in valid_indices]
        importance_scores = importance_scores[valid_indices]

        # 标准化
        norm_scores = (importance_scores - importance_scores.min()) / (
            importance_scores.max() - importance_scores.min() + 1e-8
        )

        # 自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('importance', ['blue', 'white', 'red'], N=256)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # (1) 热力图
        im = ax1.imshow(norm_scores.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks([])
        sentiment = "Positive 😊" if predicted_class == 1 else "Negative 😞"
        ax1.set_title(f"Token Importance Heatmap\nPrediction: {sentiment} (Confidence: {confidence:.3f})",
                      fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8, label='Normalized Importance')

        # (2) 柱状图
        bars = ax2.bar(range(len(tokens)), importance_scores, color='gray', alpha=0.7)
        for i, bar in enumerate(bars):
            if importance_scores[i] > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Token Importance (Occlusion Method)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        self.print_analysis_results(text, tokens, importance_scores, predicted_class, confidence, probs)

    def print_analysis_results(self, text, tokens, importance_scores, predicted_class, confidence, probs):
        """打印详细分析结果"""
        print("=" * 80)
        print("BiLSTM 模型可解释性分析结果")
        print("=" * 80)
        print(f"输入文本: {text}")
        print(f"预测结果: {'正面情感 😊' if predicted_class == 1 else '负面情感 😞'}")
        print(f"置信度: {confidence:.3f}")
        print(f"概率分布: 负面={probs[0]:.3f}, 正面={probs[1]:.3f}")
        print(f"分析方法: 遮挡法 (Occlusion)")
        print("-" * 80)

        token_pairs = list(zip(tokens, importance_scores))
        token_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        print("最重要的词汇 (按重要性排序):")
        print("-" * 40)
        for i, (token, score) in enumerate(token_pairs[:10]):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"{i + 1:2d}. {token:15s} {score:8.4f}")

        print("-" * 80)
        pos_scores = [s for s in importance_scores if s > 0]
        neg_scores = [s for s in importance_scores if s < 0]
        print(f"总token数: {len(tokens)}")
        print(f"   • 正向重要词: {len(pos_scores)} 个")
        print(f"   • 负向重要词: {len(neg_scores)} 个")
        if pos_scores:
            print(f"   • 最大正向重要性: {max(pos_scores):.4f}")
        if neg_scores:
            print(f"   • 最大负向重要性: {min(neg_scores):.4f}")
        print("=" * 80)


# ----------------------------
# 主函数
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    model = BiLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_size=128,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        bidirectional=True
    ).to(device)

    # 加载模型权重
    try:
        model.load_state_dict(torch.load('bilstm_imdb_best.pth', map_location=device))
        print("✅ 成功加载训练好的 BiLSTM 模型")
    except FileNotFoundError:
        print("❌ 未找到模型文件 'bilstm_imdb_best.pth'")
        return

    interpreter = BiLSTMInterpreter(model, tokenizer, device)

    # 示例评论
    sample_reviews = [
        "One of the best movies I've ever seen! Brilliant acting and incredible plot.",
        "I really hate this film. It's boring and the plot makes no sense at all.",
        "The movie was okay, nothing special but not terrible either. Average experience."
    ]

    print("\n" + "=" * 60)
    print("BiLSTM 电影评论情感分析 - 可解释性工具")
    print("=" * 60)

    while True:
        print("\n请选择操作:")
        print("1. 分析预设评论")
        print("2. 输入自定义评论")
        print("3. 退出")

        choice = input("\n请输入选择 (1-3): ").strip()

        if choice == '1':
            print("\n预设评论:")
            for i, review in enumerate(sample_reviews, 1):
                print(f"{i}. {review}")

            try:
                idx = int(input(f"\n请选择评论 (1-{len(sample_reviews)}): ")) - 1
                if 0 <= idx < len(sample_reviews):
                    print(f"\n正在分析评论: {sample_reviews[idx]}")
                    interpreter.visualize_importance(sample_reviews[idx])
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入有效数字")

        elif choice == '2':
            text = input("\n请输入自定义评论: ").strip()
            if text:
                print(f"\n正在分析评论: {text}")
                interpreter.visualize_importance(text)
            else:
                print("❌ 评论不能为空")

        elif choice == '3':
            print("\n再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()
