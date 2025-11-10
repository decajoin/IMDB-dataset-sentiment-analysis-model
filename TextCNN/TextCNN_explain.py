import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings
from  TextCNN_model import TextCNN

warnings.filterwarnings('ignore')

# ----------------------------
# 可解释性分析类
# ----------------------------
class TextCNNInterpreter:
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
        """使用遮挡方法计算每个token的重要性"""
        input_ids = self.preprocess_text(text).to(self.device)

        # 获取原始预测
        with torch.no_grad():
            original_output = self.model(input_ids)
            original_prob = torch.softmax(original_output, dim=1)

        if target_class is None:
            target_class = original_output.argmax(dim=1).item()

        original_score = original_prob[0, target_class].item()

        # 计算每个位置的重要性
        importance_scores = []
        tokens = input_ids[0].cpu().numpy()

        for i in range(len(tokens)):
            if tokens[i] in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                importance_scores.append(0.0)
                continue

            # 创建遮挡版本（用UNK token替换）
            masked_ids = input_ids.clone()
            masked_ids[0, i] = self.tokenizer.unk_token_id

            with torch.no_grad():
                masked_output = self.model(masked_ids)
                masked_prob = torch.softmax(masked_output, dim=1)
                masked_score = masked_prob[0, target_class].item()

            # 重要性 = 原始分数 - 遮挡后分数
            importance = original_score - masked_score
            importance_scores.append(importance)

        return np.array(importance_scores), target_class, original_score, original_prob[0].cpu().numpy()

    def visualize_importance(self, text, save_path=None):
        """可视化词重要性"""
        importance_scores, predicted_class, confidence, probs = self.get_token_importance_occlusion(text)

        # 获取tokens
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])

        # 过滤padding tokens
        valid_indices = [i for i, token in enumerate(tokens) if token != self.tokenizer.pad_token]
        tokens = [tokens[i] for i in valid_indices]
        importance_scores = importance_scores[valid_indices]

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 1. 词重要性热力图
        # 标准化重要性分数
        normalized_scores = (importance_scores - importance_scores.min()) / (
                    importance_scores.max() - importance_scores.min() + 1e-8)

        # 创建自定义颜色映射
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('importance', colors, N=256)

        # 绘制热力图
        importance_matrix = normalized_scores.reshape(1, -1)
        im = ax1.imshow(importance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # 设置x轴标签
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks([])
        ax1.set_title(
            f'Token Importance Heatmap \nPrediction: {"Positive" if predicted_class == 1 else "Negative"} (Confidence: {confidence:.3f})',
            fontsize=12, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Normalized Importance Score', fontsize=10)

        # 2. 重要性分数柱状图
        bars = ax2.bar(range(len(tokens)), importance_scores, alpha=0.7)

        # 根据重要性给柱子着色
        for i, bar in enumerate(bars):
            if importance_scores[i] > 0:
                bar.set_color('red')
                bar.set_alpha(0.7)
            else:
                bar.set_color('blue')
                bar.set_alpha(0.7)

        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Importance Score', fontsize=10)
        ax2.set_title('Token Importance Scores', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        # 打印详细分析结果
        self.print_analysis_results(text, tokens, importance_scores, predicted_class, confidence, probs)

    def print_analysis_results(self, text, tokens, importance_scores, predicted_class, confidence, probs):
        """打印详细分析结果"""
        print("=" * 80)
        print("TextCNN 模型可解释性分析结果")
        print("=" * 80)
        print(f"输入文本: {text}")
        print(f"预测结果: {'正面情感 😊' if predicted_class == 1 else '负面情感 😞'}")
        print(f"置信度: {confidence:.3f}")
        print(f"概率分布: 负面={probs[0]:.3f}, 正面={probs[1]:.3f}")
        print(f"分析方法: 遮挡法 (Occlusion)")
        print("-" * 80)

        # 找出最重要的词
        token_importance_pairs = list(zip(tokens, importance_scores))
        token_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("最重要的词汇 (按重要性排序):")
        print("-" * 40)
        for i, (token, score) in enumerate(token_importance_pairs[:10]):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"{i + 1:2d}. {token:15s} {score:8.4f}")

        print("-" * 80)

        # 统计信息
        positive_importance = [score for score in importance_scores if score > 0]
        negative_importance = [score for score in importance_scores if score < 0]

        print("统计信息:")
        print(f"   • 总token数量: {len(tokens)}")
        print(f"   • 正向重要性词汇: {len(positive_importance)} 个")
        print(f"   • 负向重要性词汇: {len(negative_importance)} 个")
        if positive_importance:
            print(f"   • 最大正向重要性: {max(positive_importance):.4f}")
        if negative_importance:
            print(f"   • 最大负向重要性: {min(negative_importance):.4f}")
        print("=" * 80)


# ----------------------------
# 主函数
# ----------------------------
def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # 初始化模型
    model = TextCNN(
        vocab_size=tokenizer.vocab_size,
        embed_size=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=2,
        dropout=0.5
    ).to(device)

    # 加载训练好的模型权重
    try:
        model.load_state_dict(torch.load('textcnn_imdb_best.pth', map_location=device))
        print("✅ 成功加载训练好的模型")
    except FileNotFoundError:
        print("❌ 未找到模型文件 'textcnn_imdb_best.pth'")
        print("请先运行训练脚本生成模型文件")
        return

    # 创建解释器
    interpreter = TextCNNInterpreter(model, tokenizer, device)

    # 预设测试评论
    sample_reviews = [
        "One of the best movies I've ever seen! Brilliant acting and incredible plot.",
        "I really hate this film. It's boring and the plot makes no sense at all.",
        "The movie was okay, nothing special but not terrible either. Average experience."
    ]

    print("\n" + "=" * 60)
    print("TextCNN 电影评论情感分析 - 可解释性工具")
    print("=" * 60)

    while True:
        print("\n请选择操作:")
        print("1. 分析预设评论")
        print("2. 输入自定义评论")
        print("3. 退出")

        choice = input("\n请输入选择 (1-3): ").strip()

        if choice == '1':
            print("\n预设评论列表:")
            for i, review in enumerate(sample_reviews, 1):
                print(f"{i}. {review}")

            try:
                idx = int(input(f"\n请选择评论 (1-{len(sample_reviews)}): ")) - 1
                if 0 <= idx < len(sample_reviews):
                    selected_review = sample_reviews[idx]

                    print(f"\n正在分析评论: {selected_review}")
                    print("使用方法: 遮挡法 (Occlusion)")
                    print("-" * 60)

                    interpreter.visualize_importance(selected_review)
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 请输入有效数字")

        elif choice == '2':
            custom_review = input("\n请输入您的评论: ").strip()
            if custom_review:
                print(f"\n正在分析您的评论: {custom_review}")
                print("使用方法: 遮挡法 (Occlusion)")
                print("-" * 60)

                interpreter.visualize_importance(custom_review)
            else:
                print("❌ 评论不能为空")

        elif choice == '3':
            print("\n再见！")
            break

        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()