import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
import json
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistilBERTInterpreter:
    """DistilBERT模型可解释性分析器"""

    def __init__(self, model_path, config_path, tokenizer_path):
        """
        初始化解释器

        Args:
            model_path: 模型权重路径 (.pth文件)
            config_path: 配置文件路径 (.json文件)
            tokenizer_path: tokenizer保存路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

        # 加载模型
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=self.config['num_labels']
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        print(f"✅ 模型加载成功! 最佳准确率: {self.config['best_accuracy']:.4f}")

    def predict(self, text):
        """
        对单个文本进行预测

        Args:
            text: 输入文本

        Returns:
            prediction: 预测类别 (0=negative, 1=positive)
            probabilities: 各类别的概率
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()

        return prediction, probs.cpu().numpy()[0]

    def integrated_gradients_analysis(self, text, target_class=None):
        """
        使用Integrated Gradients进行归因分析

        Args:
            text: 输入文本
            target_class: 目标类别(None则使用预测类别)

        Returns:
            tokens: token列表
            attributions: 各token的归因分数
        """
        # 预处理
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 如果没有指定目标类别，使用预测类别
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                target_class = torch.argmax(outputs.logits, dim=1).item()

        # 定义前向传播函数
        def forward_func(input_ids, attention_mask):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.logits

        # 创建Integrated Gradients对象
        lig = LayerIntegratedGradients(forward_func, self.model.distilbert.embeddings)

        # 计算归因
        attributions, delta = lig.attribute(
            inputs=(input_ids, attention_mask),
            target=target_class,
            n_steps=50,
            return_convergence_delta=True,
            additional_forward_args=None
        )

        # 对embedding维度求和
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        # 获取tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return tokens, attributions

    def visualize_importance(self, text, save_path=None):
        """
        可视化词重要性（热力图+柱状图组合）

        Args:
            text: 输入文本
            save_path: 保存路径
        """
        # 获取预测和归因分数
        prediction, probs = self.predict(text)
        tokens, attributions = self.integrated_gradients_analysis(text)

        # 过滤padding tokens
        valid_pairs = [(tok, attr) for tok, attr in zip(tokens, attributions)
                       if tok != '[PAD]']

        if len(valid_pairs) > 60:
            valid_pairs = valid_pairs[:60]

        tokens = [pair[0] for pair in valid_pairs]
        attributions = np.array([pair[1] for pair in valid_pairs])

        # 标准化重要性分数到[0, 1]
        attr_min, attr_max = attributions.min(), attributions.max()
        if attr_max - attr_min > 1e-8:
            normalized_scores = (attributions - attr_min) / (attr_max - attr_min)
        else:
            normalized_scores = np.zeros_like(attributions)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # ========== 1. 词重要性热力图 ==========
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('importance', colors, N=256)

        importance_matrix = normalized_scores.reshape(1, -1)
        im = ax1.imshow(importance_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # 设置x轴标签
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks([])

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probs[prediction]
        ax1.set_title(
            f'Token Importance Heatmap\n'
            f'Prediction: {sentiment} (Confidence: {confidence:.3f})',
            fontsize=12, fontweight='bold', pad=15
        )

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.15, shrink=0.8)
        cbar.set_label('Normalized Importance Score', fontsize=10)

        # ========== 2. 重要性分数柱状图 ==========
        bars = ax2.bar(range(len(tokens)), attributions.tolist(), alpha=0.7)

        # 根据重要性给柱子着色
        for i, bar in enumerate(bars):
            if attributions[i] > 0:
                bar.set_color('red')
                bar.set_alpha(0.7)
            else:
                bar.set_color('blue')
                bar.set_alpha(0.7)

        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Importance Score', fontsize=10)
        ax2.set_title('Token Importance Scores',
                      fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化图已保存到: {save_path}")

        plt.show()

        # 打印详细分析结果
        self.print_analysis_results(text, tokens, attributions, prediction, probs)

    def print_analysis_results(self, text, tokens, attributions, prediction, probs):
        """打印详细分析结果"""
        print("\n" + "=" * 80)
        print("DistilBERT 模型可解释性分析结果")
        print("=" * 80)
        print(f"输入文本: {text}")
        print(f"预测结果: {'正面情感 😊' if prediction == 1 else '负面情感 😞'}")
        print(f"置信度: {probs[prediction]:.4f}")
        print(f"概率分布: 负面={probs[0]:.4f}, 正面={probs[1]:.4f}")
        print(f"分析方法: 积分梯度法")
        print("-" * 80)

        # 找出最重要的词
        token_attr_pairs = [(tok, attr) for tok, attr in zip(tokens, attributions)
                            if tok not in ['[CLS]', '[SEP]', '[PAD]']]
        token_attr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("最重要的词汇 (按重要性排序):")
        print("-" * 60)
        for i, (token, score) in enumerate(token_attr_pairs[:15], 1):
            direction = "正向" if score > 0 else "负向"
            print(f"{i:2d}. {token:15s} {score:+8.4f}  ({direction})")

        print("-" * 80)

        # 统计信息
        positive_attrs = [attr for attr in attributions if attr > 0]
        negative_attrs = [attr for attr in attributions if attr < 0]

        print("统计信息:")
        print(f"   • 总token数量: {len(tokens)}")
        print(f"   • 正向贡献词汇: {len(positive_attrs)} 个")
        print(f"   • 负向贡献词汇: {len(negative_attrs)} 个")
        if positive_attrs:
            print(f"   • 最大正向归因: {max(positive_attrs):.4f}")
        if negative_attrs:
            print(f"   • 最大负向归因: {min(negative_attrs):.4f}")
        print("=" * 80 + "\n")


def main():
    """主函数：交互式可解释性分析"""

    print("=" * 80)
    print("DistilBERT 情感分析 - 可解释性工具")
    print("=" * 80)
    print(f"使用设备: {device}")
    print()

    # 初始化解释器
    try:
        interpreter = DistilBERTInterpreter(
            model_path="distilbert_imdb_best.pth",
            config_path="distilbert_imdb_best_config.json",
            tokenizer_path="distilbert_imdb_best_tokenizer"
        )
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到模型文件")
        print(f"   {e}")
        return

    # 预设测试评论
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Terrible film. Waste of time and money. I've never been so bored in my life.",
        "The movie had some good moments, but overall it was just okay. Nothing special.",
        "One of the best films I've ever seen! Brilliant performances and an incredible storyline.",
        "I really hate this film. It's boring and the plot makes no sense at all.",
    ]

    while True:
        print("\n" + "=" * 60)
        print("请选择操作:")
        print("=" * 60)
        print("1. 分析预设评论")
        print("2. 输入自定义评论")
        print("3. 批量分析预设评论")
        print("4. 退出")
        print("=" * 60)

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == '1':
            print("\n预设评论列表:")
            print("-" * 60)
            for i, review in enumerate(sample_reviews, 1):
                print(f"{i}. {review}")
            print("-" * 60)

            try:
                idx = int(input(f"\n请选择评论 (1-{len(sample_reviews)}): ")) - 1
                if 0 <= idx < len(sample_reviews):
                    selected_review = sample_reviews[idx]

                    print(f"\n正在分析评论...")
                    print(f"{selected_review}")
                    print("-" * 60)

                    interpreter.visualize_importance(selected_review)
                else:
                    print("❌ 无效选择，请重新输入")
            except ValueError:
                print("❌ 请输入有效数字")
            except Exception as e:
                print(f"❌ 分析出错: {e}")

        elif choice == '2':
            custom_review = input("\n请输入您的评论: ").strip()
            if custom_review:
                print(f"\n正在分析您的评论...")
                print(f"{custom_review}")
                print("-" * 60)

                try:
                    interpreter.visualize_importance(custom_review)
                except Exception as e:
                    print(f"❌ 分析出错: {e}")
            else:
                print("❌ 评论不能为空")

        elif choice == '3':
            print(f"\n🔍 批量分析 {len(sample_reviews)} 条预设评论...")
            print("-" * 60)

            for i, review in enumerate(sample_reviews, 1):
                print(f"\n[{i}/{len(sample_reviews)}] 分析中...")
                print(f"{review[:80]}..." if len(review) > 80 else f"{review}")

                try:
                    interpreter.visualize_importance(review)
                except Exception as e:
                    print(f"❌ 分析出错: {e}")
                    continue

            print("\n✅ 批量分析完成!")

        elif choice == '4':
            print("\n再见！")
            break

        else:
            print("❌ 无效选择，请输入 1-4")


if __name__ == "__main__":
    main()