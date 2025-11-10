import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer


def analyze_length(tokenizer_name="bert-base-uncased", split="train", max_samples=None,
                   save_fig_path="imdb_length_distribution.png"):
    # 加载数据集
    dataset = load_dataset("stanfordnlp/imdb")[split]
    texts = dataset["text"]

    if max_samples:
        texts = texts[:max_samples]

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 统计长度
    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]

    # 转 numpy 方便分析
    lengths = np.array(lengths)

    # 打印统计信息
    print(f"使用 tokenizer: {tokenizer_name}")
    print(f"数据集: {split}, 样本数: {len(lengths)}")
    print("-" * 50)
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"最大长度: {np.max(lengths)}")
    print(f"95% 分位数: {np.percentile(lengths, 95):.2f}")
    print(f"99% 分位数: {np.percentile(lengths, 99):.2f}")

    # 画直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Token Length")
    plt.ylabel("Count")
    plt.title(f"IMDB {split} Dataset text length distribution ({tokenizer_name})")
    plt.grid(True, linestyle="--", alpha=0.6)

    # 保存图像
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_fig_path}")

    plt.show()


if __name__ == "__main__":
    analyze_length(tokenizer_name="bert-base-uncased", split="train")