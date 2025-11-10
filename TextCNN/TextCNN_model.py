import torch
import torch.nn as nn


# ----------------------------
# TextCNN模型定义
# ----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_filters=100, filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        """
        TextCNN模型初始化

        Args:
            vocab_size: 词汇表大小
            embed_size: 词向量维度
            num_filters: 每种尺寸卷积核的数量（输出通道数）
            filter_sizes: 卷积核大小列表，对应不同n-gram窗口大小
            num_classes: 分类类别数
            dropout: dropout比率
        """
        super(TextCNN, self).__init__()

        # 嵌入层：将词索引映射为稠密向量
        # 输入: [batch_size, seq_len]
        # 输出: [batch_size, seq_len, embed_size]
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 卷积层列表：使用不同尺寸的一维卷积核并行提取多尺度特征
        # 每个Conv1d包含num_filters个卷积核，每个核大小为[kernel_size]
        # 实际权重形状: [num_filters, embed_size, kernel_size]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_size,   # 输入通道数 = 词向量维度
                out_channels=num_filters, # 输出通道数 = 卷积核数量
                kernel_size=k             # 卷积核大小 = n-gram窗口大小
            )
            for k in filter_sizes
        ])

        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 全连接层：将多尺度特征映射到分类空间
        # 输入维度: len(filter_sizes) * num_filters
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        """
        前向传播过程

        Args:
            x: 输入文本序列，形状为 [batch_size, seq_len]

        Returns:
            分类结果，形状为 [batch_size, num_classes]
        """
        # 词嵌入：将离散词索引转为连续向量
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        x = self.embedding(x)

        # 维度转换：将嵌入矩阵调整为卷积层期望的输入格式
        # [batch_size, seq_len, embed_size] -> [batch_size, embed_size, seq_len]
        # 现在embed_size成为通道维度，seq_len成为序列长度维度
        x = x.transpose(1, 2)

        # 多尺度特征提取
        conv_outputs = []
        for conv in self.convs:
            # 一维卷积 + ReLU激活
            # 输入: [batch_size, embed_size, seq_len]
            # 输出: [batch_size, num_filters, new_seq_len]
            # 每个卷积核在序列长度维度滑动，提取局部n-gram模式
            conv_out = torch.relu(conv(x))

            # 全局最大池化：提取每个特征图的最显著特征
            # [batch_size, num_filters, new_seq_len] -> [batch_size, num_filters, 1]
            # kernel_size = conv_out.size(2) 表示对整个序列长度进行池化
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))

            # 压缩维度：移除长度为1的序列维度
            # [batch_size, num_filters, 1] -> [batch_size, num_filters]
            conv_outputs.append(pooled.squeeze(2))

        # 特征融合：拼接不同尺寸卷积核提取的特征
        # 将多个[batch_size, num_filters]张量在特征维度拼接
        # 结果: [batch_size, len(filter_sizes) * num_filters]
        x = torch.cat(conv_outputs, dim=1)

        # Dropout正则化
        x = self.dropout(x)

        # 分类层：将融合特征映射到类别空间
        # [batch_size, len(filter_sizes)*num_filters] -> [batch_size, num_classes]
        x = self.fc(x)

        return x