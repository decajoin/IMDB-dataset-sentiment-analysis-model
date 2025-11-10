import torch
import torch.nn as nn


# ----------------------------
# BiLSTM 模型定义
# ----------------------------
class BiLSTM(nn.Module):
    """
    双向 BiLSTM 文本分类模型
    原理：通过前向和后向两个LSTM同时捕捉过去和未来的上下文信息，
    将两个方向的隐藏状态拼接形成包含完整上下文语义的文本表示
    """

    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=2,
                 num_classes=2, dropout=0.5, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 由于是双向 LSTM 所以隐藏层的层数要 * 2
        # 双向LSTM的最终隐藏状态维度 = 隐藏层大小 × 2（前向+后向）
        self.lstm_hidden_size = hidden_size * 2 if bidirectional else hidden_size

        # 嵌入层 - 将离散的词ID转为连续稠密的语义向量，相似含义的单词在向量空间中距离更近
        # vocab_size 嵌入层输入维度，embed_size 嵌入层输出维度
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 嵌入层dropout，防止过拟合，比率设为全dropout的一半
        self.embedding_dropout = nn.Dropout(dropout * 0.5)

        # LSTM层 - 核心双向序列建模组件
        # 输入: [batch_size, seq_len, embed_size]
        # 输出: [batch_size, seq_len, hidden_size * 2] (双向情况下)
        self.lstm = nn.LSTM(
            input_size=embed_size,  # 输入特征维度（词向量维度）
            hidden_size=hidden_size,  # 隐藏状态维度
            num_layers=num_layers,  # LSTM层数，深层网络可提取更复杂特征
            batch_first=True,  # 输入输出张量形状为[batch_size, seq_len, features]
            dropout=dropout,  # 层间dropout（除最后一层外）
            bidirectional=bidirectional  # 是否为双向LSTM
        )

        # 分类层
        self.classifier_dropout = nn.Dropout(dropout)
        # 全连接层，将 LSTM 的最终隐藏状态转换为每个样本属于每个类别的得分
        # 输入维度：双向为hidden_size*2，单向为hidden_size
        self.classifier = nn.Linear(self.lstm_hidden_size, num_classes)

        self._init_weights()

    # 通过合理的初始化提高模型的训练稳定性，避免梯度爆炸或消失。
    def _init_weights(self):
        # 嵌入层权重均匀初始化
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # LSTM参数初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 输入到隐藏层的权重：使用 Xavier 均匀初始化，保持前向传播方差稳定
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隐藏层到隐藏层的权重：使用正交初始化，避免梯度爆炸
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 偏置初始化为 0
                param.data.fill_(0)
                n = param.size(0)
                # 遗忘门的偏置初始化为 1（使 LSTM 初期倾向于保留信息，缓解梯度消失）
                param.data[(n // 4):(n // 2)].fill_(1)

        # 分类层参数初始化
        nn.init.xavier_uniform_(self.classifier.weight)  # 权重Xavier初始化
        nn.init.zeros_(self.classifier.bias)  # 偏置零初始化

    def forward(self, x):
        """
        前向传播过程

        Args:
            x: 输入文本序列，形状为 [batch_size, seq_len]

        Returns:
            logits: 分类得分，形状为 [batch_size, num_classes]
        """
        batch_size, seq_len = x.size()

        # 词嵌入: [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # 获取 BiLSTM 输出
        # lstm_out: [batch_size, seq_len, hidden_size * 2] - 每个时间步的完整输出
        # hidden: [num_layers * 2, batch_size, hidden_size] - 所有层的最终隐藏状态
        # cell: [num_layers * 2, batch_size, hidden_size] - 所有层的最终细胞状态
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 拼接最后一层的正向和反向隐藏状态作为文本的全局语义表示
        if self.bidirectional:
            # 双向 LSTM，拼接最后一层的正向（hidden[-2]）和反向（hidden[-1]）隐藏状态
            # hidden形状: [num_layers * 2, batch_size, hidden_size]
            # hidden[-2]: 最后一层前向LSTM的最终隐藏状态
            # hidden[-1]: 最后一层后向LSTM的最终隐藏状态
            # final_hidden: [batch_size, hidden_size * 2] - 包含完整上下文信息的文本表示
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向LSTM直接取最后一层的隐藏状态
            final_hidden = hidden[-1]

        # Dropout正则化
        final_hidden = self.classifier_dropout(final_hidden)

        # 分类层: 将双向语义表示映射到类别空间
        # final_hidden: [batch_size, lstm_hidden_size] -> logits: [batch_size, num_classes]
        logits = self.classifier(final_hidden)

        return logits