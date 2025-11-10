## 1. 代码文件说明

`Dataset` 文件夹

1. `dataset_download.py` IMDB 数据集的下载代码
2. `dataset_size_anlysis.py` 数据集分析代码
3. `dataset_loader.py` 加载 IMDB 数据集，并返回 DataLoader，在 TextCNN 和 BiLSTM 模型训练和测试时使用

`TextCNN` 文件夹

1. `TextCNN_model.py` TextCNN 模型定义代码
2. `TextCNN_train.py` TextCNN 模型训练代码
3. `TextCNN_explain.py` TextCNN 模型可解释性代码，使用遮盖法
4. `textcnn_imdb_best.pth` 保存的 TextCNN 最佳模型文件

`BiLSTM` 文件夹

1. `BiLSTM_model.py` BiLSTM 模型定义代码
2. `BiLSTM_train.py` BiLSTM 模型训练代码
3. `BiLSTM_explain.py` BiLSTM 模型可解释性代码，使用遮盖法
4. `bilstm_imdb_best.pth` 保存的 BiLSTM 最佳模型文件

`DistilBERT` 文件夹

1. `DistilBERT_train.py` DistilBERT 模型微调代码
2. `DistillBERT_explain.py` DistilBERT 模型可解释性代码，使用积分梯度法
3. `distilbert_imdb_best.pth` 保存的 DistilBERT 最佳模型文件

## 2. 模型环境配置

基础环境：`Python 3.9` + `Pytorch 2.8`

需要安装的包如下

``` 
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
transformers==4.56.2
datasets==4.1.1
huggingface-hub==0.35.0
tokenizers==0.22.1
accelerate==1.10.1
captum==0.8.0
numpy==1.26.4
pandas==2.3.2
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
tqdm==4.67.1
```

## 3. 模型性能与可解释化

三个模型的训练结果如下表

| 模型       | Test Accuracy | Epoch Time | 参数量 | 训练轮数 |
| ---------- | ------------- | ---------- | ------ | -------- |
| TextCNN    | 0.8674        | 66.3s      | 4.0M   | 10       |
| BiLSTM     | 0.8728        | 79.3s      | 4.5M   | 10       |
| DistilBERT | 0.9123        | 495.6s     | 66M    | 3        |

1. TextCNN 可解释化展示

![image-20251110210553404](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210553404.png)

2. BiLSTM 可解释化展示

![image-20251110210608846](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210608846.png)

![image-20251110210707482](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210707482.png)

![image-20251110210722160](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210722160.png)

3. DistilBERT 可解释展示

![image-20251110210734565](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210734565.png)

![image-20251110210749611](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20251110210749611.png)