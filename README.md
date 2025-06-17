# 🌸 PyTorch花卉分类与ResNet/AlexNet项目

## 简介 | Introduction

本项目基于PyTorch实现了经典的ResNet和AlexNet网络结构，支持在花卉数据集（flowers）和CIFAR-10数据集上进行训练与测试。项目包含数据预处理、模型训练、测试与可视化等完整流程，适合深度学习初学者和研究者学习与扩展。

This project implements classic ResNet and AlexNet architectures using PyTorch, supporting training and testing on the flowers and CIFAR-10 datasets. It covers data preprocessing, model training, evaluation, and visualization, making it suitable for deep learning beginners and researchers.

---

## 特性 | Features

- 🌼 支持ResNet18/34/50/101/152与AlexNet多种模型结构
- 📦 支持flowers和CIFAR-10两大数据集
- 🛠️ 数据增强与预处理自动化
- 📈 训练过程与结果可视化
- 🏆 支持模型保存与加载，方便迁移学习
- 📝 代码结构清晰，易于扩展

---

## 环境依赖 | Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- Pillow

安装依赖（Install requirements）:

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

## 目录结构 | Directory Structure

```
pythonProject-pytorch/
├── Resnet/
│   ├── train.py                # 训练主程序 | Training script
│   ├── test.py                 # 测试与推理 | Testing & inference
│   ├── model.py                # ResNet模型实现 | ResNet model
│   ├── model2.py               # 另一版ResNet实现 | Another ResNet version
│   ├── model_18.py             # ResNet18简化实现 | Simple ResNet18
│   ├── pretrainde_alexnet.py   # AlexNet迁移学习 | AlexNet transfer learning
│   ├── data_preprocess.py      # 数据预处理 | Data preprocessing
│   ├── flower_data/            # 花卉数据集 | Flower dataset
│   ├── pretrained_model/       # 预训练模型权重 | Pretrained weights
│   ├── model/                  # 训练好的模型 | Saved models
│   └── ...                     # 其他文件 | Others
```

---

## 快速开始 | Quick Start

### 1. 数据准备 | Data Preparation

- 下载并解压花卉数据集到 `flower_data/flower_photos` 目录。
- CIFAR-10数据集会自动下载。

### 2. 训练模型 | Train a Model

```bash
python Resnet/train.py
```

### 3. 测试模型 | Test a Model

```bash
python Resnet/test.py
```

### 4. 使用AlexNet迁移学习 | AlexNet Transfer Learning

```bash
python Resnet/pretrainde_alexnet.py
```

---

## 训练与测试说明 | Training & Testing

- 训练过程会自动保存最佳模型权重到 `./resNet18.pth` 或 `./model/` 目录。
- 测试脚本会输出预测类别、置信度，并可视化概率分布。
- 支持自定义数据增强与模型结构。

---

## 可视化 | Visualization

训练过程损失与准确率曲线会自动保存为 `training_process.png`，便于分析模型表现。

---

## 致谢 | Acknowledgement

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [花卉数据集](https://www.tensorflow.org/datasets/community_catalog/huggingface/flowers)

---

## 联系方式 | Contact

如有问题或建议，欢迎提交Issue或联系作者。

---

> **Enjoy Deep Learning! | 祝你玩得开心，学有所成！**
