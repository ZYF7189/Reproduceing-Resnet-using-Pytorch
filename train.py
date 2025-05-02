#train.py

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import time
from model2 import*
import torchvision.models.resnet


def main():
    # 配置参数
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 5
    SAVE_PATH = './resNet18.pth'
    
    # 检测设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 来自ImageNet标准化参数
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),  # 将最小边长缩放到256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 数据集路径
    data_root = os.getcwd()
    image_path = data_root + "/flower_data/"  # flower data set path
    
    # 加载训练数据集
    train_dataset = datasets.ImageFolder(root=image_path + "train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print(f"训练集样本数: {train_num}")

    # 保存类别索引映射
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    print(f"类别映射: {cla_dict}")
    # 将字典写入JSON文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               num_workers=0)
    
    # 加载验证数据集
    validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    print(f"验证集样本数: {val_num}")
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=BATCH_SIZE, 
                                                  shuffle=False,
                                                  num_workers=0)
    
    # 构建模型
    print(f"创建ResNet18模型, 类别数: {NUM_CLASSES}")
    net = resnet18(num_classes=NUM_CLASSES)
    
    # 以下是使用预训练权重的代码，如需使用请取消注释
    # model_weight_path = "./resnet34-pre.pth"
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")
    
    # 冻结参数(微调时使用)
    # for param in net.parameters():
    #     param.requires_grad = False
    # 更改全连接层结构
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, NUM_CLASSES)
    
    # 将模型移动到设备上
    net.to(device)
    
    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # 用于记录训练过程的数据
    train_losses = []
    val_accuracies = []
    
    # 开始训练
    print("开始训练...")
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # 训练阶段
        net.train()
        train_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播和反向传播
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            
            # 打印训练进度
            rate = (step + 1) / len(train_loader)
            progress_bar_length = 30
            a = "=" * int(rate * progress_bar_length)
            b = " " * (progress_bar_length - int(rate * progress_bar_length))
            print(f"\r[Epoch {epoch+1}/{EPOCHS}] Train: [{a}>{b}] {rate*100:.1f}% Loss: {loss.item():.4f}", end="")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        net.eval()
        acc = 0.0  # 准确样本计数
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels).sum().item()
                
        val_accurate = acc / val_num
        val_accuracies.append(val_accurate)
        
        # 保存最佳模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), SAVE_PATH)
            print(f"\n发现更好模型，已保存到 {SAVE_PATH}")
        
        # 计算本轮用时
        epoch_time = time.time() - epoch_start
        
        # 打印本轮训练结果
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] 训练损失: {avg_train_loss:.4f} 验证准确率: {val_accurate:.4f} 耗时: {epoch_time:.2f}秒")
    
    # 总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.2f}秒, 最佳验证准确率: {best_acc:.4f}")
    
    # 绘制训练过程曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.show()
    
    print(f"训练过程图表已保存至 'training_process.png'")


if __name__ == '__main__':
    main()
