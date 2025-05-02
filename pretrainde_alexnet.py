import torch
import torchvision
from torch import nn
from data_preprocess import get_data_cifar10, get_data_flowers

# 设置数据集
dataset = 'flowers'  # 'flowers' or 'cifar10'
# dataset = 'cifar10'  # 'flowers' or 'cifar10'



# 设置超参数
learning_rate = 0.001
epoch = 20
batch_size = 32  # 在获取数据函数时传入这个值


alexnet_true = torchvision.models.alexnet(pretrained = False)  # 加载预训练的AlexNet模型
model_path = "./pretrained_model/alexnet-owt-7be5be79.pth"  # 模型参数路径
alexnet_true.load_state_dict(torch.load(model_path))  # 加载模型参数
print(alexnet_true)  # 打印模型结构
# print(alexnet_true.state_dict())  # 打印模型参数字典


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用
print(f"使用设备: {device}")

# 冻结特征提取部分的参数
for param in alexnet_true.features.parameters():
    param.requires_grad = False


# # 多加一层线性层
if dataset == 'flowers':
    alexnet_true.classifier.add_module("added_linear",nn.Linear(1000,5))
else:
    alexnet_true.classifier.add_module("added_linear",nn.Linear(1000,10))

alexnet_true.to(device)  # 将模型放到GPU上


print(alexnet_true)  # 打印模型结构



# 加载数据集
if dataset == 'flowers':
    train_dataloader, test_dataloader, flower_class = get_data_flowers(batch_size)
else:
    train_dataloader, test_dataloader, classes = get_data_cifar10(batch_size)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # 将损失函数放到GPU上

# 优化器
optimizer1 = torch.optim.Adam(alexnet_true.parameters(), lr=learning_rate)
# optimizer1 = torch.optim.SGD(alexnet_true.parameters(), lr=learning_rate)
print(device)
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))

    alexnet_true.train()  # 设置模型为训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)  # 将数据放到GPU上
        targets = targets.to(device)  # 将标签放到GPU上

        # 前向传播
        outputs = alexnet_true(imgs)
        loss = loss_fn(outputs, targets)

        # 反向传播和优化
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

    print("第{}轮训练结束".format(i+1))

    # 测试模型
    alexnet_true.eval()  # 设置模型为评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 将数据放到GPU上
            targets = targets.to(device)  # 将标签放到GPU上

            outputs = alexnet_true(imgs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print("第{}轮测试结束".format(i+1))
    print("测试集损失：", test_loss / len(test_dataloader))
    print("测试集准确率：", correct / total)
    print("=====================================")

# 保存模型
if dataset == 'flowers':
    torch.save(alexnet_true.state_dict(), "./model/alexnet_flowers_with_pretrained.pth")  # 保存模型参数
else:
    torch.save(alexnet_true.state_dict(), "./model/alexnet_cifar10_with_pretrained.pth")  # 保存模型参数
print("模型保存成功！")
    



