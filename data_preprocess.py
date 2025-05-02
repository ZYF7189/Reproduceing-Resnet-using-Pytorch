import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from shutil import copy
import random
from torchvision import datasets, transforms

devive = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用

def mkfile(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 


def get_data_cifar10(batch_size):
   # 定义训练集数据增强和转换
    transform_train = transforms.Compose([
        transforms.Resize(256),  # 先调整到较大尺寸
        transforms.RandomCrop(224),  # 随机裁剪到需要的尺寸
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),
        # CIFAR-10的均值和标准差
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 定义测试集转换
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_data = torchvision.datasets.CIFAR10(root='../cifar-10-train', train=True, download=True,
                                            transform = transform_train)
    test_data = torchvision.datasets.CIFAR10(root='../cifar-10-test', train=False, download=True,
                                            transform = transform_test)
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print('train_data_size:', train_data_size)
    print('test_data_size:', test_data_size)

    # 用torch.utils.data.DataLoader加载数据集
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # 训练样本的种类
    train_classes = train_data.classes

    for data in train_dataloader:
        imgs, targets = data
        print(imgs.shape)
        print(targets.shape)
        break
    return train_dataloader, test_dataloader, train_classes




def get_data_flowers(batch_size):

    data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 随机旋转±15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化参数
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化参数
    ])
}


    file_path = '../flower_data/flower_photos'
    flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

    mkfile('flower_data/train')
    for cla in flower_class:
        mkfile('flower_data/train/' + cla)

    mkfile('flower_data/val')
    for cla in flower_class:
        mkfile('flower_data/val/' + cla)

    split_rate = 0.1
    for cla in flower_class:
        cla_path = file_path + '/' + cla + '/'
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            image_path = cla_path + image
            if image in eval_index:
                new_path = 'flower_data/val/' + cla
            else:
                new_path = 'flower_data/train/' + cla
            copy(image_path, new_path)
        print(f"\r[{cla}] processing [{index + 1}/{num}]", end="")
    print("\nprocessing done!")

    train_dataset = datasets.ImageFolder(root='flower_data/train', transform=data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(root='flower_data/val', transform=data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, flower_class


if __name__ == '__main__':
    get_data_cifar10(64)
    # get_data_flowers(32)
    



