# ResNet 模型实现

import torch.nn as nn
import torch 

class BasicBlock(nn.Module):
    """
    ResNet的基本残差块，用于ResNet18和ResNet34
    """
    expansion = 1  # 输出通道数与输入通道数的倍数关系

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        初始化基本残差块
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 卷积步长
            downsample: 下采样层，对应虚线残差结构（输入输出尺寸不同时使用）
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel,
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入作为捷径分支
        
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 捷径分支，如果需要下采样则处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet的瓶颈残差块，用于ResNet50、ResNet101和ResNet152
    """
    expansion = 4  # 最终输出通道数是输入通道数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        初始化瓶颈残差块
        Args:
            in_channel: 输入通道数
            out_channel: 中间层输出通道数(最终输出是out_channel*expansion)
            stride: 卷积步长
            downsample: 下采样层，对应虚线残差结构
        """
        super(Bottleneck, self).__init__()
        # 1x1卷积，用于降低通道数
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel,
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        # 3x3卷积，用于特征提取
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel,
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # 1x1卷积，用于提高通道数
        self.conv3 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel*self.expansion,
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 保存输入作为捷径分支
        
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 捷径分支，如果需要下采样则处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet模型
    """
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        """
        初始化ResNet
        Args:
            block: 残差块类型，BasicBlock或Bottleneck
            blocks_num: 每层残差块的数量列表
            num_classes: 分类类别数
            include_top: 是否包含顶层分类器，为False则可用于特征提取
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        # 分类器
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        创建残差层
        Args:
            block: 残差块类型
            channel: 基础通道数
            block_num: 块的数量
            stride: 第一个块的步长
        Returns:
            nn.Sequential: 残差层
        """
        downsample = None
        # 需要下采样的情况：1.步长不为1，2.输入通道数与输出不匹配
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel, 
                    channel * block.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        # 添加后续残差块
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            输出张量
        """
        # 初始处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类器
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet18(num_classes=1000, include_top=True):
    """
    创建ResNet18模型
    Args:
        num_classes: 分类类别数
        include_top: 是否包含顶层分类器
    Returns:
        ResNet18模型
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    """
    创建ResNet34模型
    Args:
        num_classes: 分类类别数
        include_top: 是否包含顶层分类器
    Returns:
        ResNet34模型
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)



def resnet101(num_classes=1000, include_top=True):
    """
    创建ResNet101模型
    Args:
        num_classes: 分类类别数
        include_top: 是否包含顶层分类器
    Returns:
        ResNet101模型
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    """
    创建ResNet50模型
    Args:
        num_classes: 分类类别数
        include_top: 是否包含顶层分类器
    Returns:
        ResNet50模型
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    """
    创建ResNet152模型
    Args:
        num_classes: 分类类别数
        include_top: 是否包含顶层分类器
    Returns:
        ResNet152模型
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)