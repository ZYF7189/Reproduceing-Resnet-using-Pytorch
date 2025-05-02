# resnet 模型实现
import torch
import torch.nn as nn 

class BaisicBlock(nn.Module):
    """
    ResNet的基本残差块，用于ResNet18和ResNet34，跳了两个卷积层
    """
    expansion = 1  # 输出通道数与输入通道数的倍数关系

    def __init__(self, in_channel, out_channel, stride=1, downsample = None):
        """"
        传入参数说明：
        in_channel: 输入通道数
        out_channel: 输出通道数
        stride: 卷积步长
        downsample: 下采样层，对应虚线残差结构（输入输出尺寸不同时使用）
        """

        super(BaisicBlock, self).__init__()
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, 3,1,1, bias = False),
            nn.BatchNorm2d(out_channel)
            
        )
        self.downsample = downsample
        

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.basicblock(x) + identity
        out = nn.ReLU(inplace = True)(out)
        return out
    
class BottenBlock(nn.Module):
    """
    ResNet的瓶颈残差块，用于ResNet50、ResNet101和ResNet152，跳了三个卷积层
    """
    expansion = 4 # # 输出通道数与输入通道数的倍数关系
    
    def __init__(self, in_channel, out_channel, stride=1, downsample = None):
        """
        参数说明：
        in_channel: 输入通道数
        out_channel: 中间层输出通道数(最终输出是out_channel*expansion)
        stride: 卷积步长
        downsample: 下采样层，对应虚线残差结构（输入输出尺寸不同时使用）
        """
        super(BottenBlock, self).__init__()
        self.bottenneck = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel,3, stride, 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel*self.expansion, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channel*self.expansion)
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.bottenneck(x) + identity
        out = nn.ReLU(inplace = True)(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top = True):
        """
        ResNet模型
        参数说明：
        block: 残差块类型（BaisicBlock或BottenBlock）
        block_num: 每个阶段的残差块数量
        num_classes: 分类数
        include_top: 是否包含全连接层
        """
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.include_top = include_top

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, 7, 2, 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # 四个残差层
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        # 分类器
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        创建残差层
        参数说明：
        block: 残差块类型（BaisicBlock或BottenBlock）
        channel: 基础通道数
        block_num: 块的数量
        stride: 第一个块的步长
        """
        downsample = None
        # 需要下采样的情况：步长不为1，或者输入通道数和输出不匹配
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, 1, stride, 0, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []

        # 第一个残差快，可能包含下采样
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion

        # 添加剩余的残差块
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        参数说明：
        x: 输入数据
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
    
def resnet18(num_classes=1000, include_top=True):
    """
    创建ResNet18模型
    参数说明：
    num_classes: 分类数
    include_top: 是否包含全连接层
    """
    return ResNet(BaisicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    """
    创建ResNet34模型
    参数说明：
    num_classes: 分类数
    include_top: 是否包含全连接层
    """
    return ResNet(BaisicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    """
    创建ResNet50模型
    参数说明：
    num_classes: 分类数
    include_top: 是否包含全连接层
    """
    return ResNet(BottenBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    """
    创建ResNet101模型
    参数说明：
    num_classes: 分类数
    include_top: 是否包含全连接层
    """
    return ResNet(BottenBlock, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    """
    创建ResNet152模型
    参数说明：
    num_classes: 分类数
    include_top: 是否包含全连接层
    """
    return ResNet(BottenBlock, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)




