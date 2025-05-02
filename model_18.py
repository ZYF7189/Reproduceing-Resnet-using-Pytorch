import torch
import torch.nn as nn

# 基本残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层，可能改变尺寸和通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层，保持尺寸和通道数不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不匹配，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential(
            # 1x1卷积层，调整输入通道数和步长
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # 保存输入用于跳跃连接
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        if self.stride != 1 or self.in_channels != self.out_channels:
            # 如果步长不为1或输入输出通道数不匹配，则使用1x1卷积调整
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        
        return out

# ResNet18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第一层：2个残差块，通道数64
        self.layer1_block1 = BasicBlock(64, 64)
        self.layer1_block2 = BasicBlock(64, 64)
        
        # 第二层：2个残差块，通道数128，第一个块步长为2，实现输出特征的尺寸减半
        self.layer2_block1 = BasicBlock(64, 128, stride=2)
        self.layer2_block2 = BasicBlock(128, 128)
        
        # 第三层：2个残差块，通道数256，第一个块步长为2
        self.layer3_block1 = BasicBlock(128, 256, stride=2)
        self.layer3_block2 = BasicBlock(256, 256)
        
        # 第四层：2个残差块，通道数512，第一个块步长为2
        self.layer4_block1 = BasicBlock(256, 512, stride=2)
        self.layer4_block2 = BasicBlock(512, 512)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 第一层
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        
        # 第二层
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        
        # 第三层
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        
        # 第四层
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        
        # 分类器
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 创建ResNet18模型的函数
def resnet18(num_classes=1000):
    return ResNet18(num_classes=num_classes)

# 模型测试
if __name__ == "__main__":
    # 创建模型实例
    model = resnet18()
    
    # 打印模型结构
    print(model)
    
    # 测试前向传播
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 如果安装了torchinfo库，可以使用它显示更详细的信息
    try:
        from torchinfo import summary
        summary(model, (1, 3, 224, 224))
    except ImportError:
        print("要查看更详细的模型信息，请安装torchinfo: pip install torchinfo")