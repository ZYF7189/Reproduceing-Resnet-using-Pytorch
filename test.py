# 模型预测脚本

import torch
import numpy as np
from model import*
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import time
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

def main():
    # 配置参数
    MODEL_PATH = "./model/resNet50_50.pth"
    CLASS_JSON_PATH = './class_indices.json'
    # TEST_IMAGE_PATH = "./model_test_image/flowers/tulips/tulip1.jpg"
    # TEST_IMAGE_PATH = "./model_test_image/flowers/dandelion/dandelion1.jpg"
    TEST_IMAGE_PATH = "./model_test_image/flowers/daisy/daisy1.jpg"
    # TEST_IMAGE_PATH = "./model_test_image/flowers/roses/rose1.jpg"
    # TEST_IMAGE_PATH = "./model_test_image/flowers/sunflower/sunflower1.jpg"
    NUM_CLASSES = 5
    
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 检查并加载图像
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"错误: 测试图像不存在: {TEST_IMAGE_PATH}")
        return
        
    try:
        # 加载并显示原始图像
        img = Image.open(TEST_IMAGE_PATH)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.title("原始图像")
        plt.imshow(img)
        
        # 预处理图像
        img_processed = data_transform(img)
        img_tensor = torch.unsqueeze(img_processed, dim=0).to(device)  # [N, C, H, W]
    except Exception as e:
        print(f"图像加载或处理错误: {e}")
        return
    
    # 加载类别映射
    try:
        with open(CLASS_JSON_PATH, 'r') as json_file:
            class_indict = json.load(json_file)
    except Exception as e:
        print(f"类别映射加载错误: {e}")
        return
    
    # 创建并加载模型
    try:
        model = resnet50(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载错误: {e}")
        return
    
    # 预测
    start_time = time.time()
    
    with torch.no_grad():
        # 进行预测
        output = model(img_tensor)
        output = torch.squeeze(output)
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).item()
    
    # 计算预测用时
    inference_time = time.time() - start_time
    
    # 获取预测结果和置信度
    predicted_class_name = class_indict[str(predict_class)]
    confidence = predict[predict_class].item()
    
    # 获取所有类别的预测概率
    probs = {class_indict[str(i)]: predict[i].item() for i in range(len(predict))}
    
    # 打印预测结果
    print(f"\n预测结果: {predicted_class_name}")
    print(f"置信度: {confidence:.4f}")
    print(f"推理耗时: {inference_time*1000:.2f}ms")
    
    # 显示预测结果
    plt.subplot(1, 2, 2)
    
    # 绘制条形图显示各类别预测概率
    categories = list(probs.keys())
    values = list(probs.values())
    
    colors = ['blue'] * len(categories)
    colors[predict_class] = 'red'  # 突出显示预测的类别
    
    plt.barh(categories, values, color=colors)
    plt.title("预测概率分布")
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # 保存结果图像
    result_filename = f"./predict_results/prediction_result_{int(time.time())}.png"
    plt.savefig(result_filename)
    plt.show()
    print(f"结果图表已保存至 {result_filename}")


if __name__ == '__main__':
    main()
