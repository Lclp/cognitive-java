import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

# 定义类别
CLASSES = ["cube", "not_cube"]

# 图像预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 构建与训练时相同的模型架构
def build_model(num_classes=2):
    model = models.resnet18(weights=None)
    
    # 修改模型结构，添加dropout以减少过拟合
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    return model

# 加载模型
def load_model(model_path, device):
    model = build_model(num_classes=len(CLASSES))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

# 测试单个图像
def test_image(model, image_path, device):
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在 - {image_path}")
        return None, None
    
    try:
        # 加载并预处理图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        
        # 获取预测结果
        predicted_class = CLASSES[preds.item()]
        confidence = probs[0][preds.item()].item()
        
        # 显示结果
        print(f"\n图像: {image_path}")
        print(f"预测结果: {predicted_class}")
        print(f"置信度: {confidence:.2%}")
        
        # 显示所有类别的概率
        print("各类别概率:")
        for i, cls in enumerate(CLASSES):
            print(f"  {cls}: {probs[0][i].item():.2%}")
        
        # 可视化结果
        try:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f'预测: {predicted_class} ({confidence:.2%})')
            plt.axis('off')
            
            # 保存可视化结果
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            plt.savefig(f"{output_dir}/pred_{filename}")
            plt.close()
            print(f"预测可视化已保存至: {output_dir}/pred_{filename}")
        except Exception as e:
            print(f"保存可视化结果时出错: {e}")
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

# 查找可用的模型
def find_models(models_dir="./models"):
    if not os.path.exists(models_dir):
        print(f"错误: 模型目录不存在 - {models_dir}")
        return []
    
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    return model_files

class RemoveBackgroundAndFillGray(object):
    """
    去除图像背景并填充灰色

    参数:
        background_color (tuple): 背景色的RGB值，例如白色为(255, 255, 255)
        tolerance (int): 颜色匹配的容差值，值越大，匹配的颜色范围越广
        fill_color (int): 填充颜色，默认为128（中灰色）
    """

    def __init__(self, background_color=(255, 255, 255), tolerance=30, fill_color=128):
        self.background_color = np.array(background_color)
        self.tolerance = tolerance
        self.fill_color = fill_color

    def __call__(self, img):
        # 将PIL图像转换为numpy数组
        img_array = np.array(img)

        # 创建背景掩码
        # 计算每个像素与背景色的欧氏距离
        diff = np.abs(img_array - self.background_color)
        dist = np.sqrt(np.sum(diff ** 2, axis=2)) if len(img_array.shape) == 3 else diff

        # 根据容差值创建掩码
        mask = dist <= self.tolerance

        # 创建一个全灰色图像
        if len(img_array.shape) == 3:  # 彩色图像
            gray_img = np.ones_like(img_array) * self.fill_color
        else:  # 灰度图像
            gray_img = np.ones_like(img_array) * self.fill_color

        # 将背景替换为灰色
        if len(img_array.shape) == 3:
            img_array[mask] = self.fill_color
        else:
            img_array[mask] = self.fill_color

        # 转回PIL图像
        return Image.fromarray(img_array.astype(np.uint8))


# 主函数
def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 查找可用的模型
    model_files = find_models()
    
    if not model_files:
        print("错误: 未找到模型文件。请先运行cube_detection.py训练模型。")
        return
    
    # 显示可用的模型
    print("\n可用的模型:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    # 选择模型
    selection = input("\n请选择要使用的模型 (输入编号): ")
    try:
        model_index = int(selection) - 1
        if model_index < 0 or model_index >= len(model_files):
            raise ValueError("无效的选择")
        selected_model = model_files[model_index]
    except:
        print("使用最新的模型...")
        selected_model = model_files[-1]
    
    # 加载模型
    model = load_model(selected_model, device)
    if model is None:
        return
    
    print("\n=== 图像验证模式 ===")
    print("请输入图像路径进行验证，输入'N'退出")
    
    # 持续验证图像
    while True:
        image_path = input("\n请输入图像路径: ").strip()
        
        # 检查是否退出
        if image_path.upper() == 'N':
            print("退出验证模式。")
            break
        
        # 如果路径为空，使用默认图像
        if not image_path:
            image_path = "./pics/yes.jpg"
            print(f"使用默认图像: {image_path}")
        
        # 测试图像
        test_image(model, image_path, device)

if __name__ == "__main__":
    main()