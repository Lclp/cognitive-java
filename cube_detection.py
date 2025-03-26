import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import random
import numpy as np
from PIL import Image


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

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# 定义类别
CLASSES = ["cube", "not_cube"]

# 当前训练次数
times = "05"

# 更强的数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        RemoveBackgroundAndFillGray(background_color=(255, 255, 255), tolerance=30, fill_color=128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 添加垂直翻转
        transforms.RandomRotation(30, fill=128),  # 增加旋转角度范围
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, fill=128),  # 添加仿射变换
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加颜色抖动范围
        transforms.RandomGrayscale(p=0.1),  # 有时转为灰度图
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 添加高斯模糊
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=128),  # 添加透视变换
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机擦除部分区域
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        RemoveBackgroundAndFillGray(background_color=(255, 255, 255), tolerance=30, fill_color=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 数据加载
def load_data(data_dir="dataset", val_dir="dataset_val", batch_size=16):
    # 打印文件夹结构
    print("数据集文件夹结构:")
    for folder in [data_dir, val_dir]:
        if os.path.exists(folder):
            print(f"\n{folder}:")
            for class_folder in os.listdir(folder):
                class_path = os.path.join(folder, class_folder)
                if os.path.isdir(class_path):
                    num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                    print(f"  - {class_folder}: {num_files} 图片")

    # 加载数据集
    image_datasets = {
        'train': datasets.ImageFolder(data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val']) if os.path.exists(val_dir) else None
    }

    # 打印类别映射
    print("\n类别映射:")
    print(f"训练集: {image_datasets['train'].class_to_idx}")
    if image_datasets['val'] is not None:
        print(f"验证集: {image_datasets['val'].class_to_idx}")

    # 确保类别映射一致
    if image_datasets['val'] is not None and image_datasets['train'].class_to_idx != image_datasets['val'].class_to_idx:
        print("警告: 训练集和验证集的类别映射不一致!")
        # 修正验证集的类别映射
        image_datasets['val'].class_to_idx = image_datasets['train'].class_to_idx
        image_datasets['val'].classes = image_datasets['train'].classes

    # 打印每个类别的样本数量
    print("\n类别分布:")
    for phase in ['train', 'val']:
        if image_datasets[phase] is not None:
            targets = image_datasets[phase].targets
            class_counts = {cls: targets.count(idx) for cls, idx in image_datasets[phase].class_to_idx.items()}
            print(f"{phase} 集: {class_counts}")

    # 创建数据加载器
    dataloaders = {}
    for phase in ['train', 'val']:
        if image_datasets[phase] is not None:
            if phase == 'train':
                # 为训练集创建加权采样器
                targets = image_datasets[phase].targets
                class_sample_count = np.array([targets.count(i) for i in range(len(image_datasets[phase].classes))])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in targets])
                samples_weight = torch.from_numpy(samples_weight).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                dataloaders[phase] = DataLoader(image_datasets[phase], batch_size=batch_size,
                                                sampler=sampler, num_workers=4)
            else:
                dataloaders[phase] = DataLoader(image_datasets[phase], batch_size=batch_size,
                                                shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val'] if image_datasets[x] is not None}
    class_names = image_datasets['train'].classes

    try:
        visualize_augmentations(image_datasets['train'])
    except Exception as e:
        print(f"Could not visualize augmentations: {e}")

    # 确保类别名称与我们预期的一致
    if set(class_names) != set(CLASSES):
        print(f"警告: 类别名称与预期不符! 预期: {CLASSES}, 实际: {class_names}")
        # 如果需要，可以在这里调整类别名称的映射

    return dataloaders, dataset_sizes, class_names


# 使用带有类别权重的损失函数来处理可能的类别不平衡
def get_class_weights(dataloader):
    class_counts = [0] * len(CLASSES)
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1

    total = sum(class_counts)
    class_weights = [total / (len(class_counts) * count) if count > 0 else 1.0 for count in class_counts]

    # 进一步增加稀有类别的权重
    if class_counts[1] < class_counts[0] * 0.1:  # 如果not_cube样本远少于cube样本
        class_weights[1] *= 5  # 额外增加not_cube的权重

    print(f"类别权重: {class_weights}")
    return torch.FloatTensor(class_weights)


# 修改模型构建函数，使用更适合的模型架构和参数
def build_model(num_classes=2, feature_extract=False):
    # 尝试使用更小的模型以减少过拟合
    model = models.resnet18(weights='DEFAULT')

    # 设置是否只训练最后的全连接层
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 修改模型结构，添加dropout以减少过拟合
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # 添加dropout
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),  # 添加dropout
        nn.Linear(512, num_classes)
    )

    return model


# 训练模型
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 添加早停机制
    patience = 10
    early_stop_counter = 0
    best_val_loss = float('inf')

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段跟踪梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，则反向传播+优化
                    if phase == 'train':
                        loss.backward()
                        # 添加梯度裁剪以防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 记录损失和准确率
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                # 使用验证损失更新学习率调度器
                if scheduler is not None:
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(epoch_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 如果是验证阶段，我们需要保存最佳模型
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # 早停检查
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs!")
                        # 恢复最佳模型权重
                        model.load_state_dict(best_model_wts)

                        # 绘制训练过程 - 使用实际完成的epoch数
                        plot_training(train_losses, val_losses, train_accs, val_accs, epoch + 1)

                        time_elapsed = time.time() - since
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best val Acc: {best_acc:4f}')

                        return model

            print()

    # 绘制训练过程 - 使用实际完成的epoch数
    plot_training(train_losses, val_losses, train_accs, val_accs, len(train_losses))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 绘制训练过程
# 绘制训练过程
def plot_training(train_losses, val_losses, train_accs, val_accs, epochs):
    # 获取实际完成的epoch数
    completed_epochs = len(train_losses)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(completed_epochs), train_losses, label='Training Loss')
    plt.plot(range(completed_epochs), val_losses[:completed_epochs], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(completed_epochs), train_accs, label='Training Accuracy')
    plt.plot(range(completed_epochs), val_accs[:completed_epochs], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(f"training_history_{times}.png")
    plt.close()


# 混淆矩阵可视化
def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"confusion_matrix_{times}.png")
    plt.close()

    # 确保类别名称与索引正确对应
    class_to_idx = dataloader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    ordered_class_names = [idx_to_class[i] for i in range(len(class_names))]

    # 计算每个类别的精确率和召回率
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    report = classification_report(all_labels, all_preds, target_names=ordered_class_names, zero_division=0)
    print("Classification Report:")
    print(report)

    # 特别关注"cube"类别的精确率和召回率
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    cube_idx = class_to_idx['cube']

    print("\n针对手绘正方体检测的关键指标:")
    print(f"Cube 类别的精确率: {precision[cube_idx]:.4f}")
    print(f"Cube 类别的召回率: {recall[cube_idx]:.4f}")
    print(f"Cube 类别的F1分数: {f1[cube_idx]:.4f}")

    # 评估不同阈值的性能
    evaluate_with_threshold(all_probs, all_labels, class_to_idx)


# 评估不同阈值的性能
def evaluate_with_threshold(all_probs, all_labels, class_to_idx):
    cube_idx = class_to_idx['cube']
    cube_probs = [prob[cube_idx] for prob in all_probs]

    # 使用不同阈值计算指标
    results = []
    thresholds = np.arange(0.1, 1.0, 0.1)

    for thresh in thresholds:
        # 根据阈值预测类别（大于阈值为"cube"，否则为"not_cube"）
        preds = [cube_idx if p >= thresh else 1 - cube_idx for p in cube_probs]

        # 计算指标
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average=None, zero_division=0)
        accuracy = accuracy_score(all_labels, preds)

        results.append({
            'threshold': thresh,
            'precision_cube': precision[cube_idx],
            'recall_cube': recall[cube_idx],
            'f1_cube': f1[cube_idx],
            'accuracy': accuracy
        })

    # 打印结果
    print("\n不同决策阈值的性能:")
    for result in results:
        print(f"阈值: {result['threshold']:.1f}, 精确率: {result['precision_cube']:.4f}, "
              f"召回率: {result['recall_cube']:.4f}, F1: {result['f1_cube']:.4f}, "
              f"准确率: {result['accuracy']:.4f}")

    # 绘制PR曲线
    plt.figure(figsize=(10, 6))
    plt.plot([r['recall_cube'] for r in results], [r['precision_cube'] for r in results], 'o-')
    for i, r in enumerate(results):
        plt.annotate(f"{r['threshold']:.1f}",
                     (r['recall_cube'], r['precision_cube']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Cube Detection')
    plt.grid(True)
    plt.savefig(f"precision_recall_curve_{times}.png")
    plt.close()


# 测试单个图像
def test_image(model, image_path, device):
    # 加载并预处理图像
    img = Image.open(image_path).convert('RGB')
    transform = data_transforms['val']
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    # 显示结果
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Prediction: {CLASSES[preds.item()]} ({probs[0][preds.item()]:.2%})')
    plt.axis('off')
    plt.show()

    # 返回预测结果和概率
    return CLASSES[preds.item()], probs[0][preds.item()].item()


def visualize_augmentations(dataset, num_samples=5):
    """Visualize augmentations applied to images"""
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        # Get a random image
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]

        # Original image (denormalize)
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Class: {dataset.classes[label]}")

        # Show 4 more augmented versions
        for j in range(1, 5):
            aug_img, _ = dataset[idx]
            aug_np = aug_img.numpy().transpose(1, 2, 0)
            aug_np = aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_np = np.clip(aug_np, 0, 1)
            axes[i, j].imshow(aug_np)
            axes[i, j].set_title(f"Augmentation {j}")

    plt.tight_layout()
    plt.savefig(f"augmentation_examples_{times}.png")
    print(f"Saved augmentation examples to augmentation_examples_{times}.png")

# 主函数
def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    dataloaders, dataset_sizes, class_names = load_data(batch_size=16)

    # 打印实际使用的类别名称
    print(f"\n实际使用的类别名称: {class_names}")
    print(f"类别索引映射: {dataloaders['train'].dataset.class_to_idx}")

    # 确保CLASSES全局变量与实际类别名称一致
    global CLASSES
    CLASSES = class_names

    # 计算类别权重以处理不平衡
    class_weights = get_class_weights(dataloaders['train'])

    # 构建模型
    model = build_model(num_classes=len(class_names), feature_extract=False)
    model = model.to(device)

    # 使用带权重的交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 优化器设置
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, device=device)

    # 保存模型
    torch.save(model.state_dict(), f"models/resnet18_cube_detection_{times}.pth")

    # 评估模型
    plot_confusion_matrix(model, dataloaders['val'], class_names, device)

    # 可选：测试单个图像
    try:
        test_image(model, '01.png', device)
        test_image(model, '04.png', device)
        test_image(model, '05.png', device)
        test_image(model, '08.png', device)
        test_image(model, '112.png', device)
        test_image(model, '113.png', device)
        test_image(model, '114.png', device)
        test_image(model, '115.png', device)
        test_image(model, './pics/yes.jpg', device)
    except Exception as e:
        print(f"测试单个图像时出错: {e}")


if __name__ == "__main__":
    main()
