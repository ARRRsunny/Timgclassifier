# test.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import math

# 必须与训练代码完全相同的模型定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, 
                 embed_dim=512, num_heads=8, num_layers=6, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, num_patches + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        # 添加输入尺寸验证
        assert x.shape[2] == x.shape[3] == 32, f"输入尺寸错误: {x.shape}"
        
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加class token维度验证
        assert self.class_token.shape == (1, 1, 512), "Class token维度错误"
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        return x
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = ViT(num_layers=6).to(device)  # 必须与训练时相同层数
    
    # 加载检查点
    checkpoint = torch.load('best_model.pth', map_location=device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("成功加载完整模型参数")
    except RuntimeError as e:
        print("="*50)
        print("参数加载警告：", str(e))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("已加载匹配参数，不匹配部分使用随机初始化")


    model.eval()  # 设置为评估模式

    # CIFAR-10 类别标签
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10标准归一化参数
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    # 加载CIFAR-10测试集
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建OpenCV窗口
    cv2.namedWindow('CIFAR-10 Classification', cv2.WINDOW_NORMAL)


    SCALE_FACTOR = 4
    FONT_SCALE = 0.7
    FONT_THICKNESS = 1
    TEXT_OFFSET = 10

    for i in range(len(test_dataset)):
        # 获取图片和真实标签
        image, true_label = test_dataset[i]
        
        # 转换图片格式用于显示
        img_np = image.numpy().transpose((1, 2, 0))  # CHW -> HWC
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img_np = (img_np * std + mean)  # 正确反归一化
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 放大图像
        img_display = cv2.resize(img_np, 
                               (32*SCALE_FACTOR, 32*SCALE_FACTOR),
                               interpolation=cv2.INTER_NEAREST)
                # 计算文本位置
        text_y_base = 20 * SCALE_FACTOR // 2
        text_scale = FONT_SCALE * SCALE_FACTOR / 2

        # 添加真实标签文本

        cv2.putText(img_display, f"True: {classes[true_label]}",
                   (TEXT_OFFSET, text_y_base),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   text_scale,
                   (0, 255, 0),
                   FONT_THICKNESS)
        # 准备模型输入
        input_tensor = image.unsqueeze(0).to(device)  # 添加batch维度
        
        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()
        
        # 添加预测结果文本
        color = (0, 255, 0) if predicted_label == true_label else (0, 0, 255)
        cv2.putText(img_display, f"Pred: {classes[predicted_label]}",
                   (TEXT_OFFSET, text_y_base + 20 * SCALE_FACTOR // 2),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   text_scale,
                   color,
                   FONT_THICKNESS)
        # 控制台输出
        print(f"Image {i+1}:")
        print(f"True class: {classes[true_label]}")
        print(f"Predicted class: {classes[predicted_label]}")
        print("-------------------")
        
        # 显示图片（放大4倍）

        cv2.imshow('CIFAR-10 Classification', img_display)
        
        # 按键控制
        key = cv2.waitKey(1000)  # 每张显示1秒
        if key == 27:  # ESC键退出
            break


        
    cv2.destroyAllWindows()