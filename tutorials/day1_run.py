import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
# 导入 MobileSAM 的核心组件
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置区域 =================
# 1. 设置图片路径
IMAGE_PATH = "OIP.webp"
# 2. 设置模型权重路径
CHECKPOINT_PATH = "mobile_sam.pt"
# 3. 设置模型类型 (vit_t 就是 Tiny ViT，移动端专用)
MODEL_TYPE = "vit_t"

# 4. 【重要】设置你的点击坐标 [x, y]
# 请修改下面这两个数字为你刚才在画图软件里看到的！
INPUT_POINT = np.array([[256, 87]]) 
INPUT_LABEL = np.array([1]) # 1表示正向点(我要这个)，0表示负向点(不要这个)
# ===========================================

def show_mask(mask, ax, random_color=False):
    """辅助函数：把分割出来的 Mask 画成蓝色半透明层"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) # 蓝色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    """辅助函数：把点击的小星星画出来"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def main():
    # --- 1. 准备设备 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    # --- 2. 加载模型 ---
    print("正在加载 MobileSAM 模型...")
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device)
    mobile_sam.eval() # 开启评估模式

    # 创建预测器 (Predictor)
    # 面试考点：Predictor 封装了 Image Encoder 和 Mask Decoder
    predictor = SamPredictor(mobile_sam)

    # --- 3. 读取图像 ---
    print(f"正在读取图片: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("错误：找不到图片，请检查文件名！")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转为 RGB 格式

    # --- 4. 图像编码 (最耗时的一步) ---
    print("正在进行 Vision Transformer 特征提取 (Encoder)...")
    predictor.set_image(image)

    # --- 5. 提示与解码 (瞬间完成) ---
    print(f"正在根据坐标 {INPUT_POINT} 生成掩码...")
    masks, scores, logits = predictor.predict(
        point_coords=INPUT_POINT,
        point_labels=INPUT_LABEL,
        multimask_output=True, # 输出多层级结果
    )

    # --- 6. 结果可视化 ---
    print(f"分割完成！最高置信度: {np.max(scores):.4f}")
    
    # 选取分数最高的那个结果
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(best_mask, plt.gca())
    show_points(INPUT_POINT, INPUT_LABEL, plt.gca())
    plt.title(f"MobileSAM Result (Score: {scores[best_idx]:.3f})", fontsize=18)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()