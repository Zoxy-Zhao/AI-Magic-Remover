import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import onnxruntime
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置区域 =================
IMAGE_PATH = "OIP.webp" # 你的图
ONNX_MODEL_PATH = "mobile_sam_decoder.onnx"
CHECKPOINT_PATH = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

INPUT_POINT = np.array([[256, 87]]) # 你的坐标
INPUT_LABEL = np.array([1]) 
# ===========================================

def transform_coords(coords, original_h, original_w, target_length=1024):
    """Day 3 写的坐标转换函数"""
    new_coords = coords.copy().astype(float)
    scale = target_length * 1.0 / max(original_h, original_w)
    new_coords[..., 0] *= scale 
    new_coords[..., 1] *= scale 
    return new_coords

def main():
    # --- 1. SAM 流程：生成 Mask (复用 Day 3 代码) ---
    print("1. [SAM] 正在初始化...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    image = cv2.imread(IMAGE_PATH)
    if image is None: return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("2. [SAM] 计算特征...")
    predictor.set_image(image_rgb)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    print("3. [SAM] ONNX 推理中...")
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    
    orig_h, orig_w = image.shape[:2]
    trans_point = transform_coords(INPUT_POINT, orig_h, orig_w)
    
    # 构造 ONNX 输入
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": np.concatenate([trans_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32),
        "point_labels": np.concatenate([INPUT_LABEL, np.array([-1])], axis=0)[None, :].astype(np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
    }
    
    masks, _, _ = ort_session.run(None, ort_inputs)
    best_mask = masks[0, 0, :, :] 
    binary_mask = best_mask > 0.0

    # --- 2. 魔法消除流程 (Day 4 核心) ---
    print("4. [Inpaint] 正在进行魔法消除...")

    # 步骤 A: 转换 Mask 格式
    # Inpainting 需要 uint8 格式的 Mask (0是背景，255是待修复区域)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # 步骤 B: 【关键】Mask 膨胀 (Dilation)
    # 这一步是为了让 Mask 稍微变大一点，覆盖住物体的边缘，防止修完有“鬼影”
    kernel = np.ones((10, 10), np.uint8) # 10x10 的核，数字越大扩得越多
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    # 步骤 C: 执行修复 (Inpainting)
    # cv2.INPAINT_TELEA 是基于快速行进算法的修复，适合小物体
    # radius=3 是采样的邻域半径
    inpainted_image = cv2.inpaint(image_rgb, mask_dilated, 3, cv2.INPAINT_TELEA)

    # --- 3. 可视化对比 ---
    print("5. 显示结果...")
    plt.figure(figsize=(15, 5))

    # 图1: 原图 + 选点
    plt.subplot(1, 3, 1)
    plt.title("Original + Click")
    plt.imshow(image_rgb)
    plt.scatter(INPUT_POINT[:, 0], INPUT_POINT[:, 1], color='red', marker='*', s=200)
    plt.axis('off')

    # 图2: 膨胀后的 Mask (告诉模型修哪里)
    plt.subplot(1, 3, 2)
    plt.title("Dilated Mask")
    plt.imshow(mask_dilated, cmap='gray')
    plt.axis('off')

    # 图3: 修复结果
    plt.subplot(1, 3, 3)
    plt.title("Magic Removed Result")
    plt.imshow(inpainted_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()