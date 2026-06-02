import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import onnxruntime
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置区域 =================
# 请确保这里的文件名和你文件夹里的一致！
IMAGE_PATH = "test1.png"       # 你的图片
SAM_ONNX_PATH = "mobile_sam_decoder.onnx"
LAMA_ONNX_PATH = "lama_fp32.onnx" 
CHECKPOINT_PATH = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

# 点击坐标 (必须是原图上的坐标！)
INPUT_POINT = np.array([[488, 674]])  #人的488 674  苹果 256 87
INPUT_LABEL = np.array([1]) 
# ===========================================

def transform_coords(coords, original_h, original_w, target_length=1024):
    """
    坐标转换工具：将原图坐标映射到 1024x1024 空间
    """
    new_coords = coords.copy().astype(float)
    scale = target_length * 1.0 / max(original_h, original_w)
    new_coords[..., 0] *= scale 
    new_coords[..., 1] *= scale 
    return new_coords

def run_lama_inpainting_safe(image_rgb, mask_binary, lama_session):
    """
    【修复版】LaMa 推理函数，增加了输出范围自动检测
    """
    print("   -> 正在准备 LaMa 输入数据...")
    
    # 1. 预处理：强制缩放到 512x512
    ori_h, ori_w = image_rgb.shape[:2]
    input_size = 512 
    img_resized = cv2.resize(image_rgb, (input_size, input_size))
    mask_resized = cv2.resize(mask_binary.astype(np.float32), (input_size, input_size))

    # 2. 归一化输入 (0-255 -> 0-1)
    img_inp = img_resized.astype(np.float32) / 255.0
    img_inp = img_inp.transpose(2, 0, 1)[None, ...]
    
    # Mask 处理：LaMa 需要 mask=1 代表遮挡(修复区域)
    mask_inp = (mask_resized > 0.5).astype(np.float32)[None, None, ...]

    # 3. ONNX 推理
    print("   -> LaMa 模型推理中...")
    lama_inputs = {
        'image': img_inp,
        'mask': mask_inp
    }
    output = lama_session.run(None, lama_inputs)[0]

    # 4. 【核心修复】自动检测输出范围
    # 如果输出的最大值大于 1.0，说明模型输出的是 0-255，不需要再乘 255
    # 如果最大值小于等于 1.0，说明输出的是 0-1，需要乘 255
    output_tensor = output[0].transpose(1, 2, 0) # CHW -> HWC
    
    max_val = np.max(output_tensor)
    print(f"   -> [Debug] LaMa 输出最大值: {max_val:.2f}")

    if max_val > 1.1:
        print("   -> 检测到输出为 0-255 范围，直接取整。")
        cur_res = np.clip(output_tensor, 0, 255).astype(np.uint8)
    else:
        print("   -> 检测到输出为 0-1 范围，乘以 255。")
        cur_res = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)

    # 5. 还原尺寸
    cur_res = cv2.resize(cur_res, (ori_w, ori_h))
    
    # 6. 融合
    final_image = image_rgb.copy()
    mask_indices = mask_binary > 0
    final_image[mask_indices] = cur_res[mask_indices]

    return final_image

def main():
    # --- 1. SAM 流程 ---
    print("1. [SAM] 初始化...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    image = cv2.imread(IMAGE_PATH)
    if image is None: 
        print(f"❌ 错误: 找不到图片 {IMAGE_PATH}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("2. [SAM] 计算特征...")
    predictor.set_image(image_rgb)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # --- 3. SAM ONNX 推理 (带坐标修正) ---
    print("3. [SAM] ONNX 推理...")
    ort_session_sam = onnxruntime.InferenceSession(SAM_ONNX_PATH)
    
    # 【关键点】获取原图尺寸
    orig_h, orig_w = image.shape[:2]
    
    # 【关键点】必须执行坐标转换！
    trans_point = transform_coords(INPUT_POINT, orig_h, orig_w)
    print(f"   -> 原始坐标: {INPUT_POINT}")
    print(f"   -> 转换后坐标: {trans_point} (基于1024尺度)")
    
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": np.concatenate([trans_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32),
        "point_labels": np.concatenate([INPUT_LABEL, np.array([-1])], axis=0)[None, :].astype(np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
    }
    
    masks, _, _ = ort_session_sam.run(None, ort_inputs)
    binary_mask = masks[0, 0, :, :] > 0.0

    # --- 4. 准备 Mask ---
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    # 稍微膨胀一点 Mask，防止边缘黑圈
    kernel = np.ones((5, 5), np.uint8) 
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2) # 膨胀2次更保险
    mask_for_lama = (mask_dilated > 0).astype(np.uint8)

    # --- 5. LaMa 修复流程 ---
    print("4. [LaMa] 开始修复...")
    try:
        # 强制使用 CPU 避免显存打架
        lama_session = onnxruntime.InferenceSession(LAMA_ONNX_PATH, providers=['CPUExecutionProvider'])
        
        # 调用我们修好的函数
        result_image = run_lama_inpainting_safe(image_rgb, mask_for_lama, lama_session)
        
        print("5. [Success] 任务完成！")

        # --- 可视化 ---
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Step 1: Original + Click")
        plt.imshow(image_rgb)
        # 画出转换前的原坐标点
        plt.scatter(INPUT_POINT[:, 0], INPUT_POINT[:, 1], color='red', marker='*', s=200)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Step 2: Generated Mask")
        plt.imshow(mask_dilated, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Step 3: LaMa Result")
        plt.imshow(result_image)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n❌ 出错了: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()