import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import onnxruntime  # 今天的核心主角
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置区域 =================
# 请确保这里的文件路径是正确的
IMAGE_PATH = "OIP.webp"  # 你的测试图片
ONNX_MODEL_PATH = "mobile_sam_decoder.onnx"
CHECKPOINT_PATH = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

# 点击坐标 (这是原始图片上的坐标)
INPUT_POINT = np.array([[256, 87]]) 
INPUT_LABEL = np.array([1]) 
# ===========================================

def show_mask(mask, ax):
    """可视化辅助函数：给 Mask 上色"""
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax):
    """可视化辅助函数：画出点击的点"""
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)

def transform_coords(coords, original_h, original_w, target_length=1024):
    """
    【新增的核心修复函数】
    将原始图像上的坐标，转换到 SAM 模型输入的 1024x1024 尺度空间。
    
    原理：
    SAM 的 Image Encoder 在看图之前，会先把图片的长边缩放到 1024 像素。
    所以，我们喂给 ONNX 的坐标，也必须按照同样的比例进行缩放，否则点的位置就对不上了。
    """
    new_coords = coords.copy().astype(float)
    
    # 1. 计算缩放比例 (scale)
    # 比如原图是 500x500，缩放后变成 1024x1024，比例就是 2.048
    scale = target_length * 1.0 / max(original_h, original_w)
    
    # 2. 对 x 和 y 坐标分别乘上这个比例
    new_coords[..., 0] *= scale 
    new_coords[..., 1] *= scale 
    
    return new_coords

def main():
    # --- 1. 准备 PyTorch Encoder (只用来提取特征) ---
    print("1. 正在初始化 Image Encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # --- 2. 读取图片并计算特征 ---
    print(f"2. 读取图片: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("错误：找不到图片，请检查路径！")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("3. 正在计算图像特征 (PyTorch)...")
    predictor.set_image(image)
    
    # 获取图像特征，转为 Numpy 格式
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    # --- 3. 准备 ONNX Runtime ---
    print(f"4. 正在加载 ONNX 模型: {ONNX_MODEL_PATH} ...")
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    # --- 4. 构造 ONNX 输入数据 (Data Marshalling) ---
    print("5. 正在构造 ONNX 输入数据 (包含坐标变换)...")

    # 获取原图尺寸
    orig_h, orig_w = image.shape[:2]

    # 【关键修复】: 调用 transform_coords 将坐标映射到 1024 空间
    # 这一步是 PyTorch 内部自动做的，但 ONNX 需要我们手动做
    trans_point = transform_coords(INPUT_POINT, orig_h, orig_w)

    # (1) 构造坐标输入
    # 我们使用转换后的 trans_point，而不是原始的 INPUT_POINT
    # 并在末尾拼一个 padding 点 [0,0] 以匹配模型要求的形状
    onnx_coord = np.concatenate([trans_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
    
    # (2) 构造标签输入
    # 对应地，标签也要加一个 -1 (表示这个 padding 点是无效的)
    onnx_label = np.concatenate([INPUT_LABEL, np.array([-1])], axis=0)[None, :].astype(np.float32)

    # (3) Mask 输入 (用于连续交互，第一次点击时全是 0)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    
    # (4) Has Mask (指示是否有上一次的 Mask)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    
    # (5) 图片原始尺寸 (用于让模型知道最后要把 Mask 放大回多大)
    orig_im_size = np.array([orig_h, orig_w], dtype=np.float32)

    # 打包成字典
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": orig_im_size
    }

    # --- 5. 执行 ONNX 推理 ---
    print("6. 执行 ONNX 推理 (CPU)...")
    # ort_session.run(输出节点名列表, 输入字典)
    masks, _, _ = ort_session.run(None, ort_inputs)
    
    # masks 的形状是 (1, 4, H, W)，第 0 个通常是分最高的
    best_mask = masks[0, 0, :, :] 

    # --- 6. 结果后处理与可视化 ---
    # ONNX 输出的是 Logits (未归一化数值)，大于 0 即为前景
    binary_mask = best_mask > 0.0

    print("7. 推理完成，正在显示结果...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(binary_mask, plt.gca())
    
    # 注意：画图时我们依然画原始坐标 INPUT_POINT，因为底图是原图
    show_points(INPUT_POINT, INPUT_LABEL, plt.gca())
    
    plt.axis('off')
    plt.title("Result from ONNX Runtime (Corrected)", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()