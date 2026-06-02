import time
import numpy as np
import torch
import onnxruntime
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置 =================
CHECKPOINT_PATH = "mobile_sam.pt"
ONNX_PATH = "mobile_sam_decoder.onnx"
MODEL_TYPE = "vit_t"
DEVICE = "cuda"  # 为了公平对比，我们先在 CPU 上测（ONNX 在 CPU 上优势最明显）
# 如果你想测 GPU，把这里改为 "cuda" 并且确保 ONNX Runtime 安装了 gpu 版
# =======================================

def benchmark_pytorch():
    print(f"--- 正在准备 PyTorch ({DEVICE}) ---")
    # 1. 加载模型
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(DEVICE)
    mobile_sam.eval()
    
    # Decoder 需要 image embedding 和 prompts
    # 模拟 Image Encoder 的输出 (1, 256, 64, 64)
    dummy_embedding = torch.randn(1, 256, 64, 64).to(DEVICE)
    
    # 模拟点坐标 (1, 5, 2) 和 标签 (1, 5)
    dummy_coords = torch.randint(0, 1024, (1, 5, 2)).float().to(DEVICE)
    dummy_labels = torch.randint(0, 2, (1, 5)).float().to(DEVICE)

    # 预热
    print("PyTorch 预热中...")
    with torch.no_grad():
        for _ in range(10):
            mobile_sam.prompt_encoder(
                points=(dummy_coords, dummy_labels),
                boxes=None,
                masks=None,
            )
            # 注意：这里我们很难直接单独调用 mask_decoder，因为 Python 代码耦合度高
            # 所以我们测试整个 prompt_encoder + mask_decoder 流程
            # 这也是实际使用的流程

    # 正式测速
    print("PyTorch 开始测速 (100次)...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            # 1. 编码提示点
            sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
                points=(dummy_coords, dummy_labels),
                boxes=None,
                masks=None,
            )
            # 2. 解码 Mask
            mobile_sam.mask_decoder(
                image_embeddings=dummy_embedding,
                image_pe=mobile_sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            if DEVICE == "cuda": torch.cuda.synchronize() # 确保 GPU 跑完
            
    avg_time = (time.time() - start_time) / 100
    print(f"✅ PyTorch 平均耗时: {avg_time * 1000:.2f} ms")
    return avg_time

def benchmark_onnx():
    print(f"\n--- 正在准备 ONNX Runtime ({DEVICE}) ---")
    # 1. 加载模型
    # 如果是 CPU 测速
    providers = ['CPUExecutionProvider']
    if DEVICE == "cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    ort_session = onnxruntime.InferenceSession(ONNX_PATH, providers=providers)

    # 构造输入 (NumPy 格式)
    dummy_inputs = {
        "image_embeddings": np.random.randn(1, 256, 64, 64).astype(np.float32),
        "point_coords": np.random.randint(0, 1024, (1, 5, 2)).astype(np.float32),
        "point_labels": np.random.randint(0, 4, (1, 5)).astype(np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([1024, 1024], dtype=np.float32)
    }

    # 预热
    print("ONNX 预热中...")
    for _ in range(10):
        ort_session.run(None, dummy_inputs)

    # 正式测速
    print("ONNX 开始测速 (100次)...")
    start_time = time.time()
    for _ in range(100):
        ort_session.run(None, dummy_inputs)
        
    avg_time = (time.time() - start_time) / 100
    print(f"✅ ONNX 平均耗时: {avg_time * 1000:.2f} ms")
    return avg_time

if __name__ == "__main__":
    t_pt = benchmark_pytorch()
    t_onnx = benchmark_onnx()
    
    speedup = t_pt / t_onnx
    print(f"\n🏆 结论: ONNX 比 PyTorch 快了 {speedup:.2f} 倍")