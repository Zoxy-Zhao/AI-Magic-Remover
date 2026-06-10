import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
import warnings

# 忽略一些无关紧要的警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
CHECKPOINT_PATH = "mobile_sam.pt"  # 改成你本地的 MobileSAM 权重路径
MODEL_TYPE = "vit_t"  # Tiny ViT 模型
OUTPUT_ONNX_PATH = "mobile_sam_decoder.onnx"
# ===========================================


def export_onnx():
    print("1. 正在加载 MobileSAM 模型...")
    # 加载 PyTorch 模型
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device="cpu")
    mobile_sam.eval()

    print("2. 正在封装模型 (Wrapper)...")
    # 【关键知识点】
    # 我们不能直接导出整个 SAM，因为它的逻辑里包含了很多 Python 的 if-else 控制流。
    # 我们使用 SamOnnxModel 这个官方提供的 Wrapper 类，
    # 它专门把 Decoder 部分提取出来，去掉了不必要的 Python 逻辑，只保留纯计算图。
    onnx_model = SamOnnxModel(mobile_sam, return_single_mask=True)

    # 定义虚拟输入 (Dummy Inputs)
    # ONNX 导出需要让模型“假装”跑一次，来记录数据流向。
    # 1. 图像特征 (Image Embeddings): 维度是 [1, 256, 64, 64]
    embed_dim = mobile_sam.prompt_encoder.embed_dim
    # image_embedding_size = mobile_sam.image_encoder.img_size // mobile_sam.image_encoder.patch_embed.patch_size
    image_embedding_size = 64  # 直接改为 64，因为 1024 / 16 = 64
    dummy_inputs = {
        "image_embeddings": torch.randn(
            1, embed_dim, image_embedding_size, image_embedding_size, dtype=torch.float
        ),
        "point_coords": torch.randint(
            low=0, high=1024, size=(1, 5, 2), dtype=torch.float
        ),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, 256, 256, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
    }

    # 把字典转成 tuple 序列，这是 torch.onnx.export 要求的格式
    _tuple_inputs = tuple(dummy_inputs.values())

    print("3. 开始导出 ONNX (这可能需要几秒钟)...")
    # 导出核心代码
    torch.onnx.export(
        onnx_model,
        _tuple_inputs,
        OUTPUT_ONNX_PATH,
        export_params=True,  # 导出权重
        opset_version=11,  # ONNX 算子版本，11 兼容性最好
        do_constant_folding=True,  # 常量折叠优化（把能预计算的先算好）
        input_names=list(dummy_inputs.keys()),  # 输入节点名称
        output_names=["masks", "iou_predictions", "low_res_masks"],  # 输出节点名称
        dynamic_axes={  # 【面试考点】动态轴：允许输入不同数量的点
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
    )

    print(f"✅ 成功！模型已保存为: {OUTPUT_ONNX_PATH}")
    print("现在你拥有了一个工业级的推理模型文件。")


if __name__ == "__main__":
    export_onnx()
