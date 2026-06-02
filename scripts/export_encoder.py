import torch
from mobile_sam import sam_model_registry

# 1. 加载 PyTorch 模型
checkpoint = "mobile_sam.pt"
model_type = "vit_t"
mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
mobile_sam.eval()

# 2. 准备假输入 (MobileSAM 输入固定为 1024x1024)
# 格式: (Batch, Channel, Height, Width)
dummy_input = torch.randn(1, 3, 1024, 1024)

# 3. 导出为 ONNX
output_path = "mobile_sam_encoder.onnx"
print(f"正在导出 {output_path} ...")

torch.onnx.export(
    mobile_sam.image_encoder,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_image'],
    output_names=['image_embeddings'],
    dynamic_axes=None  # Encoder 输入尺寸固定，不需要动态轴
)

print("✅ 导出成功！现在你有了 mobile_sam_encoder.onnx")