import time
import traceback
import cv2  # OpenCV库，用于图像处理（缩放、读取、颜色转换等）
import numpy as np  # NumPy库，用于矩阵运算和数据处理
import onnxruntime as ort  # ONNX Runtime，用于加载和运行 .onnx 模型
import gradio as gr  # Gradio库，用于构建 Web 用户界面

# ================= 1. 配置与全局变量 =================

# 模型文件路径
# 注意：这三个 .onnx 文件必须放在与本脚本相同的目录下
ENCODER_PATH = "mobile_sam_encoder.onnx"  # MobileSAM 的编码器，负责提取图像特征
DECODER_PATH = "mobile_sam_decoder.onnx"  # MobileSAM 的解码器，负责根据提示点生成 Mask
LAMA_PATH = "lama_fp32.onnx"  # LaMa 模型，负责根据原图和 Mask 进行图像修复（消除）

# 定义推理设备的优先级
# 'CUDAExecutionProvider': 尝试使用 NVIDIA 显卡加速
# 'CPUExecutionProvider': 如果没有显卡，回退到 CPU 运行
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 全局变量，用于存储加载后的模型 Session
# 使用全局变量是为了避免每次处理图片都重新加载模型，节省时间
sess_encoder = None
sess_decoder = None
sess_lama = None


# ================= 2. 模型加载 (带 GPU 检测) =================
def load_models():
    """
    初始化并加载所有 ONNX 模型。
    在启动 UI 前会调用此函数。
    """
    global sess_encoder, sess_decoder, sess_lama

    # 如果模型已经加载过，直接跳过，避免重复加载
    if sess_encoder is not None:
        return

    print("🚀 正在初始化 GPU 加速引擎...")
    try:
        # 获取当前 ONNX Runtime 支持的硬件提供商（如 CUDA, CPU）
        available_providers = ort.get_available_providers()
        print(f"   -> 检测到可用设备: {available_providers}")

        # 加载三个模型，传入 providers 参数以启用 GPU
        sess_encoder = ort.InferenceSession(ENCODER_PATH, providers=PROVIDERS)
        sess_decoder = ort.InferenceSession(DECODER_PATH, providers=PROVIDERS)
        sess_lama = ort.InferenceSession(LAMA_PATH, providers=PROVIDERS)

        # 验证当前模型实际运行在什么设备上
        device = sess_encoder.get_providers()[0]
        print(f"✅ 模型加载成功！当前运行在: {device} (如果是 CUDA 则说明 GPU 生效)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # 打印详细的错误堆栈信息，方便调试
        traceback.print_exc()


# ================= 3. 核心算法改进版 =================


def preprocess_image_for_sam(image_rgb):
    """
    MobileSAM Encoder 的预处理函数。
    作用：将输入图片调整为 SAM 模型要求的格式（1024x1024，特定归一化）。
    """
    target_size = 1024
    # SAM 模型训练时使用的均值和方差，用于归一化
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)

    h, w = image_rgb.shape[:2]
    # 计算缩放比例，保持长边为 1024
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)

    # 缩放图片
    img = cv2.resize(image_rgb, (new_w, new_h))

    # 计算需要填充（Padding）的大小，使图片变成正方形 1024x1024
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    # 使用黑色填充右下角
    img = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # 归一化：(像素值 - 均值) / 方差
    img = (img - pixel_mean) / pixel_std

    # 转换维度顺序：HWC (高,宽,通道) -> CHW (通道,高,宽)，这是 PyTorch/ONNX 常用的格式
    img = img.transpose(2, 0, 1).astype(np.float32)

    # 增加 Batch 维度：CHW -> BCHW (1, 通道, 高, 宽)
    img = img[None, :, :, :]
    return img, scale


def run_lama_inpainting_optimized(image_rgb, mask_binary, lama_sess):
    """
    LaMa 修复算法核心逻辑。
    参数:
        image_rgb: 原始图片 (H, W, 3)
        mask_binary: 二值化的掩膜 (H, W)，0为背景，1为需要修复的区域
        lama_sess: 加载好的 LaMa ONNX session
    """
    ori_h, ori_w = image_rgb.shape[:2]

    # --- 1. 预处理：保持比例缩放 + Pad ---
    # LaMa 建议输入尺寸为 512 左右，且长宽必须是 8 的倍数
    target_size = 512
    scale = target_size / max(ori_h, ori_w)
    new_h, new_w = int(ori_h * scale), int(ori_w * scale)

    # 强制调整为 8 的倍数（向下取整）
    new_h = new_h - (new_h % 8)
    new_w = new_w - (new_w % 8)

    # 缩放原图和 Mask
    img_resized = cv2.resize(image_rgb, (new_w, new_h))
    mask_resized = cv2.resize(mask_binary.astype(np.float32), (new_w, new_h))

    # 计算需要填充的边缘，使其达到 target_size (512x512)
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # 图片使用反射填充（BORDER_REFLECT），这对于修补任务效果更好，避免边缘突变
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    # Mask 使用黑色填充（不修复填充区域）
    mask_padded = cv2.copyMakeBorder(
        mask_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    # --- 2. 归一化输入 ---
    # 将像素值归一化到 0-1 之间
    img_inp = img_padded.astype(np.float32) / 255.0
    # 调整维度顺序 HWC -> CHW，并增加 Batch 维度 -> BCHW
    img_inp = img_inp.transpose(2, 0, 1)[None, ...]

    # 处理 Mask：确保是 0 或 1 的浮点数，并调整维度 -> BCHW
    mask_inp = (mask_padded > 0.5).astype(np.float32)[None, None, ...]

    # --- 3. 模型推理 ---
    # 运行 LaMa 模型，输入图片和 Mask，得到修复后的结果
    output = lama_sess.run(None, {"image": img_inp, "mask": mask_inp})[0]

    # 取出结果：去除 Batch 维度，调整回 HWC (高,宽,通道)
    output_tensor = output[0].transpose(1, 2, 0)

    # --- 关键修复：处理输出数值范围 ---
    # 不同的 LaMa 模型导出版本输出范围不同。
    # 有的是直接输出 0-255，有的是输出 0-1。
    # 这里通过检测最大值来自动判断。
    if np.max(output_tensor) > 1.1:
        # 如果最大值 > 1.1，说明已经是 0-255 范围
        cur_res = np.clip(output_tensor, 0, 255).astype(np.uint8)
    else:
        # 如果是 0-1 范围，则需要乘以 255 并截断
        cur_res = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)

    # --- 4. 后处理 ---
    # 裁剪掉之前为了凑 512x512 而填充的部分
    cur_res = cur_res[:new_h, :new_w, :]

    # 将结果缩放回原始图片的大小
    cur_res = cv2.resize(cur_res, (ori_w, ori_h))

    # --- 5. 图像融合 ---
    # 创建最终结果的副本
    final_image = image_rgb.copy()
    # 仅将 Mask 区域（需要修复的地方）替换为 LaMa 生成的结果
    # 这样可以保持背景的高分辨率细节不变
    final_image[mask_binary > 0] = cur_res[mask_binary > 0]

    return final_image


# ================= 4. 交互处理 (适配 Gradio 3.x) =================


def process_smart_click(history_state, evt: gr.SelectData):
    """
    处理【智能点击】事件。
    当用户点击图片时触发。
    evt.index 包含点击坐标 (x, y)。
    """
    # 如果没有上传图片，直接返回
    if not history_state:
        return history_state, None, None, None, "❌ 无图"

    # history_state 是一个列表，[-1] 表示取最后一张图（当前状态图）
    image = history_state[-1]
    x, y = evt.index  # 获取点击坐标
    orig_h, orig_w = image.shape[:2]

    start_time = time.time()
    try:
        # 1. 运行 Encoder 提取特征
        input_tensor, scale = preprocess_image_for_sam(image)
        # 注意：这里每次点击都运行了一次 Encoder，理想情况下可以缓存 Encoder 的结果以加速
        embedding = sess_encoder.run(None, {"input_image": input_tensor})[
            0
        ]  # 获取图像特征

        # 2. 准备 Decoder 的输入点坐标和标签
        # MobileSAM 需要坐标和对应的标签（1表示前景点，-1表示背景点）
        # 这里还追加了一个 [0,0] 的填充点以满足模型输入格式
        onnx_coord = np.concatenate(
            [np.array([[x, y]]) * scale, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([np.array([1]), np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)

        # 3. 运行 Decoder 生成 Mask
        masks, _, _ = sess_decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "point_coords": onnx_coord.astype(np.float32),
                "point_labels": onnx_label,
                "mask_input": np.zeros(
                    (1, 1, 256, 256), dtype=np.float32
                ),  # 初始 Mask 输入为空
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32),
            },
        )

        # 4. 处理生成的 Mask
        mask = masks[0, 0, :, :] > 0.0  # 阈值化，转为布尔值
        mask_uint8 = (mask * 255).astype(np.uint8)  # 转为 0-255 图片格式

        # 5. Mask 膨胀 (Dilation)
        # 将 Mask 稍微变大一点，确保覆盖物体边缘，避免消除后留下轮廓
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

        # 6. 调用 LaMa 进行修复
        final_result = run_lama_inpainting_optimized(
            image, (mask_dilated > 0).astype(np.uint8), sess_lama
        )

        # 7. 更新历史记录
        history_state.append(final_result)
        cost = time.time() - start_time
        return (
            history_state,
            final_result,
            final_result,
            mask_dilated,
            f"✅ GPU消除完成 | 耗时: {cost:.3f}s",
        )

    except Exception as e:
        traceback.print_exc()
        return history_state, image, image, None, f"❌ 错误: {e}"


def process_manual_brush(history_state, input_dict):
    """
    处理【手动涂抹】事件。
    适配 Gradio 3.x 的输入格式。
    input_dict 包含了原图 'image' 和 蒙版 'mask'。
    """
    if not history_state or input_dict is None:
        return history_state, None, None, "❌ 请先涂抹"

    current_image = history_state[-1]

    try:
        # Gradio 3.x 中，tool="sketch" 返回的是字典 {'image': ..., 'mask': ...}
        # mask 通常是黑底白画笔（画笔区域为白色）
        mask_drawn = input_dict["mask"]  # 获取用户涂抹的 Mask 部分

        if mask_drawn is None:
            return history_state, current_image, current_image, "⚠️ 未检测到 Mask"

        # 处理 Mask 通道：如果是 RGB 格式，只取一个通道
        if len(mask_drawn.shape) == 3:
            mask_single = mask_drawn[:, :, 0]
        else:
            mask_single = mask_drawn

        # 二值化处理：确保 Mask 只有 0 和 255
        _, mask_uint8 = cv2.threshold(mask_single, 10, 255, cv2.THRESH_BINARY)

        # 检查是否真的画了东西
        if np.max(mask_uint8) == 0:
            return history_state, current_image, current_image, "⚠️ 未检测到涂抹区域"

        # 稍微膨胀，保证覆盖笔触边缘锯齿
        mask_dilated = cv2.dilate(mask_uint8, np.ones((5, 5), np.uint8), iterations=2)

        # 调用 LaMa 修复
        final_result = run_lama_inpainting_optimized(
            current_image, (mask_dilated > 0).astype(np.uint8), sess_lama
        )

        history_state.append(final_result)
        return history_state, final_result, final_result, "✅ 手动修复完成"
    except Exception as e:
        traceback.print_exc()
        return history_state, current_image, current_image, f"❌ 手动修复出错: {e}"


def undo_action(history):
    """撤销操作：弹出历史记录的最后一张"""
    msg = ""
    if len(history) > 1:
        history.pop()
        msg = "↩️ 已撤销"
    else:
        msg = "⚠️ 已是原图"

    img = history[-1] if history else None
    # 需要同时更新 智能模式图、手动模式图
    return history, img, img, msg


def reset_action(history):
    """重置操作：回到历史记录的第一张（原图）"""
    if not history:
        return [], None, None, "无图"
    orig = history[0]
    # 清空历史，只保留原图
    return [orig], orig, orig, "🔄 已重置"


def init_state(img):
    """上传图片时的初始化：将图片放入历史列表"""
    return [img], img, img


# ================= 5. UI 构建 (Pro版 - Gradio 3.x 适配) =================
def main_ui():
    load_models()  # 启动前先加载模型

    # 自定义 CSS 样式
    css = """
    .header {text-align: center; margin-bottom: 20px;}
    .header h1 {font-size: 3rem; color: #6366f1;}
    .container {max-width: 1400px; margin: auto; padding: 20px;}
    .img-container {min-height: 600px; border-radius: 12px; border: 2px solid #e5e7eb;}
    """

    # 创建 Gradio Blocks 应用
    with gr.Blocks(
        title="AI Magic Eraser Pro (GPU)", css=css, theme=gr.themes.Soft()
    ) as app:
        # 状态变量：用于存储图片的历史记录列表，方便撤销
        state = gr.State([])

        with gr.Column(elem_classes="container"):
            # 标题部分
            gr.HTML(
                "<div class='header'><h1>✨ AI 魔法消除 Pro (GPU加速版)</h1><p>基于 ONNX Runtime-GPU | 智能补全背景</p></div>"
            )

            with gr.Row():
                # --- 左侧控制栏 ---
                with gr.Column(scale=1):
                    upload_img = gr.Image(
                        label="📁 步骤1: 上传图片", type="numpy", height=250
                    )
                    gr.Markdown("---")
                    undo_btn = gr.Button("↩️ 撤销一步", variant="secondary")
                    reset_btn = gr.Button("🔄 重置原图", variant="stop")
                    log_box = gr.Textbox(label="运行日志", lines=3)

                    gr.Markdown("""
                    ### 🎮 操作指南
                    1. **智能点击**：点一下物体，AI 自动识别并消除。
                    2. **手动涂抹**：像 PS 一样涂哪里消哪里。
                    3. **显卡状态**：查看控制台输出确认是否调用了 CUDA。
                    """)

                # --- 右侧工作区 ---
                with gr.Column(scale=4):
                    with gr.Tabs():
                        # Tab 1: 智能点击功能
                        with gr.TabItem("👆 智能点击消除"):
                            img_smart = gr.Image(
                                label="点击画面中的物体",
                                type="numpy",
                                elem_classes="img-container",
                                interactive=True,
                            )

                        # Tab 2: 手动涂抹功能
                        with gr.TabItem("🎨 手动涂抹修补"):
                            img_manual = gr.Image(
                                label="使用画笔涂抹",
                                type="numpy",
                                tool="sketch",  # 重要：设置为 sketch 模式允许用户画图
                                brush_radius=25,
                                elem_classes="img-container",
                                interactive=True,
                            )
                            manual_btn = gr.Button(
                                "✨ 开始修复涂抹区域", variant="primary"
                            )

                        # Tab 3: 调试视图
                        with gr.TabItem("👀 Mask 预览"):
                            mask_view = gr.Image(
                                label="算法识别到的区域", type="numpy", height=400
                            )

        # --- 事件绑定逻辑 ---

        # 1. 图片上传后，初始化状态（左侧上传框 或 右侧智能模式直接上传 都可以）
        upload_img.upload(init_state, upload_img, [state, img_smart, img_manual])
        img_smart.upload(init_state, img_smart, [state, img_smart, img_manual])

        # 2. 智能点击：用户点击图片 -> 计算 Mask -> 消除 -> 更新所有图片框
        img_smart.select(
            process_smart_click,
            [state],
            [state, img_smart, img_manual, mask_view, log_box],
        )

        # 3. 手动模式：用户点击按钮 -> 读取画笔区域 -> 消除 -> 更新所有图片框
        manual_btn.click(
            process_manual_brush,
            [state, img_manual],
            [state, img_smart, img_manual, log_box],
        )

        # 4. 撤销和重置
        undo_btn.click(undo_action, [state], [state, img_smart, img_manual, log_box])
        reset_btn.click(reset_action, [state], [state, img_smart, img_manual, log_box])

    # 启动应用，允许浏览器内访问
    app.queue().launch(inbrowser=True)


if __name__ == "__main__":
    main_ui()
