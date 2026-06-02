import os
import sys
import time
import traceback
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ================= 导入依赖 =================
import gradio as gr
import numpy as np
import cv2
import onnxruntime
import torch
from mobile_sam import sam_model_registry, SamPredictor

# ================= 配置区域 =================
# 请确保这些文件在同一目录下
SAM_ONNX_PATH = "mobile_sam_decoder.onnx"
LAMA_ONNX_PATH = "lama_fp32.onnx"
CHECKPOINT_PATH = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

# ================= 全局模型变量 =================
predictor = None
ort_session_sam = None
lama_session = None

# ================= 1. 模型加载与 GPU 优化 =================

def load_models():
    """加载模型，自动检测 GPU"""
    global predictor, ort_session_sam, lama_session
    if predictor is not None: 
        return 
    
    print("🚀 系统启动中...")
    
    # --- 检测设备 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> PyTorch 设备: {device}")
    
    # --- 1. 加载 SAM (Encoder) ---
    print("   -> 正在加载 MobileSAM...")
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mobile_sam.to(device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
    
    # --- 2. 加载 SAM (Decoder ONNX) ---
    # ONNX Runtime 的 GPU 提供者列表
    ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    print(f"   -> ONNX 推理引擎: {ort_providers[0]}")
    
    ort_session_sam = onnxruntime.InferenceSession(SAM_ONNX_PATH, providers=ort_providers)
    
    # --- 3. 加载 LaMa (Inpainting ONNX) ---
    lama_session = onnxruntime.InferenceSession(LAMA_ONNX_PATH, providers=ort_providers)
    
    print("✅ 所有模型加载完毕！")

# ================= 2. 核心算法逻辑 (保持不变) =================

def transform_coords(coords, original_h, original_w, target_length=1024):
    """坐标映射：原图 -> 1024"""
    new_coords = coords.copy().astype(float)
    scale = target_length * 1.0 / max(original_h, original_w)
    new_coords[..., 0] *= scale 
    new_coords[..., 1] *= scale 
    return new_coords

def run_lama_inpainting(image_rgb, mask_binary, lama_sess):
    """LaMa 修复核心函数"""
    ori_h, ori_w = image_rgb.shape[:2]
    input_size = 512 
    
    img_resized = cv2.resize(image_rgb, (input_size, input_size))
    mask_resized = cv2.resize(mask_binary.astype(np.float32), (input_size, input_size))

    img_inp = img_resized.astype(np.float32) / 255.0
    img_inp = img_inp.transpose(2, 0, 1)[None, ...]
    mask_inp = (mask_resized > 0.5).astype(np.float32)[None, None, ...]

    lama_inputs = {'image': img_inp, 'mask': mask_inp}
    output = lama_sess.run(None, lama_inputs)[0]
    
    output_tensor = output[0].transpose(1, 2, 0)
    
    if np.max(output_tensor) > 1.1:
        cur_res = np.clip(output_tensor, 0, 255).astype(np.uint8)
    else:
        cur_res = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)

    cur_res = cv2.resize(cur_res, (ori_w, ori_h))
    
    final_image = image_rgb.copy()
    mask_indices = mask_binary > 0
    final_image[mask_indices] = cur_res[mask_indices]
    
    return final_image

# ================= 3. 交互处理函数 (保持不变) =================

def init_state(image):
    if image is None: return [], None, None
    return [image], image, image

def sync_tabs(history_state):
    if not history_state:
        return None, None
    current_img = history_state[-1]
    return current_img, current_img

def process_smart_click(history_state, evt: gr.SelectData, progress=gr.Progress()):
    if not history_state:
        return history_state, None, None, None, "❌ 请先上传图片"
    
    current_image = history_state[-1]
    x, y = evt.index
    orig_h, orig_w = current_image.shape[:2]
    
    progress(0.1, desc="🤖 SAM 正在观察...")
    start_time = time.time()
    
    try:
        predictor.set_image(current_image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()

        progress(0.4, desc="🎯 锁定目标...")
        trans_point = transform_coords(np.array([[x, y]]), orig_h, orig_w)
        
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": np.concatenate([trans_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32),
            "point_labels": np.concatenate([np.array([1]), np.array([-1])], axis=0)[None, :].astype(np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32)
        }
        masks, _, _ = ort_session_sam.run(None, ort_inputs)
        binary_mask = masks[0, 0, :, :] > 0.0
        
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        kernel = np.ones((10, 10), np.uint8) 
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2) 
        mask_for_lama = (mask_dilated > 0).astype(np.uint8)

        progress(0.7, desc="✨ 施展魔法消除...")
        final_result = run_lama_inpainting(current_image, mask_for_lama, lama_session)
        
        history_state.append(final_result)
        
        cost_time = time.time() - start_time
        log = f"✅ 点击消除完成 | 耗时: {cost_time:.3f}s"
        
        return history_state, final_result, final_result, mask_dilated, log

    except Exception as e:
        traceback.print_exc()
        return history_state, current_image, current_image, None, f"❌ 错误: {str(e)}"

def process_manual_brush(history_state, input_dict, progress=gr.Progress()):
    if not history_state or input_dict is None:
        return history_state, None, None, "❌ 无图片或未涂抹"

    current_image = history_state[-1] 
    mask_drawn = input_dict["mask"]   
    
    if np.max(mask_drawn) == 0:
        return history_state, current_image, current_image, "⚠️ 你还没有涂抹任何区域"
    
    progress(0.2, desc="🎨 获取手动涂抹区域...")
    start_time = time.time()

    try:
        if len(mask_drawn.shape) == 3:
            mask_single = mask_drawn[:, :, 0] 
        else:
            mask_single = mask_drawn
            
        _, mask_uint8 = cv2.threshold(mask_single, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=4)
        mask_for_lama = (mask_dilated > 0).astype(np.uint8)

        progress(0.6, desc="✨ 施展魔法消除...")
        final_result = run_lama_inpainting(current_image, mask_for_lama, lama_session)

        history_state.append(final_result)
        
        cost_time = time.time() - start_time
        log = f"✅ 手动消除完成 | 耗时: {cost_time:.3f}s"

        return history_state, final_result, final_result, log

    except Exception as e:
        traceback.print_exc()
        return history_state, current_image, current_image, f"❌ 错误: {str(e)}"

def undo_action(history_state):
    if not history_state:
        return [], None, None, "无图"
    
    msg = ""
    if len(history_state) > 1:
        history_state.pop()
        msg = f"↩️ 已撤销，剩余 {len(history_state)-1} 步"
    else:
        msg = "⚠️ 已是原图"
        
    current_img = history_state[-1]
    return history_state, current_img, current_img, msg

def reset_action(history_state):
    if not history_state: return [], None, None, "无图"
    orig = history_state[0]
    return [orig], orig, orig, "🔄 已重置为原图"

# ================= 4. UI 界面构建 (核心修改部分) =================

def main_ui():
    load_models() 
    
    # 定义 CSS 样式
    css = """
    .header {text-align: center; margin-bottom: 20px;}
    .header h1 {font-size: 3rem; margin-bottom: 5px; color: #4F46E5;}
    .container {max-width: 1200px; margin: auto; padding: 20px;}
    .img-container {min-height: 500px; border-radius: 12px; border: 2px solid #eee;}
    .debug-area {border: 1px dashed #ccc; padding: 10px; border-radius: 8px; background-color: #fafafa;}
    """
    
    with gr.Blocks(title="AI 魔法消除 Pro", css=css, theme=gr.themes.Soft()) as app:
        
        # 状态管理
        state_history = gr.State([]) 

        with gr.Column(elem_classes="container"):
            # 标题头
            gr.HTML("""
            <div class='header'>
                <h1>✨ AI Magic Remover Pro</h1>
                <p>智能点击 + 手动修补 | 连续消除 | GPU 加速</p>
            </div>
            """)

            with gr.Row():
                # 左侧：公共控制区
                with gr.Column(scale=1):
                    upload_img = gr.Image(label="📁 步骤1: 上传图片", type="numpy", height=200)
                    gr.Markdown("---")
                    undo_btn = gr.Button("↩️ 撤销 (Undo)", variant="secondary")
                    reset_btn = gr.Button("🔄 重置 (Reset)", variant="stop")
                    log_output = gr.Textbox(label="系统日志", lines=3)
                    
                    # 可以在这里加一些使用说明
                    gr.Markdown("""
                    ### 💡 使用小贴士
                    1. **智能模式**：直接点击物体即可消除。
                    2. **手动模式**：类似PS的污点修复画笔。
                    3. **调试模式**：在右侧第三个Tab查看算法识别区域。
                    """)
                    
                # 右侧：功能 Tab 区 (修改点：增加了第三个Tab)
                with gr.Column(scale=4):
                    with gr.Tabs():
                        # === Tab 1: 智能模式 ===
                        with gr.TabItem("👆 智能点击消除"):
                            img_smart = gr.Image(
                                label="直接点击图片中的物体", 
                                type="numpy", 
                                elem_classes="img-container",
                                interactive=True
                            )

                        # === Tab 2: 手动模式 ===
                        with gr.TabItem("🎨 手动涂抹修补"):
                            gr.Markdown("如果 AI 没修干净，可以在这里用笔刷涂抹残留区域，然后点击开始修复。")
                            img_manual = gr.Image(
                                label="涂抹要消除的区域", 
                                type="numpy", 
                                tool="sketch", 
                                elem_classes="img-container",
                                brush_radius=20
                            )
                            manual_run_btn = gr.Button("✨ 执行手动修复", variant="primary")
                        
                        # === Tab 3: 调试模式 (新加的) ===
                        with gr.TabItem("👀 调试与分析"):
                            gr.Markdown("""
                            ### 🔍 算法掩膜 (Mask) 可视化
                            这里展示的是 MobileSAM 算法识别到的需要消除的区域（白色部分）。
                            LaMa 模型会根据这个掩膜，自动填补背景。
                            """)
                            mask_preview = gr.Image(
                                label="Mask (Binary)", 
                                type="numpy",
                                height=400,
                                elem_classes="debug-area"
                            )

        # ================= 事件绑定 =================
        
        # 1. 上传图片 -> 初始化所有状态
        upload_img.upload(
            init_state, 
            inputs=[upload_img], 
            outputs=[state_history, img_smart, img_manual]
        )
        
        # 2. Tab 1: 智能点击
        img_smart.select(
            process_smart_click,
            inputs=[state_history],
            outputs=[state_history, img_smart, img_manual, mask_preview, log_output]
        )
        
        # 3. Tab 2: 手动涂抹
        manual_run_btn.click(
            process_manual_brush,
            inputs=[state_history, img_manual],
            outputs=[state_history, img_smart, img_manual, log_output]
        )
        
        # 4. 撤销
        undo_btn.click(
            undo_action,
            inputs=[state_history],
            outputs=[state_history, img_smart, img_manual, log_output]
        )
        
        # 5. 重置
        reset_btn.click(
            reset_action,
            inputs=[state_history],
            outputs=[state_history, img_smart, img_manual, log_output]
        )

    print("\n🚀 服务已启动！请访问本地链接...")
    app.queue().launch()

if __name__ == "__main__":
    main_ui()