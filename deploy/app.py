"""
AI Magic Remover - Hugging Face Spaces 部署版 (Gradio 6.x)
基于 MobileSAM + LaMa 的智能物体消除工具
"""

import time
import traceback
import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr

# ================= 配置 =================
ENCODER_PATH = "mobile_sam_encoder.onnx"
DECODER_PATH = "mobile_sam_decoder.onnx"
LAMA_PATH = "lama_fp32.onnx"
PROVIDERS = ["CPUExecutionProvider"]

sess_encoder = None
sess_decoder = None
sess_lama = None


# ================= 模型加载 =================
def load_models():
    global sess_encoder, sess_decoder, sess_lama
    if sess_encoder is not None:
        return
    print("Loading models...")
    sess_encoder = ort.InferenceSession(ENCODER_PATH, providers=PROVIDERS)
    sess_decoder = ort.InferenceSession(DECODER_PATH, providers=PROVIDERS)
    sess_lama = ort.InferenceSession(LAMA_PATH, providers=PROVIDERS)
    print(f"Models loaded. Provider: {sess_encoder.get_providers()[0]}")


# ================= 核心算法 =================
def preprocess_image_for_sam(image_rgb):
    target_size = 1024
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
    h, w = image_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    img = cv2.resize(image_rgb, (new_w, new_h))
    pad_h, pad_w = target_size - new_h, target_size - new_w
    img = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    img = (img - pixel_mean) / pixel_std
    img = img.transpose(2, 0, 1).astype(np.float32)
    return img[None, :, :, :], scale


def run_lama_inpainting(image_rgb, mask_binary, lama_sess):
    ori_h, ori_w = image_rgb.shape[:2]
    target_size = 512
    scale = target_size / max(ori_h, ori_w)
    new_h, new_w = int(ori_h * scale), int(ori_w * scale)
    new_h -= new_h % 8
    new_w -= new_w % 8
    img_resized = cv2.resize(image_rgb, (new_w, new_h))
    mask_resized = cv2.resize(mask_binary.astype(np.float32), (new_w, new_h))
    pad_h, pad_w = target_size - new_h, target_size - new_w
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    mask_padded = cv2.copyMakeBorder(
        mask_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )
    img_inp = (img_padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    mask_inp = (mask_padded > 0.5).astype(np.float32)[None, None, ...]
    output = lama_sess.run(None, {"image": img_inp, "mask": mask_inp})[0]
    output_tensor = output[0].transpose(1, 2, 0)
    if np.max(output_tensor) > 1.1:
        cur_res = np.clip(output_tensor, 0, 255).astype(np.uint8)
    else:
        cur_res = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)
    cur_res = cur_res[:new_h, :new_w, :]
    cur_res = cv2.resize(cur_res, (ori_w, ori_h))
    final_image = image_rgb.copy()
    final_image[mask_binary > 0] = cur_res[mask_binary > 0]
    return final_image


def extract_mask_from_editor(editor_value):
    """从 ImageEditor 的绘制层中提取 mask"""
    if editor_value is None:
        return None, None

    background = None
    mask = None

    if isinstance(editor_value, dict):
        background = editor_value.get("background", None)
        layers = editor_value.get("layers", [])
        composite = editor_value.get("composite", None)

        if layers and len(layers) > 0:
            for layer in layers:
                if layer is None:
                    continue
                if len(layer.shape) == 3 and layer.shape[2] == 4:
                    alpha = layer[:, :, 3]
                    if np.max(alpha) > 0:
                        mask = alpha
                        break
                elif len(layer.shape) == 3 and layer.shape[2] == 3:
                    gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                    if np.max(gray) > 0:
                        mask = gray
                        break
                elif len(layer.shape) == 2:
                    if np.max(layer) > 0:
                        mask = layer
                        break
    elif isinstance(editor_value, np.ndarray):
        background = editor_value

    return background, mask


def find_mask_center(mask):
    """找到 mask 绘制区域的中心坐标"""
    binary = (mask > 10).astype(np.uint8)
    coords = np.where(binary > 0)
    if len(coords[0]) == 0:
        return None
    cy = int(np.mean(coords[0]))
    cx = int(np.mean(coords[1]))
    return (cx, cy)


# ================= 交互处理 =================
def process_smart_remove(history_state, editor_value):
    """智能消除：用户在图片上标记一个点/小区域，AI 自动识别并消除整个物体"""
    bg, mask = extract_mask_from_editor(editor_value)

    if not history_state and bg is not None:
        history_state = [bg]

    if not history_state:
        return history_state, None, None, "Please upload an image first."

    if mask is None:
        return (
            history_state,
            history_state[-1],
            None,
            "Please draw a mark on the object you want to remove.",
        )

    center = find_mask_center(mask)
    if center is None:
        return (
            history_state,
            history_state[-1],
            None,
            "Could not find marked area. Please draw again.",
        )

    image = history_state[-1]
    x, y = center
    orig_h, orig_w = image.shape[:2]
    start_time = time.time()

    try:
        input_tensor, scale = preprocess_image_for_sam(image)
        embedding = sess_encoder.run(None, {"input_image": input_tensor})[0]

        onnx_coord = np.concatenate(
            [np.array([[x, y]]) * scale, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([np.array([1]), np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)

        masks, _, _ = sess_decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "point_coords": onnx_coord.astype(np.float32),
                "point_labels": onnx_label,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32),
            },
        )

        seg_mask = masks[0, 0, :, :] > 0.0
        mask_uint8 = (seg_mask * 255).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

        final_result = run_lama_inpainting(
            image, (mask_dilated > 0).astype(np.uint8), sess_lama
        )
        history_state.append(final_result)
        cost = time.time() - start_time
        return (
            history_state,
            final_result,
            mask_dilated,
            f"Smart remove done! Point: ({x}, {y}) | Time: {cost:.2f}s",
        )
    except Exception as e:
        traceback.print_exc()
        return history_state, history_state[-1], None, f"Error: {e}"


def process_manual_remove(history_state, editor_value):
    """手动消除：用户涂抹整个区域，直接用涂抹区域作为 LaMa 修复的 mask"""
    bg, mask = extract_mask_from_editor(editor_value)

    if not history_state and bg is not None:
        history_state = [bg]

    if not history_state:
        return history_state, None, None, "Please upload an image first."

    if mask is None:
        return (
            history_state,
            history_state[-1],
            None,
            "Please paint over the area you want to remove.",
        )

    current_image = history_state[-1]
    start_time = time.time()

    try:
        _, mask_uint8 = cv2.threshold(mask.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        if np.max(mask_uint8) == 0:
            return history_state, current_image, None, "No brush area detected."

        mask_dilated = cv2.dilate(mask_uint8, np.ones((5, 5), np.uint8), iterations=2)
        final_result = run_lama_inpainting(
            current_image, (mask_dilated > 0).astype(np.uint8), sess_lama
        )
        history_state.append(final_result)
        cost = time.time() - start_time
        return (
            history_state,
            final_result,
            mask_dilated,
            f"Manual repair done! Time: {cost:.2f}s",
        )
    except Exception as e:
        traceback.print_exc()
        return history_state, current_image, None, f"Error: {e}"


def undo_action(history):
    if not history:
        return [], None, None, "No image."
    if len(history) > 1:
        history.pop()
        msg = "Undone."
    else:
        msg = "Already at original."
    img = history[-1] if history else None
    return history, img, None, msg


def reset_action(history):
    if not history:
        return [], None, None, "No image."
    orig = history[0]
    return [orig], orig, None, "Reset to original."


def on_upload(img):
    if img is None:
        return [], None, None
    return [img], img, None


# ================= UI (Gradio 6.x) =================
def main_ui():
    load_models()

    css = """
    .header {text-align: center; margin-bottom: 20px;}
    .header h1 {font-size: 2.5rem; color: #6366f1;}
    .container {max-width: 1400px; margin: auto; padding: 20px;}
    """

    with gr.Blocks(title="AI Magic Remover") as app:
        state = gr.State([])

        with gr.Column(elem_classes="container"):
            gr.HTML("""<div class='header'>
                <h1>AI Magic Remover</h1>
                <p>Remove any object from images. Powered by MobileSAM + LaMa (ONNX).</p>
            </div>""")

            with gr.Row():
                # Left panel
                with gr.Column(scale=1):
                    upload_img = gr.Image(
                        label="1. Upload Image", type="numpy", height=250
                    )
                    gr.Markdown("---")
                    undo_btn = gr.Button("Undo", variant="secondary")
                    reset_btn = gr.Button("Reset", variant="stop")
                    log_box = gr.Textbox(label="Log", lines=3)
                    gr.Markdown("""
                    ### How to Use
                    **Smart Remove**: Draw a small mark on the object,
                    then click "Smart Remove". AI will detect the full
                    object and remove it.

                    **Manual Brush**: Paint over the entire area to remove,
                    then click "Manual Remove". Good for fine touch-ups.
                    """)

                # Right panel
                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.TabItem("Smart Remove"):
                            gr.Markdown(
                                "**Draw a small mark** on the object you want to remove, then click the button below."
                            )
                            editor_smart = gr.ImageEditor(
                                label="Mark the object to remove",
                                type="numpy",
                                interactive=True,
                                brush=gr.Brush(default_size=8, colors=["#ff0000"]),
                                eraser=gr.Eraser(default_size=15),
                            )
                            smart_btn = gr.Button(
                                "Smart Remove (AI detect + remove)",
                                variant="primary",
                                size="lg",
                            )

                        with gr.TabItem("Manual Brush"):
                            gr.Markdown(
                                "**Paint over the entire area** you want to remove, then click the button below."
                            )
                            editor_manual = gr.ImageEditor(
                                label="Paint area to remove",
                                type="numpy",
                                interactive=True,
                                brush=gr.Brush(default_size=25, colors=["#ffffff"]),
                                eraser=gr.Eraser(default_size=25),
                            )
                            manual_btn = gr.Button(
                                "Manual Remove (direct inpaint)",
                                variant="primary",
                                size="lg",
                            )

                        with gr.TabItem("Mask Preview"):
                            mask_view = gr.Image(
                                label="Detected region (white = area to remove)",
                                type="numpy",
                                height=400,
                            )

                        with gr.TabItem("Result"):
                            result_view = gr.Image(
                                label="Latest result", type="numpy", height=500
                            )

        # --- Events ---
        # Upload: init state and set as background in both editors
        def handle_upload(img):
            if img is None:
                return [], None, None, None
            return [img], img, None, "Image uploaded. Draw on the object to remove."

        upload_img.upload(
            handle_upload, upload_img, [state, result_view, mask_view, log_box]
        )

        # Smart remove
        smart_btn.click(
            process_smart_remove,
            [state, editor_smart],
            [state, result_view, mask_view, log_box],
        )

        # Manual remove
        manual_btn.click(
            process_manual_remove,
            [state, editor_manual],
            [state, result_view, mask_view, log_box],
        )

        # Undo / Reset
        undo_btn.click(undo_action, [state], [state, result_view, mask_view, log_box])
        reset_btn.click(reset_action, [state], [state, result_view, mask_view, log_box])

    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        css=css,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main_ui()
