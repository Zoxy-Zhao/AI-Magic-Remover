# 学习教程 - 从零构建 AI 物体消除工具

本目录包含从基础到完整应用的渐进式开发代码，记录了整个实训的学习过程。

## 前置要求

教程代码依赖 PyTorch 和 MobileSAM 库（最终版 `app_pure_onnx.py` 不需要这些）：

```bash
pip install torch torchvision
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install onnxruntime opencv-python matplotlib gradio
```

同时需要将 `mobile_sam.pt` 模型权重放到项目根目录。

---

## Day 1: PyTorch 原生推理 MobileSAM

**文件**: `day1_run.py`

**学到什么**:
- SAM (Segment Anything Model) 的 Encoder-Decoder 架构
- `SamPredictor` 的使用流程：`set_image()` → `predict()`
- 通过点坐标 + 标签实现交互式分割
- `multimask_output=True` 输出多层级结果，取置信度最高的 Mask

**运行**:
```bash
python day1_run.py
```

---

## Day 2: 导出 ONNX 模型

**文件**: `day2_export_onnx.py`

**学到什么**:
- 为什么不能直接导出整个 SAM（Python 控制流不兼容 ONNX 静态图）
- `SamOnnxModel` Wrapper 的作用：提取纯计算图
- `torch.onnx.export()` 的关键参数：`opset_version`、`dynamic_axes`、`do_constant_folding`
- Dummy Input 的构造方法

**输出**: `mobile_sam_decoder.onnx`

---

## Day 3: ONNX Runtime 混合推理

**文件**: `day3_onnx_inference.py`

**学到什么**:
- ONNX Runtime 的 `InferenceSession` 使用方法
- **坐标空间变换** (核心难点)：原图坐标 → 1024x1024 特征空间坐标
- Data Marshalling：PyTorch Tensor → NumPy Array 的格式转换
- Padding 点 `[0, 0]` 和标签 `-1` 的作用

**架构**: PyTorch Encoder + ONNX Decoder（混合推理）

---

## Day 4: 图像修复 - 从基础到进阶

### 4a. OpenCV Inpainting 基础版

**文件**: `day4_magic_remove.py`

**学到什么**:
- `cv2.inpaint()` 的 TELEA 算法原理（快速行进法）
- Mask 膨胀 (Dilation) 的作用：扩大修复区域，消除物体轮廓残留
- SAM 分割 → 格式转换 → 修复 的完整 Pipeline

### 4b. LaMa 深度学习修复

**文件**: `day4_lama_pro.py`

**学到什么**:
- LaMa (Large Mask Inpainting) 模型的 ONNX 推理流程
- 输入预处理：512x512 缩放 + 归一化
- **输出范围自动检测**：不同导出版本输出 0-1 或 0-255，需自适应处理
- Mask 区域融合：仅替换修复区域，保持背景原始分辨率

---

## Day 5: Gradio 交互式应用

**文件**: `day5_app.py`

**学到什么**:
- Gradio Blocks 布局：`Row`、`Column`、`Tabs` 组合
- `gr.Image` 的 `select` 事件：捕获用户点击坐标
- `tool="sketch"` 画笔模式：实现手动涂抹交互
- `gr.State` 状态管理：实现操作历史的撤销/重置
- `gr.Progress` 进度条反馈

**架构**: PyTorch Encoder + ONNX Decoder + ONNX LaMa

---

## 最终版: 纯 ONNX 推理应用

**文件**: 项目根目录 `app_pure_onnx.py`

相比 Day 5 的改进：
- **去除 PyTorch 依赖**: Encoder 也使用 ONNX，部署更轻量
- **手动预处理**: 自行实现 SAM 的图像归一化和 Padding（替代 `SamPredictor`）
- **GPU 自动检测**: ONNX Runtime 自动切换 CUDA/CPU
- **LaMa 优化**: 增加反射填充 (BORDER_REFLECT)、8 倍对齐、Padding 裁剪

---

## 演进路线图

```
Day 1: PyTorch 全流程
  │
  ▼
Day 2: 导出 Decoder → ONNX
  │
  ▼
Day 3: Encoder(PyTorch) + Decoder(ONNX) 混合推理
  │
  ▼
Day 4: + 图像修复 (OpenCV → LaMa)
  │
  ▼
Day 5: + Gradio UI (交互式应用)
  │
  ▼
Final: 全 ONNX 推理 (脱离 PyTorch)
```
