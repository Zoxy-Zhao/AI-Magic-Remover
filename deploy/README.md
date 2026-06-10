# Hugging Face Spaces 部署指南

本目录包含将 AI Magic Remover 部署到 [Hugging Face Spaces](https://huggingface.co/spaces) 所需的全部文件。

## 部署步骤

### 1. 创建 Space

1. 登录 [Hugging Face](https://huggingface.co/)
2. 点击头像 → New Space
3. 填写信息：
   - **Space name**: `AI-Magic-Remover`
   - **SDK**: Gradio
   - **Hardware**: CPU basic (免费)
4. 点击 Create Space

### 2. 上传文件

将以下文件上传到 Space 仓库：

```
app.py                    ← 本目录的 app.py
requirements.txt          ← 本目录的 requirements.txt
mobile_sam_encoder.onnx   ← 从本地复制 (28 MB)
mobile_sam_decoder.onnx   ← 从本地复制 (16 MB)
lama_fp32.onnx            ← 从本地复制 (208 MB, 需要 Git LFS)
```

可以用 Git 上传：

```bash
# 克隆 Space 仓库
git clone https://huggingface.co/spaces/Zayn-ai/AI-Magic-Remover
cd AI-Magic-Remover

# 安装 Git LFS (lama 模型超过 100MB)
git lfs install
git lfs track "*.onnx"

# 复制文件
cp /path/to/deploy/app.py .
cp /path/to/deploy/requirements.txt .
cp /path/to/models/*.onnx .

# 提交并推送
git add .
git commit -m "Initial deployment"
git push
```

### 3. 等待构建

推送后 HF Spaces 会自动构建并启动应用，通常需要 2-5 分钟。
构建成功后你会得到一个在线链接，例如：

```
https://huggingface.co/spaces/Zayn-ai/AI-Magic-Remover
```

将这个链接添加到项目主 README.md 中即可。

## 注意事项

- 免费版 CPU 推理速度较慢（单次消除约 5-15 秒），但功能完整
- lama_fp32.onnx (208MB) 需要 Git LFS 上传
- HF Spaces 免费版有 16GB RAM 限制，纯 ONNX CPU 推理够用
- 如果需要更快速度，可以升级到 GPU 硬件（付费）
