# Running DeepSeek Janus Locally on Mac

This guide will help you set up and run DeepSeek Janus models on your Mac.

## System Requirements

### Minimum Requirements
- **Mac with Apple Silicon** (M1/M2/M3) or Intel Mac with 16GB+ RAM
- **Python 3.8+**
- **Storage**: ~15GB free space for models
- **Memory**: 
  - Janus-1.5B: 8GB RAM minimum
  - Janus-Pro-7B: 32GB RAM recommended

### Recommended Setup
- Mac with M1 Pro/Max or M2/M3 with 32GB+ RAM
- macOS Ventura or later
- Python 3.10 or 3.11

## Installation Steps

### 1. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv janus_env

# Activate it
source janus_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install PyTorch for Mac

For Apple Silicon Macs (M1/M2/M3):
```bash
pip install torch torchvision torchaudio
```

For Intel Macs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# Clone and install Janus
git clone https://github.com/deepseek-ai/Janus
cd Janus
pip install -e .
cd ..
```

### 4. Verify Installation

```python
# Test if everything is working
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "from janus.models import MultiModalityCausalLM; print('Janus imported successfully')"
```

## Running the Examples

### Quick Start

```bash
# Run the simple demo (uses less memory)
python simple_janus_demo.py

# Run the full-featured demo
python run_janus_local.py
```

### Memory Management Tips

1. **Use the smaller model** if you have limited RAM:
   ```python
   runner = JanusLocalRunner(use_smaller_model=True)
   ```

2. **Monitor memory usage**:
   ```bash
   # In another terminal
   watch -n 1 "ps aux | grep python | grep -v grep"
   ```

3. **Clear cache between operations**:
   ```python
   import torch
   torch.mps.empty_cache()  # For MPS
   ```

## Understanding the Workflow

### Image Generation Flow
```
Text Prompt → Tokenizer → Language Model → 576 Image Tokens → Vision Decoder → 384×384 Image
```

### Image Understanding Flow
```
Image → SigLIP Encoder → Visual Features → Language Model → Text Response
```

### Key Components

1. **VLChatProcessor**: Handles all preprocessing for text and images
2. **MultiModalityCausalLM**: The main model combining language and vision
3. **Two Encoders**:
   - SigLIP for understanding (semantic features)
   - VQ Tokenizer for generation (spatial details)

## Common Issues and Solutions

### Out of Memory Error
- Use Janus-1.5B instead of 7B
- Close other applications
- Reduce batch size to 1
- Use `low_cpu_mem_usage=True` when loading

### Slow Generation
- Ensure MPS is being used (check device)
- Image generation takes ~2-5 minutes on M1
- Understanding is much faster (~10-30 seconds)

### Import Errors
- Make sure you're in the virtual environment
- Reinstall Janus: `cd Janus && pip install -e . --force-reinstall`

## Example Use Cases

### 1. Basic Image Generation
```python
model, processor, device = setup_model(use_small_model=True)
generate_image_simple(model, processor, device, "A sunset over mountains")
```

### 2. Image Analysis
```python
answer = understand_image_simple(
    model, processor, device, 
    "photo.jpg", 
    "What objects are in this image?"
)
```

### 3. Creative Workflow
```python
# Analyze an image's style
style = understand_image_simple(model, processor, device, "art.jpg", "Describe the artistic style")

# Generate new image in that style
generate_image_simple(model, processor, device, f"A city in the style of: {style}")
```

## Performance Expectations

| Model | Task | M1 Mac | M1 Pro/Max | M2/M3 |
|-------|------|---------|------------|--------|
| Janus-1.5B | Generation | ~3 min | ~2 min | ~1.5 min |
| Janus-1.5B | Understanding | ~20s | ~10s | ~8s |
| Janus-7B | Generation | ~8 min | ~5 min | ~4 min |
| Janus-7B | Understanding | ~45s | ~25s | ~20s |

## Next Steps

1. Try different prompts and see what works best
2. Experiment with temperature and cfg_weight parameters
3. Build your own applications using the provided classes
4. Consider fine-tuning for specific use cases

## Resources

- [Janus GitHub](https://github.com/deepseek-ai/Janus)
- [Model Weights](https://huggingface.co/deepseek-ai)
- [Technical Paper](https://arxiv.org/abs/2410.13848)
- Our analysis: `deepseek-janus-analysis.md`

Happy generating! 🎨🤖