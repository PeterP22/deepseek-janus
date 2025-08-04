# DeepSeek Janus Analysis and Experiments

This repository contains a comprehensive analysis of DeepSeek's Janus multimodal AI model, including practical experiments and detailed architectural insights.

## Contents

- `DeepSeek_Janus_Pro_7b.ipynb` - Jupyter notebook with hands-on experiments using Janus-Pro-7B
- `deepseek-janus-analysis.md` - Comprehensive analysis of the Janus architecture and innovations

## Quick Start

### Prerequisites
```bash
pip install torch transformers timm accelerate sentencepiece attrdict einops
```

### Installation
```bash
git clone https://github.com/deepseek-ai/Janus
cd Janus
pip install -e .
```

### Basic Usage

#### Image Generation
```python
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Load model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Generate image from text
prompt = "A stunning cat portrait in nature photography style"
# ... see notebook for full implementation
```

#### Image Understanding
```python
# Analyze an image
conversation = [
    {
        "role": "<|User|>",
        "content": "<image_placeholder>\nDescribe this image",
        "images": ['path/to/image.png'],
    }
]
# ... see notebook for full implementation
```

## Key Findings

### 1. Decoupled Visual Encoding
Janus introduces separate encoding pathways for understanding and generation tasks, solving the fundamental conflict between semantic understanding and detailed generation.

### 2. Unified Architecture
Despite separate encoders, Janus maintains a single transformer backbone, enabling seamless switching between tasks.

### 3. Superior Performance
Janus-Pro-7B outperforms DALL-E 3 on standard benchmarks while maintaining strong visual understanding capabilities.

## Model Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Input     │ --> │ Decoupled Visual │ --> │  Unified    │
│   Image     │     │    Encoders      │     │ Transformer │
└─────────────┘     └──────────────────┘     └─────────────┘
                           /    \
                          /      \
                  Understanding  Generation
                    (SigLIP)    (VQ Tokenizer)
```

## Resources

- [Original Paper (2024)](https://arxiv.org/html/2410.13848v1)
- [Janus-Pro Paper (2025)](https://arxiv.org/abs/2501.17811)
- [Official GitHub](https://github.com/deepseek-ai/Janus)
- [Hugging Face Models](https://huggingface.co/deepseek-ai/Janus-Pro-7B)

## License

- Code: MIT License
- Model Weights: DeepSeek Model License

## Citation

```bibtex
@article{wu2024janus,
  title={Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation},
  author={Wu, Chengyue and Chen, Xiaokang and others},
  journal={arXiv preprint arXiv:2410.13848},
  year={2024}
}

@article{chen2025januspro,
  title={Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling},
  author={Chen, Xiaokang and Wu, Zhiyu and others},
  journal={arXiv preprint arXiv:2501.17811},
  year={2025}
}
```