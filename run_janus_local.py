#!/usr/bin/env python3
"""
DeepSeek Janus Local Runner
==========================

This script demonstrates how to run DeepSeek Janus models locally on your Mac.
It includes both image generation and image understanding capabilities.

Requirements:
- Python 3.8+
- PyTorch with MPS (Metal Performance Shaders) support for Mac
- Transformers library
- Other dependencies from the Janus package

Author: Your Name
Date: August 2025
"""

import os
import sys
import torch
import warnings
from typing import Optional, List, Union
from pathlib import Path

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Check if running on Mac with Apple Silicon
def check_device():
    """
    Check and return the best available device for inference.
    On Mac with Apple Silicon, this will use MPS (Metal Performance Shaders).
    """
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) detected and will be used for acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU detected")
        return torch.device("cuda")
    else:
        print("⚠️  No GPU detected, using CPU (this will be slower)")
        return torch.device("cpu")

# Import Janus components
try:
    from transformers import AutoModelForCausalLM
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
    import PIL.Image
    print("✅ Successfully imported all required libraries")
except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
    print("\nPlease install the required packages:")
    print("pip install torch transformers timm accelerate sentencepiece attrdict einops pillow")
    print("\nThen clone and install Janus:")
    print("git clone https://github.com/deepseek-ai/Janus")
    print("cd Janus && pip install -e .")
    sys.exit(1)


class JanusLocalRunner:
    """
    A class to manage DeepSeek Janus model for local inference.
    
    This class handles:
    1. Model loading and initialization
    2. Image generation from text prompts
    3. Image understanding and question answering
    4. Memory management for Mac systems
    """
    
    def __init__(self, model_path: str = "deepseek-ai/Janus-Pro-7B", 
                 device: Optional[torch.device] = None,
                 use_smaller_model: bool = False):
        """
        Initialize the Janus model runner.
        
        Args:
            model_path: HuggingFace model path or local path
            device: PyTorch device to use (auto-detected if None)
            use_smaller_model: If True, uses Janus-1B instead of 7B for lower memory usage
        """
        self.device = device if device else check_device()
        
        # Use smaller model if requested (better for Macs with limited memory)
        if use_smaller_model:
            model_path = "deepseek-ai/Janus-1.5B"
            print("📊 Using smaller Janus-1.5B model for reduced memory usage")
        
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        print(f"\n🚀 Initializing Janus model: {model_path}")
        self._load_model()
        
    def _load_model(self):
        """
        Load the model and processor with optimizations for Mac.
        """
        try:
            # Load the processor (handles text and image preprocessing)
            print("📥 Loading VLChatProcessor...")
            self.processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer
            
            # Load the model with optimizations
            print("📥 Loading Janus model (this may take a few minutes)...")
            
            # For Mac, we'll use bfloat16 if available, otherwise float16
            dtype = torch.bfloat16 if self.device.type == "mps" else torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True  # Important for Mac memory management
            )
            
            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device).eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"📊 Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    @torch.inference_mode()
    def generate_image(self, 
                      prompt: str, 
                      num_images: int = 1,
                      temperature: float = 1.0,
                      cfg_weight: float = 5.0,
                      save_dir: str = "./generated_images"):
        """
        Generate images from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate (default: 1)
            temperature: Sampling temperature (higher = more random)
            cfg_weight: Classifier-free guidance weight (higher = stronger prompt adherence)
            save_dir: Directory to save generated images
        
        Returns:
            List of paths to generated images
        """
        print(f"\n🎨 Generating {num_images} image(s) from prompt: '{prompt}'")
        
        # Prepare the conversation format for generation
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Apply the SFT template and prepare for generation
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + self.processor.image_start_tag
        
        # Tokenize the prompt
        input_ids = self.processor.tokenizer.encode(prompt_text)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        
        # Key parameters for image generation
        img_size = 384  # Janus generates 384x384 images
        patch_size = 16
        image_token_num = (img_size // patch_size) ** 2  # 576 tokens
        
        print(f"🔧 Image size: {img_size}x{img_size}")
        print(f"🔧 Total image tokens to generate: {image_token_num}")
        
        # Initialize tokens for parallel generation
        parallel_size = num_images * 2  # For CFG, we need 2x
        tokens = input_ids.repeat(parallel_size, 1)
        
        # Mask odd indices for classifier-free guidance
        for i in range(parallel_size):
            if i % 2 != 0:
                tokens[i, 1:-1] = self.processor.pad_id
        
        # Get initial embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        
        # Storage for generated tokens
        generated_tokens = torch.zeros((num_images, image_token_num), dtype=torch.int).to(self.device)
        
        print("🏃 Generating image tokens...")
        # Generate tokens autoregressively
        past_key_values = None
        for i in range(image_token_num):
            # Show progress
            if i % 50 == 0:
                print(f"   Progress: {i}/{image_token_num} tokens ({i/image_token_num*100:.1f}%)")
            
            # Forward pass through the language model
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            
            # Get logits from the generation head
            logits = self.model.gen_head(hidden_states[:, -1, :])
            
            # Apply classifier-free guidance
            logit_cond = logits[0::2, :]    # Conditional logits
            logit_uncond = logits[1::2, :]  # Unconditional logits
            logits_guided = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            
            # Sample next tokens
            probs = torch.softmax(logits_guided / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            # Prepare embeddings for next iteration
            next_token_expanded = next_token.unsqueeze(dim=1).repeat(1, 2).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token_expanded)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        print("🎨 Decoding tokens to images...")
        # Decode tokens to images using the vision model
        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[num_images, 8, img_size//patch_size, img_size//patch_size]
        )
        
        # Convert to numpy and denormalize
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        visual_imgs = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        # Save images
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        for i in range(num_images):
            save_path = os.path.join(save_dir, f"janus_generated_{i}.png")
            PIL.Image.fromarray(visual_imgs[i]).save(save_path)
            saved_paths.append(save_path)
            print(f"💾 Saved image {i+1} to: {save_path}")
        
        return saved_paths
    
    @torch.inference_mode()
    def understand_image(self, 
                        image_path: str, 
                        question: str,
                        max_new_tokens: int = 512) -> str:
        """
        Analyze an image and answer questions about it.
        
        Args:
            image_path: Path to the image file
            question: Question to ask about the image
            max_new_tokens: Maximum tokens to generate in response
            
        Returns:
            String containing the model's response
        """
        print(f"\n🔍 Analyzing image: {image_path}")
        print(f"❓ Question: {question}")
        
        # Prepare conversation with image
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load and preprocess the image
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self.device)
        
        # Run image encoder to get embeddings
        print("🧠 Processing image through vision encoder...")
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        print("💭 Generating response...")
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for understanding
            use_cache=True,
        )
        
        # Decode the response
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|Assistant|>:" in answer:
            answer = answer.split("<|Assistant|>:")[-1].strip()
        
        return answer
    
    def interactive_mode(self):
        """
        Run an interactive session where users can generate images or ask questions.
        """
        print("\n🤖 Welcome to Janus Interactive Mode!")
        print("Commands:")
        print("  'generate: <prompt>' - Generate an image from text")
        print("  'understand: <image_path>' - Analyze an image")
        print("  'quit' - Exit the program")
        print("-" * 50)
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.startswith('generate:'):
                prompt = user_input[9:].strip()
                if prompt:
                    self.generate_image(prompt, num_images=1)
                else:
                    print("❌ Please provide a prompt after 'generate:'")
            
            elif user_input.startswith('understand:'):
                image_path = user_input[11:].strip()
                if os.path.exists(image_path):
                    question = input("What would you like to know about this image? ")
                    answer = self.understand_image(image_path, question)
                    print(f"\n🤖 Answer: {answer}")
                else:
                    print(f"❌ Image not found: {image_path}")
            
            else:
                print("❌ Unknown command. Use 'generate:', 'understand:', or 'quit'")


# Import numpy after other imports
import numpy as np


def main():
    """
    Main function to demonstrate Janus capabilities.
    """
    print("=" * 60)
    print("🚀 DeepSeek Janus Local Runner")
    print("=" * 60)
    
    # Initialize the model
    # Use use_smaller_model=True if you have limited RAM
    runner = JanusLocalRunner(use_smaller_model=False)
    
    # Example 1: Generate an image
    print("\n" + "="*60)
    print("Example 1: Image Generation")
    print("="*60)
    
    prompts = [
        "A serene Japanese garden with cherry blossoms and a koi pond",
        "A futuristic cityscape at sunset with flying cars",
        "A cozy coffee shop interior with warm lighting"
    ]
    
    for prompt in prompts[:1]:  # Generate just one example to save time
        runner.generate_image(prompt, num_images=2)
    
    # Example 2: Understand an image
    print("\n" + "="*60)
    print("Example 2: Image Understanding")
    print("="*60)
    
    # You can test with any image you have
    test_image = "./generated_images/janus_generated_0.png"
    if os.path.exists(test_image):
        answer = runner.understand_image(
            test_image, 
            "Describe this image in detail. What can you see?"
        )
        print(f"\n🤖 Model's response:\n{answer}")
    
    # Example 3: Interactive mode
    print("\n" + "="*60)
    print("Example 3: Starting Interactive Mode")
    print("="*60)
    
    runner.interactive_mode()


if __name__ == "__main__":
    main()