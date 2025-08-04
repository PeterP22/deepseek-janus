#!/usr/bin/env python3
"""
Simple Janus Demo - Minimal Example
===================================

This is a simplified version focusing on the essential workflow
for running DeepSeek Janus locally on your Mac.

Key Concepts:
1. Janus uses separate encoders for understanding vs generation
2. The model runs on MPS (Metal) for Mac GPU acceleration  
3. Image generation creates 384x384 images by default
4. The model can both generate images AND understand them

Author: Your Name
Date: August 2025
"""

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import PIL.Image
import numpy as np
import os


def setup_model(use_small_model=True):
    """
    Step 1: Load the Janus model
    
    We have two options:
    - Janus-1.5B: Smaller, faster, uses less memory (~6GB)
    - Janus-Pro-7B: Larger, better quality, needs more memory (~28GB)
    """
    # Choose model based on your Mac's memory
    if use_small_model:
        model_path = "deepseek-ai/Janus-1.5B"
        print("📊 Using Janus-1.5B (smaller model, good for most Macs)")
    else:
        model_path = "deepseek-ai/Janus-Pro-7B"
        print("📊 Using Janus-Pro-7B (larger model, needs 32GB+ RAM)")
    
    # Load the processor (handles text/image preprocessing)
    print("Loading processor...")
    processor = VLChatProcessor.from_pretrained(model_path)
    
    # Load the model with Mac optimizations
    print("Loading model (this takes 1-2 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for Mac
        low_cpu_mem_usage=True        # Important for memory efficiency
    )
    
    # Use Mac's GPU (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Mac GPU acceleration (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (will be slower)")
    
    model = model.to(device).eval()
    
    return model, processor, device


def generate_image_simple(model, processor, device, prompt):
    """
    Step 2: Generate an image from text
    
    The generation process:
    1. Convert prompt to tokens
    2. Generate 576 image tokens (24x24 grid)
    3. Decode tokens to actual image pixels
    """
    print(f"\n🎨 Generating image: '{prompt}'")
    
    # Format the prompt in Janus conversation style
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # Prepare the prompt
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag
    
    # Tokenize
    input_ids = processor.tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    
    # Key parameters explained:
    # - 384x384: Output image size
    # - 16: Patch size (each patch becomes a token)
    # - 576: Total tokens (384/16)² = 24² = 576
    img_size = 384
    patch_size = 16
    num_tokens = 576
    
    print(f"Generating {num_tokens} image tokens...")
    
    # For classifier-free guidance, we need 2 copies
    tokens = input_ids.repeat(2, 1)
    tokens[1, 1:-1] = processor.pad_id  # Mask the second copy
    
    # Get initial embeddings
    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    
    # Generate tokens one by one
    generated_tokens = []
    past_key_values = None
    
    for i in range(num_tokens):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_tokens} tokens")
        
        # Forward pass
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        
        # Get next token probabilities
        logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
        
        # Classifier-free guidance (improves quality)
        cfg_weight = 5.0
        logits_guided = logits[1] + cfg_weight * (logits[0] - logits[1])
        
        # Sample next token
        probs = torch.softmax(logits_guided, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token.item())
        
        # Prepare for next iteration
        next_token_both = torch.tensor([next_token, next_token]).to(device)
        img_embeds = model.prepare_gen_img_embeds(next_token_both)
        inputs_embeds = img_embeds.unsqueeze(1)
    
    # Convert tokens to image
    print("Decoding to image...")
    generated_tokens = torch.tensor(generated_tokens).unsqueeze(0).to(device)
    
    # Decode using the vision model
    image = model.gen_vision_model.decode_code(
        generated_tokens.int(),
        shape=[1, 8, 24, 24]  # [batch, channels, height, width]
    )
    
    # Convert to PIL Image
    image = image[0].cpu().float().numpy()
    image = (image + 1) / 2 * 255  # Denormalize from [-1,1] to [0,255]
    image = np.clip(image.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    
    # Save
    save_path = "generated_image.png"
    PIL.Image.fromarray(image).save(save_path)
    print(f"✅ Saved to {save_path}")
    
    return save_path


def understand_image_simple(model, processor, device, image_path, question):
    """
    Step 3: Understand an image
    
    The understanding process:
    1. Load image through SigLIP encoder
    2. Convert visual features to text space
    3. Generate text response
    """
    print(f"\n🔍 Understanding image: {image_path}")
    print(f"❓ Question: {question}")
    
    # Format as conversation
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # Load and preprocess image
    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(device)
    
    # Get image embeddings from vision encoder
    print("Processing image...")
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    
    # Generate text response
    print("Generating response...")
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )
    
    # Decode response
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's answer
    if "<|Assistant|>:" in response:
        response = response.split("<|Assistant|>:")[-1].strip()
    
    return response


def main():
    """
    Main demo showing both capabilities
    """
    print("="*60)
    print("🚀 Simple Janus Demo")
    print("="*60)
    
    # Setup (do this once)
    model, processor, device = setup_model(use_small_model=True)
    
    # Demo 1: Generate an image
    print("\n" + "="*60)
    print("Demo 1: Text-to-Image Generation")
    print("="*60)
    
    prompt = "A cute robot watering plants in a garden"
    image_path = generate_image_simple(model, processor, device, prompt)
    
    # Demo 2: Understand the generated image
    print("\n" + "="*60)
    print("Demo 2: Image Understanding")
    print("="*60)
    
    question = "What is happening in this image? Describe what you see."
    answer = understand_image_simple(model, processor, device, image_path, question)
    
    print(f"\n🤖 Janus says: {answer}")
    
    # Demo 3: Understanding + Generation workflow
    print("\n" + "="*60)
    print("Demo 3: Complete Workflow")
    print("="*60)
    
    # First understand an image
    if os.path.exists("example.jpg"):
        analysis = understand_image_simple(
            model, processor, device,
            "example.jpg",
            "Describe the style and mood of this image"
        )
        print(f"Analysis: {analysis}")
        
        # Then generate a similar image
        new_prompt = f"Create an image in a similar style: {analysis[:100]}..."
        generate_image_simple(model, processor, device, new_prompt)


if __name__ == "__main__":
    main()