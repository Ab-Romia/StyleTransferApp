"""
Neural Style Transfer Application for Hugging Face Spaces
Fast, real-time style transfer with AdaIN architecture
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO

from models.adain_model import AdaINStyleTransfer
from torchvision import transforms

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    """Load the AdaIN model"""
    global model
    if model is None:
        model = AdaINStyleTransfer(use_attention=True).to(device)
        model.eval()

        # Try to load pretrained weights
        checkpoint_path = Path('models/adain_weights.pth')
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("✓ Loaded pretrained weights")
            except Exception as e:
                print(f"Using untrained model: {e}")
    return model

def process_image(content_img, style_img, style_strength, preserve_color):
    """Process style transfer"""
    if content_img is None or style_img is None:
        return None

    # Load model
    model = load_model()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    # Process images
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    # Apply style transfer
    with torch.no_grad():
        output = model(
            content_tensor,
            style_tensor,
            alpha=style_strength,
            preserve_color=preserve_color
        )

    # Convert to PIL
    output = output.squeeze(0).cpu().clamp(0, 1)
    output = transforms.ToPILImage()(output)

    return output

def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(title="Neural Style Transfer") as demo:
        gr.Markdown("""
        # 🎨 Neural Style Transfer

        Transform your photos with artistic styles using state-of-the-art AdaIN architecture.

        **Features:**
        - ⚡ Real-time processing (100x faster than traditional methods)
        - 🎨 Arbitrary style transfer
        - 🎛️ Adjustable style strength
        - 🌈 Optional color preservation
        """)

        with gr.Row():
            with gr.Column():
                content_input = gr.Image(
                    label="Content Image",
                    type="pil",
                    height=300
                )
                style_input = gr.Image(
                    label="Style Image",
                    type="pil",
                    height=300
                )

                with gr.Row():
                    style_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        label="Style Strength"
                    )
                    preserve_color = gr.Checkbox(
                        label="Preserve Colors",
                        value=False
                    )

                process_btn = gr.Button("✨ Apply Style Transfer", variant="primary")

            with gr.Column():
                output_image = gr.Image(
                    label="Result",
                    type="pil",
                    height=600
                )

        # Examples
        gr.Markdown("### 📸 Try These Examples")
        gr.Examples(
            examples=[
                ["examples/content1.jpg", "examples/style1.jpg", 1.0, False],
                ["examples/content2.jpg", "examples/style2.jpg", 0.8, False],
                ["examples/content3.jpg", "examples/style3.jpg", 1.0, True],
            ],
            inputs=[content_input, style_input, style_strength, preserve_color],
            outputs=output_image,
            fn=process_image,
            cache_examples=False,
        )

        # Information
        gr.Markdown("""
        ### ℹ️ About

        This demo uses **AdaIN (Adaptive Instance Normalization)** for real-time style transfer.

        **Tips:**
        - Upload your own images or try the examples
        - Adjust style strength for different effects
        - Enable color preservation to keep original colors
        - Works best with images under 1024x1024

        **Model:** AdaIN with Attention Mechanism
        **Speed:** ~50-100ms per image on GPU
        **Architecture:** VGG19 Encoder + Enhanced Decoder

        ---

        **Developed by:** Abdelrahman Abouroumia
        **Repository:** [GitHub](https://github.com/Ab-Romia/StyleTransferApp)
        """)

        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[content_input, style_input, style_strength, preserve_color],
            outputs=output_image
        )

    return demo

if __name__ == "__main__":
    # Load model at startup
    print("Loading model...")
    load_model()
    print("✓ Model ready")

    # Launch app
    demo = create_demo()
    demo.launch()
