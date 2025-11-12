"""
Setup script for Hugging Face deployment
Downloads example images and prepares the environment
"""

import os
from pathlib import Path
import urllib.request
from PIL import Image
import io

def download_image(url, save_path):
    """Download image from URL"""
    try:
        print(f"Downloading {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ Downloaded {save_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {save_path}: {e}")
        return False

def create_placeholder_image(save_path, text, size=(512, 512)):
    """Create a placeholder image"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', size, color=(200, 200, 220))
    draw = ImageDraw.Draw(img)

    # Draw text
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill=(100, 100, 120))

    img.save(save_path)
    print(f"✓ Created placeholder: {save_path}")

def setup_examples():
    """Setup example images"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    # Example images (using publicly available images)
    example_urls = {
        "content1.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512",
        "content2.jpg": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=512",
        "content3.jpg": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=512",
        "style1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/512px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "style2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Claude_Monet_-_Water_Lilies_-_1916_-_Google_Art_Project.jpg/512px-Claude_Monet_-_Water_Lilies_-_1916_-_Google_Art_Project.jpg",
        "style3.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/512px-The_Scream.jpg",
    }

    print("Setting up example images...")
    for filename, url in example_urls.items():
        save_path = examples_dir / filename
        if not save_path.exists():
            success = download_image(url, save_path)
            if not success:
                # Create placeholder if download fails
                img_type = "Content" if "content" in filename else "Style"
                create_placeholder_image(
                    save_path,
                    f"{img_type} Example\n{filename}"
                )

    print("\n✓ Example images ready")

def check_models():
    """Check if model files exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("✗ Models directory not found")
        return False

    required_files = [
        "adain_model.py",
        "cnn_model.py",
        "vit_model.py",
        "losses.py",
        "__init__.py"
    ]

    print("\nChecking model files...")
    all_exist = True
    for file in required_files:
        path = models_dir / file
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            all_exist = False

    # Check for pretrained weights (optional)
    weights_path = models_dir / "adain_weights.pth"
    if weights_path.exists():
        print(f"✓ Pretrained weights found")
    else:
        print(f"⚠ No pretrained weights (will use untrained model)")

    return all_exist

def main():
    """Main setup function"""
    print("=" * 60)
    print("Hugging Face Deployment Setup")
    print("=" * 60)

    # Setup examples
    setup_examples()

    # Check models
    print()
    if check_models():
        print("\n✓ All required files present")
    else:
        print("\n⚠ Some files are missing")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python gradio_app.py")
    print("2. Or deploy to Hugging Face Spaces")
    print("3. Use README_HF.md as your Space README")

if __name__ == "__main__":
    main()
