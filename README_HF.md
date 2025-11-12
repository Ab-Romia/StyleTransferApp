---
title: Neural Style Transfer
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.0.0
app_file: gradio_app.py
pinned: false
license: mit
---

# 🎨 Neural Style Transfer

Transform your photos with artistic styles using state-of-the-art deep learning.

## Features

- ⚡ **Real-time processing** - 100x faster than traditional methods
- 🎨 **Arbitrary style transfer** - Use any image as a style reference
- 🎛️ **Adjustable parameters** - Control style strength and color preservation
- 🚀 **GPU accelerated** - Optimized for fast inference

## How It Works

This demo uses **AdaIN (Adaptive Instance Normalization)** architecture:

1. **Content Encoder** - Extracts features from your photo
2. **Style Encoder** - Captures artistic patterns from style image
3. **AdaIN Layer** - Aligns content features with style statistics
4. **Decoder** - Reconstructs the stylized image

## Usage

1. Upload a content image (your photo)
2. Upload a style image (artwork/pattern)
3. Adjust style strength (0.0 - 1.0)
4. Enable color preservation if desired
5. Click "Apply Style Transfer"

## Technical Details

- **Model:** AdaIN with Attention Mechanism
- **Encoder:** VGG19 (pretrained on ImageNet)
- **Decoder:** Enhanced decoder with residual connections
- **Speed:** ~50-100ms per image on GPU
- **Max Resolution:** 1024x1024 pixels

## Examples

Try the built-in examples to see different artistic styles applied to photos.

## Citation

Based on:
```bibtex
@inproceedings{huang2017adain,
  title={Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={ICCV},
  year={2017}
}
```

## Repository

Full source code and training notebooks available at:
[github.com/Ab-Romia/StyleTransferApp](https://github.com/Ab-Romia/StyleTransferApp)

## Author

**Abdelrahman Abouroumia**
- Email: aabouroumia@gmail.com
- GitHub: [@Ab-Romia](https://github.com/Ab-Romia)

## License

MIT License - See LICENSE file for details
