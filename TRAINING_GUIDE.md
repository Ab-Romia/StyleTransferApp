# 🎓 Style Transfer Training Guide for Google Colab

## 📋 Overview

This guide explains how to train the three style transfer models (AdaIN, CNN, ViT) using the provided Jupyter notebook on Google Colab Free GPU.

## 🚀 Quick Start

### 1. Open in Colab

1. Upload `train_style_transfer_colab.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Or use this direct link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Enable GPU

1. Go to `Runtime` → `Change runtime type`
2. Select `T4 GPU` (free tier)
3. Click `Save`

### 3. Run the Notebook

Simply run all cells in order (or use `Runtime` → `Run all`)

## 📊 What You'll Get

### Training Outputs

The notebook trains all three models and provides:

1. **Model Weights** (`.pth` files)
   - `outputs/AdaIN/checkpoints/best_model.pth`
   - `outputs/CNN/checkpoints/best_model.pth`
   - `outputs/ViT/checkpoints/best_model.pth`

2. **Performance Visualizations**
   - Training/validation loss curves
   - Model comparison charts
   - Speed benchmarks
   - Quality metrics (LPIPS, MSE)

3. **Sample Outputs**
   - Generated images during training
   - Side-by-side model comparisons
   - Visual quality assessment

## 📈 Dataset Information

### Content Images: MS-COCO 2017
- **Size**: 10,000 images (subset for Colab)
- **Type**: Natural photographs
- **Source**: http://cocodataset.org
- **Purpose**: Content structure

### Style Images: WikiArt
- **Size**: Variable (depends on Kaggle download)
- **Type**: Artistic paintings
- **Styles**: 27 different art movements
- **Source**: Kaggle (via kaggle.json)
- **Purpose**: Artistic styles

### Important Notes
- You'll need a **Kaggle API token** (`kaggle.json`)
- Get it from: https://www.kaggle.com/settings/account
- The notebook will prompt you to upload it

## 🎯 Training Configuration

### Optimized for Colab Free GPU

```python
CONFIG = {
    'image_size': 256,      # Memory-efficient
    'batch_size': 4,        # Fits in 15GB VRAM
    'num_epochs': 20,       # ~3-4 hours total
    'learning_rate': 1e-4,
    'use_amp': True,        # Mixed precision (2x speedup)
}
```

### Hardware Requirements
- **GPU**: T4 (15GB VRAM) - Free on Colab
- **RAM**: ~12GB system RAM
- **Storage**: ~8GB for datasets + models
- **Time**: 3-4 hours for all 3 models

## 📊 Expected Performance

### Training Time (20 epochs each)
- **AdaIN**: ~45-60 minutes
- **CNN**: ~50-70 minutes
- **ViT**: ~70-90 minutes
- **Total**: ~3-4 hours

### Inference Speed (512x512)
- **AdaIN**: ~50-80ms (fastest)
- **CNN**: ~80-120ms
- **ViT**: ~150-200ms (best quality)

### Quality Metrics
After training, you'll see:
- **Loss curves** showing convergence
- **LPIPS scores** (perceptual quality)
- **MSE values** (structural preservation)
- **Visual comparisons** of all models

## 🎨 Model Comparison

### AdaIN (Recommended)
✅ **Pros**:
- Fastest inference (real-time capable)
- Arbitrary style transfer
- Good quality
- Flexible (multi-style, color preservation)

⚠️ **Cons**:
- Slightly less detail than ViT

🎯 **Best for**: Web apps, mobile, real-time video

### CNN (Traditional)
✅ **Pros**:
- Predictable, stable results
- Style intensity control
- Good for specific styles

⚠️ **Cons**:
- One style per model
- Medium speed

🎯 **Best for**: Controlled stylization, batch processing

### ViT (Transformer)
✅ **Pros**:
- Highest quality
- Global context understanding
- Best semantic awareness

⚠️ **Cons**:
- Slowest inference
- More memory intensive

🎯 **Best for**: Professional art, high-quality outputs

## 📥 Downloading Results

The notebook automatically packages everything:

```python
# Download all results as ZIP
files.download('trained_models.zip')
```

Contains:
- ✅ All model weights
- ✅ Training curves (PNG)
- ✅ Sample outputs
- ✅ Comparison visualizations
- ✅ Performance metrics

## 🔧 Customization Options

### Adjust Training Duration

```python
CONFIG['num_epochs'] = 30  # More epochs = better quality
```

### Change Image Resolution

```python
CONFIG['image_size'] = 512  # Higher = better quality, slower
# Note: May need to reduce batch_size for 512px
```

### Modify Loss Weights

```python
CONFIG['style_weight'] = 150.0  # Stronger style
CONFIG['content_weight'] = 2.0  # More content preservation
```

### Use Different Datasets

Replace the download sections with your own datasets:
```python
content_dir = 'path/to/your/content/images'
style_dir = 'path/to/your/style/images'
```

## 📊 Understanding the Plots

### 1. Training Loss Curves
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should decrease (may plateau)
- **Gap**: Small gap = good generalization

### 2. Loss Improvement Chart
- Shows % improvement from start
- Good models show 50-80% improvement

### 3. Speed Comparison
- Horizontal bar chart
- Lower = faster (better for production)

### 4. Visual Comparison
- Side-by-side outputs
- Compare quality across models

### 5. Metrics Radar Chart
- Speed, Quality (LPIPS), Structure (MSE)
- Larger area = better overall

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
CONFIG['batch_size'] = 2

# Reduce image size
CONFIG['image_size'] = 224

# Disable LPIPS
CONFIG['use_lpips'] = False
```

### Kaggle Download Fails
```python
# Use alternative style images
# The notebook will auto-download famous artworks
# Or upload your own to data/wikiart/
```

### Training Too Slow
```python
# Reduce epochs
CONFIG['num_epochs'] = 10

# Use smaller dataset
# Keep only 5000 content images instead of 10000
```

### Colab Disconnects
```python
# The notebook saves checkpoints every 5 epochs
# You can resume from last checkpoint:
trainer.load_checkpoint('outputs/AdaIN/checkpoints/checkpoint_epoch015.pth')
```

## 🎓 Advanced Usage

### Fine-tune on Specific Style

1. Replace WikiArt with your style images
2. Train with higher style weight:
```python
CONFIG['style_weight'] = 200.0
```

### Transfer Learning

Start from pre-trained weights:
```python
model.load_state_dict(torch.load('pretrained_weights.pth'))
# Then continue training
```

### Export for Production

Convert to ONNX for deployment:
```python
torch.onnx.export(
    model,
    (dummy_content, dummy_style),
    'adain_model.onnx',
    opset_version=11
)
```

## 📚 Additional Resources

### Papers
- **AdaIN**: [Arbitrary Style Transfer in Real-time](https://arxiv.org/abs/1703.06868)
- **Original Style Transfer**: [Image Style Transfer Using CNNs](https://arxiv.org/abs/1508.06576)

### Datasets
- **MS-COCO**: http://cocodataset.org/
- **WikiArt**: https://www.wikiart.org/
- **Kaggle WikiArt**: https://www.kaggle.com/datasets

### Related Projects
- Original AdaIN: https://github.com/naoto0804/pytorch-AdaIN
- Fast Style Transfer: https://github.com/pytorch/examples/tree/master/fast_neural_style

## 🤝 Contributing

Found a bug or have improvements? Please open an issue or PR:
https://github.com/Ab-Romia/StyleTransferApp/issues

## 📧 Support

- **Issues**: GitHub Issues
- **Email**: aabouroumia@gmail.com
- **Discussions**: GitHub Discussions

---

## ✅ Checklist

Before starting training:
- [ ] Enable GPU in Colab (Runtime → Change runtime type → GPU)
- [ ] Have Kaggle API token ready (`kaggle.json`)
- [ ] Ensure stable internet connection
- [ ] Have 3-4 hours for full training
- [ ] Enough Google Drive space (~2GB for outputs)

During training:
- [ ] Monitor loss curves (should decrease)
- [ ] Check sample outputs periodically
- [ ] Watch for OOM errors
- [ ] Keep Colab tab active (prevents disconnect)

After training:
- [ ] Download trained models
- [ ] Save training plots
- [ ] Test models on new images
- [ ] Compare quality vs speed
- [ ] Choose best model for your use case

---

**Happy Training! 🎨**

For the latest updates, visit: https://github.com/Ab-Romia/StyleTransferApp
