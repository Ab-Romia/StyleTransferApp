# 🚀 Hugging Face Deployment Guide

Complete guide for deploying the Neural Style Transfer app on Hugging Face Spaces.

## Prerequisites

- Hugging Face account
- Git with LFS installed
- Trained model weights (optional, will work without)

## Quick Deployment

### Option 1: Web Interface (Easiest)

1. **Go to Hugging Face Spaces**
   - Visit https://huggingface.co/spaces
   - Click "Create new Space"

2. **Configure Space**
   - Name: `neural-style-transfer`
   - License: `MIT`
   - SDK: `Gradio`
   - Hardware: `CPU Basic` (or `T4 small` for GPU)

3. **Upload Files**
   - Drag and drop these files:
     - `gradio_app.py`
     - `requirements_hf.txt` → rename to `requirements.txt`
     - `README_HF.md` → rename to `README.md`
     - `models/` folder (all Python files)
     - `examples/` folder (optional)

4. **Add Model Weights** (Optional)
   - If you have trained weights:
     - Upload to `models/adain_weights.pth`
     - Use Git LFS for files >10MB

5. **Build and Launch**
   - Space will automatically build
   - Check logs for any errors
   - Should be live in 2-5 minutes

### Option 2: Git (Advanced)

```bash
# 1. Create Space on Hugging Face
# Get your Space URL: https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

# 2. Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME

# 3. Setup Git LFS
git lfs install
git lfs track "*.pth"

# 4. Copy files
cp /path/to/StyleTransferApp/gradio_app.py app.py
cp /path/to/StyleTransferApp/requirements_hf.txt requirements.txt
cp /path/to/StyleTransferApp/README_HF.md README.md
cp -r /path/to/StyleTransferApp/models .
cp -r /path/to/StyleTransferApp/examples .

# 5. Add .gitattributes
cat > .gitattributes << EOF
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
EOF

# 6. Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## File Structure

Your Hugging Face Space should have:

```
space_name/
├── app.py                    # gradio_app.py renamed
├── requirements.txt          # requirements_hf.txt renamed
├── README.md                 # README_HF.md renamed
├── .gitattributes           # For Git LFS
├── models/
│   ├── __init__.py
│   ├── adain_model.py
│   ├── cnn_model.py
│   ├── vit_model.py
│   ├── losses.py
│   └── adain_weights.pth    # Optional
└── examples/
    ├── content1.jpg
    ├── content2.jpg
    ├── content3.jpg
    ├── style1.jpg
    ├── style2.jpg
    └── style3.jpg
```

## Setup Examples

Run the setup script to download example images:

```bash
python setup_hf.py
```

This will:
- Create `examples/` directory
- Download sample content and style images
- Verify all required files exist

## Model Weights

### Without Pretrained Weights

The app will work but produce random/poor results:
- Good for demonstration of architecture
- Fast deployment
- Can still show the interface

### With Pretrained Weights

For best results, include trained weights:

1. **Train the model** (see TRAINING_GUIDE.md)
2. **Get the best weights**:
   ```bash
   cp outputs/AdaIN/checkpoints/best_model.pth models/adain_weights.pth
   ```
3. **Upload using Git LFS**:
   ```bash
   git lfs track "*.pth"
   git add models/adain_weights.pth
   git commit -m "Add trained weights"
   git push
   ```

## Hardware Options

### CPU Basic (Free)
- **Cost**: Free
- **Speed**: 2-5 seconds per image
- **Best for**: Demos, testing
- **Limitation**: Slower inference

### T4 Small (Paid)
- **Cost**: ~$0.60/hour
- **Speed**: 50-100ms per image
- **Best for**: Production, fast response
- **Benefit**: GPU acceleration

## Configuration

### Adjust Settings in gradio_app.py

**Max Image Size:**
```python
transform = transforms.Compose([
    transforms.Resize(512),  # Change to 1024 for higher quality
    transforms.ToTensor(),
])
```

**Default Parameters:**
```python
style_strength = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=1.0,  # Change default here
    ...
)
```

**Add More Examples:**
```python
gr.Examples(
    examples=[
        ["examples/my_content.jpg", "examples/my_style.jpg", 1.0, False],
        # Add more here
    ],
    ...
)
```

## Troubleshooting

### Build Fails

**Error: "Module not found"**
- Check `requirements.txt` has all dependencies
- Verify spelling and versions

**Error: "Out of memory"**
- Reduce max image size in code
- Use CPU instead of GPU hardware
- Add memory cleanup: `torch.cuda.empty_cache()`

### App Loads but Errors

**"Model not found"**
- Check `models/` directory exists
- Verify all `.py` files are present
- Check file paths in `gradio_app.py`

**"No such file: examples/..."**
- Run `python setup_hf.py` to download examples
- Or remove examples from `gr.Examples()`

**Poor Quality Results**
- Upload trained weights to `models/adain_weights.pth`
- Train model using `train_style_transfer_colab.ipynb`

## Testing Locally

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements_hf.txt

# Run setup
python setup_hf.py

# Launch app
python gradio_app.py
```

Open http://localhost:7860 in your browser.

## Post-Deployment

### Monitor Performance

1. **Check Logs**
   - Go to your Space settings
   - View build and runtime logs
   - Monitor for errors

2. **Test Functionality**
   - Upload various images
   - Try different parameters
   - Check processing time

3. **Gather Feedback**
   - Share your Space URL
   - Monitor user interactions
   - Iterate on UI/UX

### Update Space

To update your deployed app:

```bash
# Make changes locally
git add .
git commit -m "Update description"
git push
```

Space will automatically rebuild.

## Example Space URL

After deployment, your Space will be at:
```
https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer
```

Share this URL to let others use your app!

## Additional Features

### Add Custom Styles

1. Upload style images to `examples/`
2. Update `gr.Examples()` in `gradio_app.py`
3. Commit and push

### Enable API

In Space settings:
- Enable "API" toggle
- Users can call your model programmatically
- Generates API documentation automatically

### Analytics

- View Space analytics in settings
- Track usage and popularity
- Monitor performance metrics

## Support

If you encounter issues:
1. Check Hugging Face documentation
2. Review build logs
3. Test locally first
4. Open GitHub issue if needed

## Resources

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://gradio.app/docs
- **Git LFS**: https://git-lfs.github.com/
- **Project Repo**: https://github.com/Ab-Romia/StyleTransferApp

---

**Ready to deploy!** 🚀

Follow the steps above and your Style Transfer app will be live on Hugging Face Spaces.
