# 🚀 Quick Start: Hugging Face Deployment

Deploy your Neural Style Transfer model to Hugging Face Spaces in 3 simple steps.

## Prerequisites

- Hugging Face account (free)
- Trained model weights (optional)

## Step 1: Prepare Files

Run the setup script:

```bash
python setup_hf.py
```

This will:
- Download example images
- Verify all required files
- Check model structure

## Step 2: Test Locally

```bash
python gradio_app.py
```

Open http://localhost:7860 and test:
- Upload content and style images
- Try different settings
- Verify results look good

## Step 3: Deploy

### Option A: Web Interface (Easiest)

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `neural-style-transfer`
4. SDK: Gradio
5. Upload these files:
   - `gradio_app.py` → rename to `app.py`
   - `requirements_hf.txt` → rename to `requirements.txt`
   - `README_HF.md` → rename to `README.md`
   - `models/` folder
   - `examples/` folder

### Option B: Git (Advanced)

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer
cd neural-style-transfer

# Setup Git LFS
git lfs install

# Copy files
cp gradio_app.py app.py
cp requirements_hf.txt requirements.txt
cp README_HF.md README.md
cp -r models .
cp -r examples .
cp .gitattributes .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## That's It!

Your Space will build automatically and be live in 2-5 minutes at:
```
https://huggingface.co/spaces/YOUR_USERNAME/neural-style-transfer
```

## Need Help?

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Detailed instructions
- Troubleshooting guide
- Advanced configuration
- Adding custom features

## Files Overview

| File | Purpose |
|------|---------|
| `gradio_app.py` | Main Gradio interface |
| `requirements_hf.txt` | Python dependencies |
| `README_HF.md` | Space documentation |
| `setup_hf.py` | Setup and download examples |
| `DEPLOYMENT.md` | Complete deployment guide |
| `.gitattributes` | Git LFS configuration |

## Tips

✅ Test locally first with `python gradio_app.py`
✅ Use GPU hardware for faster inference
✅ Include trained weights for best results
✅ Check logs if build fails
✅ Share your Space URL with others!

---

**Ready to deploy!** Follow the steps above and your style transfer app will be live on Hugging Face.
