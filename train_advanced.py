"""
Advanced Training Pipeline for Style Transfer

Features:
- AdaIN-based real-time style transfer
- Multi-scale training
- Advanced data augmentation
- Perceptual + LPIPS losses
- Comprehensive validation
- Model checkpointing
- TensorBoard logging
- Mixed precision training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import our models and losses
from models.adain_model import AdaINStyleTransfer, total_variation_loss
from models.losses import CombinedLoss, MultiScaleLoss, VGGPerceptualLoss

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class AdvancedStyleTransferDataset(Dataset):
    """
    Advanced dataset with better augmentation and handling
    """
    def __init__(
        self,
        content_dir,
        style_dir,
        image_size=256,
        mode='train',
        augment=True
    ):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.image_size = image_size
        self.mode = mode
        self.augment = augment

        # Collect image paths
        self.content_images = self._collect_images(self.content_dir)
        self.style_images = self._collect_images(self.style_dir)

        print(f"Loaded {len(self.content_images)} content images")
        print(f"Loaded {len(self.style_images)} style images")

        # Define transforms
        self.transform = self._get_transforms()

    def _collect_images(self, directory):
        """Collect all image files from directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = []

        if directory.is_dir():
            for ext in valid_extensions:
                images.extend(list(directory.rglob(f'*{ext}')))
                images.extend(list(directory.rglob(f'*{ext.upper()}')))

        return sorted(images)

    def _get_transforms(self):
        """Get appropriate transforms based on mode"""
        if self.mode == 'train' and self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')

        # Random style image
        style_idx = np.random.randint(0, len(self.style_images))
        style_path = self.style_images[style_idx]
        style_img = Image.open(style_path).convert('RGB')

        # Apply transforms
        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)

        # Random style strength for training
        if self.mode == 'train':
            alpha = np.random.uniform(0.5, 1.0)
        else:
            alpha = 1.0

        return {
            'content': content_tensor,
            'style': style_tensor,
            'alpha': torch.tensor(alpha, dtype=torch.float32),
            'content_path': str(content_path),
            'style_path': str(style_path)
        }


class Trainer:
    """
    Advanced trainer for style transfer models
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.sample_dir = self.output_dir / 'samples'
        self.log_dir = self.output_dir / 'logs'

        for dir in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print("Initializing model...")
        self.model = AdaINStyleTransfer(use_attention=config['use_attention']).to(self.device)

        # Initialize losses
        print("Initializing loss functions...")
        self.criterion = CombinedLoss(
            content_weight=config['content_weight'],
            style_weight=config['style_weight'],
            perceptual_weight=config['perceptual_weight'],
            lpips_weight=config['lpips_weight'],
            tv_weight=config['tv_weight'],
            use_lpips=config['use_lpips']
        ).to(self.device)

        # Multi-scale loss wrapper
        if config['use_multiscale']:
            self.criterion = MultiScaleLoss(
                self.criterion,
                scales=config['multiscale_scales'],
                weights=config['multiscale_weights']
            )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['lr_patience'],
            verbose=True,
            min_lr=1e-7
        )

        # Mixed precision training
        self.use_amp = config['use_amp'] and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")

        # TensorBoard
        if HAS_TENSORBOARD and config['use_tensorboard']:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0,
            'content': 0,
            'style': 0,
            'perceptual': 0,
            'lpips': 0,
            'tv': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            content = batch['content'].to(self.device)
            style = batch['style'].to(self.device)
            alpha = batch['alpha'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    # Generate stylized image
                    output = self.model(content, style, alpha=alpha.mean().item())

                    # Compute loss
                    if isinstance(self.criterion, MultiScaleLoss):
                        loss = self.criterion(output, content)
                    else:
                        loss, loss_components = self.criterion(
                            output, content, style, return_components=True
                        )

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Generate stylized image
                output = self.model(content, style, alpha=alpha.mean().item())

                # Compute loss
                if isinstance(self.criterion, MultiScaleLoss):
                    loss = self.criterion(output, content)
                    loss_components = {'total': loss}
                else:
                    loss, loss_components = self.criterion(
                        output, content, style, return_components=True
                    )

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                if key in loss_components:
                    epoch_losses[key] += loss_components[key].item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to TensorBoard
            if self.writer and self.global_step % self.config['log_interval'] == 0:
                for key, value in loss_components.items():
                    self.writer.add_scalar(f'train/{key}_loss', value.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

            # Save sample images
            if self.global_step % self.config['sample_interval'] == 0:
                self.save_samples(content[:4], style[:4], output[:4], prefix='train')

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        return epoch_losses

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_losses = {
            'total': 0,
            'content': 0,
            'style': 0,
            'perceptual': 0,
            'lpips': 0,
            'tv': 0
        }

        progress_bar = tqdm(val_loader, desc="Validation")

        for batch_idx, batch in enumerate(progress_bar):
            content = batch['content'].to(self.device)
            style = batch['style'].to(self.device)
            alpha = batch['alpha'].to(self.device)

            # Generate stylized image
            output = self.model(content, style, alpha=alpha.mean().item())

            # Compute loss
            if isinstance(self.criterion, MultiScaleLoss):
                loss = self.criterion(output, content)
                loss_components = {'total': loss}
            else:
                loss, loss_components = self.criterion(
                    output, content, style, return_components=True
                )

            # Accumulate losses
            for key in val_losses.keys():
                if key in loss_components:
                    val_losses[key] += loss_components[key].item()

            # Save first batch samples
            if batch_idx == 0:
                self.save_samples(content[:4], style[:4], output[:4], prefix='val')

        # Average losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)

        return val_losses

    def save_samples(self, content, style, output, prefix='train'):
        """Save sample images"""
        import torchvision.utils as vutils

        # Denormalize if needed
        content = content.clamp(0, 1)
        style = style.clamp(0, 1)
        output = output.clamp(0, 1)

        # Create grid
        samples = torch.cat([content, style, output], dim=0)
        grid = vutils.make_grid(samples, nrow=len(content), padding=2, normalize=False)

        # Save
        save_path = self.sample_dir / f'{prefix}_epoch{self.epoch:04d}_step{self.global_step:07d}.png'
        vutils.save_image(grid, save_path)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_image(f'{prefix}/samples', grid, self.global_step)

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{self.epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_loss: {self.best_val_loss:.4f}")

        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch*.pth'))
        if len(checkpoints) > self.config['keep_checkpoints']:
            for old_checkpoint in checkpoints[:-self.config['keep_checkpoints']]:
                old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Total training samples: {len(train_loader.dataset)}")
        print(f"Total validation samples: {len(val_loader.dataset)}")

        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader)
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['total']:.4f}")

            # Validate
            val_losses = self.validate(val_loader)
            print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}")

            # Log to TensorBoard
            if self.writer:
                for key in train_losses:
                    self.writer.add_scalar(f'epoch/train_{key}', train_losses[key], epoch)
                    self.writer.add_scalar(f'epoch/val_{key}', val_losses[key], epoch)

            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            if (epoch + 1) % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

        print("\n✓ Training completed!")

        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Style Transfer Model')
    parser.add_argument('--content_dir', type=str, required=True, help='Content images directory')
    parser.add_argument('--style_dir', type=str, required=True, help='Style images directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Training image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Configuration
    config = {
        # Data
        'content_dir': args.content_dir,
        'style_dir': args.style_dir,
        'output_dir': args.output_dir,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,

        # Training
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'lr_patience': 5,

        # Model
        'use_attention': True,

        # Losses
        'content_weight': 1.0,
        'style_weight': 100.0,
        'perceptual_weight': 0.5,
        'lpips_weight': 0.5,
        'tv_weight': 1e-4,
        'use_lpips': True,
        'use_multiscale': True,
        'multiscale_scales': [1.0, 0.5],
        'multiscale_weights': [0.7, 0.3],

        # Training options
        'use_amp': True,
        'use_tensorboard': True,

        # Logging
        'log_interval': 100,
        'sample_interval': 500,
        'save_interval': 5,
        'keep_checkpoints': 5,
    }

    # Create datasets
    print("Creating datasets...")
    train_dataset = AdvancedStyleTransferDataset(
        content_dir=config['content_dir'],
        style_dir=config['style_dir'],
        image_size=config['image_size'],
        mode='train',
        augment=True
    )

    val_dataset = AdvancedStyleTransferDataset(
        content_dir=config['content_dir'],
        style_dir=config['style_dir'],
        image_size=config['image_size'],
        mode='val',
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
