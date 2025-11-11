"""
Advanced Loss Functions for Style Transfer

Includes:
- Perceptual Loss (VGG-based)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Total Variation Loss
- Content Loss
- Style Loss (Gram Matrix)
- Multi-scale losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    Measures content similarity in feature space rather than pixel space
    """
    def __init__(self, feature_layers=[1, 6, 11, 20, 29], weights=None):
        """
        Args:
            feature_layers: VGG19 layer indices to extract features from
            weights: Weight for each layer (default: equal weights)
        """
        super(VGGPerceptualLoss, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_layers = feature_layers

        # Split VGG into blocks for feature extraction
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev_layer:layer_idx+1]))
            prev_layer = layer_idx + 1

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Layer weights
        if weights is None:
            self.weights = [1.0 / len(feature_layers)] * len(feature_layers)
        else:
            assert len(weights) == len(feature_layers), "Weights length must match feature_layers"
            self.weights = weights

        # Normalization for VGG input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        """Normalize input for VGG"""
        # Assuming x is in [0, 1] or [-1, 1]
        if x.min() < 0:
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        Returns:
            Perceptual loss (scalar)
        """
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)

        loss = 0
        x = pred
        y = target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            # L2 loss in feature space
            layer_loss = F.mse_loss(x, y)
            loss += self.weights[i] * layer_loss

        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices
    Captures texture and style information
    """
    def __init__(self):
        super(StyleLoss, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Use standard style layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21]) # relu4_1
        self.slice5 = nn.Sequential(*list(vgg.children())[21:30]) # relu5_1

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Layer weights (deeper layers = more weight)
        self.layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std

    def gram_matrix(self, x):
        """Compute Gram matrix"""
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(C * H * W)

    def extract_features(self, x):
        """Extract features from all style layers"""
        x = self.normalize_input(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Style target [B, 3, H, W]
        Returns:
            Style loss (scalar)
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0
        for i, (pred_feat, target_feat, weight) in enumerate(
            zip(pred_features, target_features, self.layer_weights)
        ):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += weight * F.mse_loss(pred_gram, target_gram)

        return loss


class ContentLoss(nn.Module):
    """
    Content loss using VGG relu4_1 features
    Preserves high-level content structure
    """
    def __init__(self):
        super(ContentLoss, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Use relu4_1 for content (layer 20)
        self.features = nn.Sequential(*list(vgg.children())[:21])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Content target [B, 3, H, W]
        Returns:
            Content loss (scalar)
        """
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)

        pred_features = self.features(pred)
        target_features = self.features(target)

        return F.mse_loss(pred_features, target_features)


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness
    Reduces high-frequency noise and artifacts
    """
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, C, H, W]
        Returns:
            TV loss (scalar)
        """
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)

        # Horizontal and vertical differences
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()

        return self.weight * (h_tv / count_h + w_tv / count_w) / batch_size


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity
    Better correlates with human perception than L2 or VGG losses

    Simplified version using VGG features with learned weights
    """
    def __init__(self):
        super(LPIPS, self).__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Extract features from multiple layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])
        self.slice5 = nn.Sequential(*list(vgg.children())[27:36])

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Learnable channel weights for each layer
        self.chns = [64, 128, 256, 512, 512]
        self.lin0 = nn.Conv2d(self.chns[0], 1, 1, bias=False)
        self.lin1 = nn.Conv2d(self.chns[1], 1, 1, bias=False)
        self.lin2 = nn.Conv2d(self.chns[2], 1, 1, bias=False)
        self.lin3 = nn.Conv2d(self.chns[3], 1, 1, bias=False)
        self.lin4 = nn.Conv2d(self.chns[4], 1, 1, bias=False)

        # Initialize weights
        for m in [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]:
            nn.init.constant_(m.weight, 1.0)

        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x):
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std

    def normalize_tensor(self, x, eps=1e-10):
        """Normalize features for comparison"""
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def spatial_average(self, x):
        """Average across spatial dimensions"""
        return x.mean([2, 3], keepdim=True)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        Returns:
            LPIPS distance (scalar)
        """
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)

        # Extract features
        pred_feats = []
        target_feats = []

        h = pred
        t = target

        h = self.slice1(h)
        t = self.slice1(t)
        pred_feats.append(h)
        target_feats.append(t)

        h = self.slice2(h)
        t = self.slice2(t)
        pred_feats.append(h)
        target_feats.append(t)

        h = self.slice3(h)
        t = self.slice3(t)
        pred_feats.append(h)
        target_feats.append(t)

        h = self.slice4(h)
        t = self.slice4(t)
        pred_feats.append(h)
        target_feats.append(t)

        h = self.slice5(h)
        t = self.slice5(t)
        pred_feats.append(h)
        target_feats.append(t)

        # Compute differences
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        loss = 0

        for i in range(len(pred_feats)):
            # Normalize features
            pred_feat = self.normalize_tensor(pred_feats[i])
            target_feat = self.normalize_tensor(target_feats[i])

            # Compute difference
            diff = (pred_feat - target_feat) ** 2

            # Apply learned weights
            diff = lins[i](diff)

            # Spatial average
            diff = self.spatial_average(diff)

            loss += diff.mean()

        return loss


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for better detail preservation at different resolutions
    """
    def __init__(self, loss_fn, scales=[1.0, 0.5, 0.25], weights=None):
        """
        Args:
            loss_fn: Base loss function to apply at each scale
            scales: List of scales to process
            weights: Weight for each scale (default: equal)
        """
        super(MultiScaleLoss, self).__init__()
        self.loss_fn = loss_fn
        self.scales = scales

        if weights is None:
            self.weights = [1.0 / len(scales)] * len(scales)
        else:
            assert len(weights) == len(scales)
            self.weights = weights

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        Returns:
            Multi-scale loss (scalar)
        """
        total_loss = 0
        H, W = pred.size(2), pred.size(3)

        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                h, w = int(H * scale), int(W * scale)
                pred_scaled = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target

            scale_loss = self.loss_fn(pred_scaled, target_scaled)
            total_loss += weight * scale_loss

        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for style transfer training

    Combines:
    - Content loss (VGG)
    - Style loss (Gram matrix)
    - Perceptual loss (VGG features)
    - LPIPS (learned perceptual)
    - Total variation (smoothness)
    """
    def __init__(
        self,
        content_weight=1.0,
        style_weight=100.0,
        perceptual_weight=0.5,
        lpips_weight=0.5,
        tv_weight=1e-4,
        use_lpips=True
    ):
        super(CombinedLoss, self).__init__()

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        self.tv_weight = tv_weight

        # Loss modules
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.tv_loss = TotalVariationLoss()

        self.use_lpips = use_lpips
        if use_lpips:
            self.lpips_loss = LPIPS()

    def forward(self, pred, content, style, return_components=False):
        """
        Args:
            pred: Generated image [B, 3, H, W]
            content: Content target [B, 3, H, W]
            style: Style target [B, 3, H, W]
            return_components: If True, return individual loss components
        Returns:
            Total loss (and optionally dict of components)
        """
        losses = {}

        # Content loss
        if self.content_weight > 0:
            losses['content'] = self.content_weight * self.content_loss(pred, content)
        else:
            losses['content'] = torch.tensor(0.0, device=pred.device)

        # Style loss
        if self.style_weight > 0:
            losses['style'] = self.style_weight * self.style_loss(pred, style)
        else:
            losses['style'] = torch.tensor(0.0, device=pred.device)

        # Perceptual loss
        if self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_weight * self.perceptual_loss(pred, content)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)

        # LPIPS loss
        if self.use_lpips and self.lpips_weight > 0:
            losses['lpips'] = self.lpips_weight * self.lpips_loss(pred, content)
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)

        # Total variation loss
        if self.tv_weight > 0:
            losses['tv'] = self.tv_weight * self.tv_loss(pred)
        else:
            losses['tv'] = torch.tensor(0.0, device=pred.device)

        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        if return_components:
            return total_loss, losses
        return total_loss


# Test code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dummy images
    pred = torch.randn(2, 3, 256, 256).to(device)
    content = torch.randn(2, 3, 256, 256).to(device)
    style = torch.randn(2, 3, 256, 256).to(device)

    print("Testing loss functions...")

    # Test individual losses
    content_loss = ContentLoss().to(device)
    print(f"Content Loss: {content_loss(pred, content).item():.4f}")

    style_loss = StyleLoss().to(device)
    print(f"Style Loss: {style_loss(pred, style).item():.4f}")

    perceptual_loss = VGGPerceptualLoss().to(device)
    print(f"Perceptual Loss: {perceptual_loss(pred, content).item():.4f}")

    tv_loss = TotalVariationLoss().to(device)
    print(f"TV Loss: {tv_loss(pred).item():.6f}")

    lpips = LPIPS().to(device)
    print(f"LPIPS: {lpips(pred, content).item():.4f}")

    # Test combined loss
    combined = CombinedLoss().to(device)
    total_loss, components = combined(pred, content, style, return_components=True)
    print(f"\nCombined Loss: {total_loss.item():.4f}")
    print("Components:")
    for name, value in components.items():
        if name != 'total':
            print(f"  {name}: {value.item():.6f}")

    # Test multi-scale
    ms_perceptual = MultiScaleLoss(perceptual_loss, scales=[1.0, 0.5]).to(device)
    print(f"\nMulti-scale Perceptual: {ms_perceptual(pred, content).item():.4f}")

    print("\n✓ All loss tests passed!")
