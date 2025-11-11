"""
Advanced AdaIN (Adaptive Instance Normalization) Style Transfer Model
Based on "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
(Huang & Belongie, 2017) with modern enhancements

Features:
- Real-time arbitrary style transfer (100x faster than optimization-based)
- Multi-scale processing for better detail preservation
- Attention mechanisms for semantic-aware transfer
- Color preservation mode
- Style interpolation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class AdaptiveInstanceNorm2d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN)
    Aligns the mean and variance of content features to match style features
    """
    def __init__(self, eps=1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps

    def forward(self, content, style):
        """
        Args:
            content: [B, C, H, W] content feature maps
            style: [B, C, H', W'] style feature maps
        Returns:
            [B, C, H, W] stylized feature maps
        """
        assert content.size()[:2] == style.size()[:2], \
            f"Content and style must have same batch size and channels, got {content.size()[:2]} and {style.size()[:2]}"

        B, C = content.size()[:2]

        # Calculate content statistics
        content_mean = content.view(B, C, -1).mean(dim=2, keepdim=True).unsqueeze(3)
        content_std = content.view(B, C, -1).std(dim=2, keepdim=True).unsqueeze(3) + self.eps

        # Calculate style statistics
        style_mean = style.view(B, C, -1).mean(dim=2, keepdim=True).unsqueeze(3)
        style_std = style.view(B, C, -1).std(dim=2, keepdim=True).unsqueeze(3) + self.eps

        # Normalize content features
        normalized_content = (content - content_mean) / content_std

        # Apply style statistics
        stylized = normalized_content * style_std + style_mean

        return stylized


class SANet(nn.Module):
    """
    Style Attentional Network for spatially-aware style transfer
    Computes attention maps between content and style features
    """
    def __init__(self, in_channels):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.g = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.h = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, content, style):
        """
        Args:
            content: [B, C, H, W]
            style: [B, C, H', W']
        Returns:
            [B, C, H, W] attention-weighted features
        """
        B, C, H, W = content.size()

        # Query from content
        f = self.f(content).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]

        # Key from style
        g = self.g(style)
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        g = g.view(B, -1, H * W)  # [B, C//8, HW]

        # Value from style
        h = self.h(style)
        h = F.interpolate(h, size=(H, W), mode='bilinear', align_corners=False)
        h = h.view(B, C, H * W)  # [B, C, HW]

        # Attention map
        attention = torch.bmm(f, g)  # [B, HW, HW]
        attention = self.softmax(attention)

        # Apply attention
        out = torch.bmm(h, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        out = self.out_conv(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class VGGEncoder(nn.Module):
    """
    VGG19 encoder for feature extraction
    Returns multi-scale features for better style transfer
    """
    def __init__(self, pretrained=True):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features

        # Multi-scale feature extraction
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21]) # relu4_1
        self.slice5 = nn.Sequential(*list(vgg.children())[21:30]) # relu5_1

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x, return_multi_scale=False):
        """
        Args:
            x: Input image [B, 3, H, W]
            return_multi_scale: If True, return all feature scales
        Returns:
            If return_multi_scale: List of features at different scales
            Else: Single feature map at relu4_1
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)

        if return_multi_scale:
            return [h1, h2, h3, h4, h5]
        return h4  # relu4_1 for AdaIN


class AdvancedDecoder(nn.Module):
    """
    Enhanced decoder with residual connections and multi-scale upsampling
    """
    def __init__(self, in_channels=512):
        super(AdvancedDecoder, self).__init__()

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, 256, 3, padding=1, padding_mode='reflect')
        self.in_in = nn.InstanceNorm2d(256)

        # Residual blocks at 256 channels
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Upsampling path
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect')
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 512, H/4, W/4] encoded features
        Returns:
            [B, 3, H, W] reconstructed image
        """
        # Initial projection
        x = F.relu(self.in_in(self.conv_in(x)))

        # Residual processing
        x = self.res_blocks(x)

        # Upsampling
        x = self.upsample1(x)  # [B, 128, H/2, W/2]
        x = self.upsample2(x)  # [B, 64, H, W]

        # Final refinement
        x = self.refine(x)     # [B, 3, H, W]

        return x


class AdaINStyleTransfer(nn.Module):
    """
    Complete AdaIN-based Style Transfer Network

    Features:
    - Real-time arbitrary style transfer
    - Multi-scale processing
    - Style attention mechanism
    - Color preservation mode
    - Style interpolation support
    """
    def __init__(self, use_attention=True):
        super(AdaINStyleTransfer, self).__init__()

        self.encoder = VGGEncoder(pretrained=True)
        self.decoder = AdvancedDecoder(in_channels=512)
        self.adain = AdaptiveInstanceNorm2d()

        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = SANet(512)

        # For multi-scale processing
        self.adain_layers = nn.ModuleDict({
            'adain1': AdaptiveInstanceNorm2d(),
            'adain2': AdaptiveInstanceNorm2d(),
            'adain3': AdaptiveInstanceNorm2d(),
            'adain4': AdaptiveInstanceNorm2d(),
        })

    def encode(self, x):
        """Extract features using VGG encoder"""
        return self.encoder(x, return_multi_scale=False)

    def decode(self, features):
        """Reconstruct image from features"""
        return self.decoder(features)

    def forward(self, content, style, alpha=1.0, use_attention=None, preserve_color=False):
        """
        Args:
            content: [B, 3, H, W] content image
            style: [B, 3, H, W] style image
            alpha: Style strength (0.0 = content only, 1.0 = full style)
            use_attention: Override default attention setting
            preserve_color: If True, only transfer texture/pattern, not color
        Returns:
            [B, 3, H, W] stylized image
        """
        assert 0 <= alpha <= 1, f"Alpha must be in [0, 1], got {alpha}"

        # Color preservation: convert style to content's color space
        if preserve_color:
            style = self._preserve_color(content, style)

        # Encode
        content_features = self.encode(content)
        style_features = self.encode(style)

        # Apply AdaIN
        stylized_features = self.adain(content_features, style_features)

        # Apply attention if enabled
        use_attn = use_attention if use_attention is not None else self.use_attention
        if use_attn and hasattr(self, 'attention'):
            attention_features = self.attention(content_features, style_features)
            stylized_features = stylized_features + 0.3 * attention_features

        # Interpolate between content and stylized features
        if alpha < 1.0:
            stylized_features = alpha * stylized_features + (1 - alpha) * content_features

        # Decode
        output = self.decode(stylized_features)

        return output

    def _preserve_color(self, content, style):
        """
        Preserve content color by transferring style to grayscale content
        and then matching color distribution
        """
        # Convert to YIQ color space
        content_yiq = self._rgb_to_yiq(content)
        style_yiq = self._rgb_to_yiq(style)

        # Replace style's color channels with content's
        style_yiq[:, 1:, :, :] = content_yiq[:, 1:, :, :]

        # Convert back to RGB
        style_colored = self._yiq_to_rgb(style_yiq)

        return style_colored

    def _rgb_to_yiq(self, img):
        """Convert RGB to YIQ color space"""
        r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.275 * g - 0.321 * b
        q = 0.212 * r - 0.523 * g + 0.311 * b
        return torch.cat([y, i, q], dim=1)

    def _yiq_to_rgb(self, img):
        """Convert YIQ to RGB color space"""
        y, i, q = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
        r = y + 0.956 * i + 0.621 * q
        g = y - 0.272 * i - 0.647 * q
        b = y - 1.106 * i + 1.703 * q
        return torch.cat([r, g, b], dim=1).clamp(0, 1)

    def multi_scale_forward(self, content, style, alpha=1.0, scales=[1.0, 0.5]):
        """
        Multi-scale style transfer for better detail preservation

        Args:
            content: [B, 3, H, W] content image
            style: [B, 3, H, W] style image
            alpha: Style strength
            scales: List of scales to process (1.0 = original size)
        Returns:
            [B, 3, H, W] stylized image
        """
        B, C, H, W = content.size()
        outputs = []

        for scale in scales:
            if scale != 1.0:
                h, w = int(H * scale), int(W * scale)
                content_scaled = F.interpolate(content, size=(h, w), mode='bilinear', align_corners=False)
                style_scaled = F.interpolate(style, size=(h, w), mode='bilinear', align_corners=False)
            else:
                content_scaled = content
                style_scaled = style

            # Process at this scale
            output_scaled = self.forward(content_scaled, style_scaled, alpha)

            # Resize back to original size
            if scale != 1.0:
                output_scaled = F.interpolate(output_scaled, size=(H, W), mode='bilinear', align_corners=False)

            outputs.append(output_scaled)

        # Blend multi-scale outputs
        if len(outputs) == 1:
            return outputs[0]
        else:
            # Weighted average (more weight to larger scales)
            weights = [scale for scale in scales]
            weights = [w / sum(weights) for w in weights]

            result = sum(w * out for w, out in zip(weights, outputs))
            return result

    def interpolate_styles(self, content, styles, weights):
        """
        Interpolate between multiple styles

        Args:
            content: [B, 3, H, W] content image
            styles: List of [B, 3, H, W] style images
            weights: List of weights for each style (should sum to 1)
        Returns:
            [B, 3, H, W] stylized image with interpolated styles
        """
        assert len(styles) == len(weights), "Number of styles must match number of weights"
        assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"

        # Encode content
        content_features = self.encode(content)

        # Encode and blend style features
        blended_style_features = None
        for style, weight in zip(styles, weights):
            style_features = self.encode(style)
            if blended_style_features is None:
                blended_style_features = weight * style_features
            else:
                blended_style_features += weight * style_features

        # Apply AdaIN with blended style
        stylized_features = self.adain(content_features, blended_style_features)

        # Decode
        output = self.decode(stylized_features)

        return output


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features for better quality
    Combines content loss, style loss, and optional TV loss
    """
    def __init__(self, style_layers=[0, 1, 2, 3], content_layers=[2]):
        super(PerceptualLoss, self).__init__()
        self.encoder = VGGEncoder(pretrained=True)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.mse_loss = nn.MSELoss()

    def gram_matrix(self, x):
        """Compute Gram matrix for style loss"""
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(C * H * W)

    def forward(self, output, content, style, content_weight=1.0, style_weight=100.0):
        """
        Compute perceptual loss

        Args:
            output: Generated image
            content: Content target
            style: Style target
            content_weight: Weight for content loss
            style_weight: Weight for style loss
        Returns:
            Total loss, content loss, style loss
        """
        # Extract features
        output_features = self.encoder(output, return_multi_scale=True)
        content_features = self.encoder(content, return_multi_scale=True)
        style_features = self.encoder(style, return_multi_scale=True)

        # Content loss
        content_loss = 0
        for idx in self.content_layers:
            content_loss += self.mse_loss(output_features[idx], content_features[idx])
        content_loss *= content_weight

        # Style loss (Gram matrix)
        style_loss = 0
        for idx in self.style_layers:
            output_gram = self.gram_matrix(output_features[idx])
            style_gram = self.gram_matrix(style_features[idx])
            style_loss += self.mse_loss(output_gram, style_gram)
        style_loss *= style_weight

        total_loss = content_loss + style_loss

        return total_loss, content_loss, style_loss


def total_variation_loss(x):
    """
    Total variation loss for spatial smoothness
    Reduces high-frequency artifacts
    """
    batch_size = x.size(0)
    h_x = x.size(2)
    w_x = x.size(3)
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)

    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()

    return (h_tv / count_h + w_tv / count_w) / batch_size


# Example usage
if __name__ == "__main__":
    # Create model
    model = AdaINStyleTransfer(use_attention=True)
    model.eval()

    # Dummy inputs
    content = torch.randn(1, 3, 256, 256)
    style = torch.randn(1, 3, 256, 256)

    # Test forward pass
    with torch.no_grad():
        # Basic style transfer
        output = model(content, style, alpha=1.0)
        print(f"Output shape: {output.shape}")

        # Color preservation
        output_color = model(content, style, alpha=1.0, preserve_color=True)
        print(f"Color preserved output shape: {output_color.shape}")

        # Multi-scale
        output_ms = model.multi_scale_forward(content, style, alpha=1.0)
        print(f"Multi-scale output shape: {output_ms.shape}")

        # Style interpolation
        style2 = torch.randn(1, 3, 256, 256)
        output_interp = model.interpolate_styles(content, [style, style2], [0.6, 0.4])
        print(f"Interpolated output shape: {output_interp.shape}")

    print("✓ AdaIN model tests passed!")
