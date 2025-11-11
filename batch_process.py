"""
Batch Processing Script for Style Transfer

Supports:
- Batch image processing
- Video style transfer
- Multiple styles
- Progress tracking
- GPU optimization
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import time
from datetime import datetime

# Import models
from models.adain_model import AdaINStyleTransfer
from torchvision import transforms
import cv2
import numpy as np


class BatchStyleTransfer:
    """
    Batch processing for style transfer
    """
    def __init__(self, method='adain', device='auto', checkpoint=None):
        """
        Args:
            method: Transfer method ('adain', 'cnn', 'vit')
            device: Device to use ('cuda', 'cpu', 'auto')
            checkpoint: Path to model checkpoint
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.method = method
        self.model = None
        self._load_model(checkpoint)

    def _load_model(self, checkpoint=None):
        """Load model"""
        print(f"Loading {self.method} model...")

        if self.method == 'adain':
            self.model = AdaINStyleTransfer(use_attention=True).to(self.device)

            if checkpoint:
                checkpoint_data = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                print(f"✓ Loaded checkpoint from {checkpoint}")
            else:
                print("⚠ Using untrained model (encoder is pre-trained)")

            self.model.eval()
        else:
            raise NotImplementedError(f"Method {self.method} not yet implemented for batch processing")

    def preprocess(self, image, max_size=1024):
        """Preprocess image"""
        # Resize if needed
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to tensor
        transform = transforms.ToTensor()
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def postprocess(self, tensor):
        """Convert tensor back to PIL image"""
        tensor = tensor.cpu().squeeze(0).clamp(0, 1)
        tensor = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray((tensor * 255).astype(np.uint8))
        return image

    def transfer_single(self, content_path, style_path, output_path, alpha=1.0, preserve_color=False):
        """
        Transfer style for a single image

        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Path to save result
            alpha: Style strength
            preserve_color: Preserve content colors
        """
        # Load images
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        original_size = content_img.size

        # Preprocess
        content_tensor = self.preprocess(content_img)
        style_tensor = self.preprocess(style_img)

        # Transfer
        with torch.no_grad():
            output = self.model(
                content_tensor,
                style_tensor,
                alpha=alpha,
                preserve_color=preserve_color
            )

        # Postprocess
        result = self.postprocess(output)

        # Resize back to original
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)

        # Save
        result.save(output_path, quality=95, optimize=True)

        return result

    def transfer_batch(self, content_dir, style_path, output_dir, **kwargs):
        """
        Transfer style for a batch of images

        Args:
            content_dir: Directory containing content images
            style_path: Path to style image
            output_dir: Directory to save results
            **kwargs: Additional arguments for transfer_single
        """
        content_dir = Path(content_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        content_images = []
        for ext in image_extensions:
            content_images.extend(list(content_dir.glob(f'*{ext}')))
            content_images.extend(list(content_dir.glob(f'*{ext.upper()}')))

        print(f"Found {len(content_images)} images to process")

        # Process each image
        results = []
        start_time = time.time()

        for content_path in tqdm(content_images, desc="Processing images"):
            try:
                output_filename = f"stylized_{content_path.stem}.jpg"
                output_path = output_dir / output_filename

                self.transfer_single(
                    content_path,
                    style_path,
                    output_path,
                    **kwargs
                )

                results.append({
                    'content': str(content_path),
                    'output': str(output_path),
                    'success': True
                })

            except Exception as e:
                print(f"Error processing {content_path}: {e}")
                results.append({
                    'content': str(content_path),
                    'error': str(e),
                    'success': False
                })

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(content_images) if content_images else 0

        # Save report
        report = {
            'total_images': len(content_images),
            'successful': sum(r['success'] for r in results),
            'failed': sum(not r['success'] for r in results),
            'elapsed_time': elapsed_time,
            'avg_time_per_image': avg_time,
            'results': results
        }

        report_path = output_dir / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Batch processing completed!")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Average time per image: {avg_time:.2f}s")
        print(f"Successful: {report['successful']}/{report['total_images']}")
        print(f"Report saved to: {report_path}")

        return report


class VideoStyleTransfer:
    """
    Video style transfer
    """
    def __init__(self, method='adain', device='auto', checkpoint=None):
        """
        Args:
            method: Transfer method
            device: Device to use
            checkpoint: Path to model checkpoint
        """
        self.batch_processor = BatchStyleTransfer(method, device, checkpoint)

    def transfer_video(
        self,
        video_path,
        style_path,
        output_path,
        alpha=1.0,
        preserve_color=False,
        fps=None,
        max_frames=None
    ):
        """
        Transfer style for video

        Args:
            video_path: Path to input video
            style_path: Path to style image
            output_path: Path to save output video
            alpha: Style strength
            fps: Output FPS (default: same as input)
            max_frames: Maximum frames to process (None = all)
        """
        print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_fps = fps if fps is not None else input_fps
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

        print(f"Video info: {width}x{height} @ {input_fps:.2f} FPS, {total_frames} frames")
        print(f"Processing {frames_to_process} frames @ {output_fps:.2f} FPS")

        # Load style image
        style_img = Image.open(style_path).convert('RGB')
        style_tensor = self.batch_processor.preprocess(style_img)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

        # Process frames
        frame_idx = 0
        start_time = time.time()

        with tqdm(total=frames_to_process, desc="Processing video") as pbar:
            while cap.isOpened() and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                content_img = Image.fromarray(frame_rgb)

                # Preprocess
                content_tensor = self.batch_processor.preprocess(content_img)

                # Transfer
                with torch.no_grad():
                    output = self.batch_processor.model(
                        content_tensor,
                        style_tensor,
                        alpha=alpha,
                        preserve_color=preserve_color
                    )

                # Postprocess
                result = self.batch_processor.postprocess(output)

                # Resize to original size
                if result.size != (width, height):
                    result = result.resize((width, height), Image.Resampling.LANCZOS)

                # Convert back to OpenCV format
                result_array = np.array(result)
                result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(result_bgr)

                frame_idx += 1
                pbar.update(1)

                # Show estimated time remaining
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = frame_idx / elapsed
                    remaining = (frames_to_process - frame_idx) / fps_processing
                    pbar.set_postfix({
                        'FPS': f'{fps_processing:.2f}',
                        'ETA': f'{remaining:.0f}s'
                    })

        # Cleanup
        cap.release()
        out.release()

        elapsed_time = time.time() - start_time
        print(f"\n✓ Video processing completed!")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Average FPS: {frames_to_process / elapsed_time:.2f}")
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch Style Transfer Processing')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Batch image processing
    batch_parser = subparsers.add_parser('batch', help='Batch process images')
    batch_parser.add_argument('--content_dir', type=str, required=True, help='Content images directory')
    batch_parser.add_argument('--style', type=str, required=True, help='Style image path')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    batch_parser.add_argument('--method', type=str, default='adain', help='Transfer method')
    batch_parser.add_argument('--alpha', type=float, default=1.0, help='Style strength')
    batch_parser.add_argument('--preserve_color', action='store_true', help='Preserve colors')
    batch_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    batch_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')

    # Video processing
    video_parser = subparsers.add_parser('video', help='Process video')
    video_parser.add_argument('--video', type=str, required=True, help='Input video path')
    video_parser.add_argument('--style', type=str, required=True, help='Style image path')
    video_parser.add_argument('--output', type=str, required=True, help='Output video path')
    video_parser.add_argument('--method', type=str, default='adain', help='Transfer method')
    video_parser.add_argument('--alpha', type=float, default=1.0, help='Style strength')
    video_parser.add_argument('--preserve_color', action='store_true', help='Preserve colors')
    video_parser.add_argument('--fps', type=float, help='Output FPS')
    video_parser.add_argument('--max_frames', type=int, help='Max frames to process')
    video_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    video_parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')

    args = parser.parse_args()

    if args.command == 'batch':
        processor = BatchStyleTransfer(
            method=args.method,
            device=args.device,
            checkpoint=args.checkpoint
        )

        processor.transfer_batch(
            content_dir=args.content_dir,
            style_path=args.style,
            output_dir=args.output_dir,
            alpha=args.alpha,
            preserve_color=args.preserve_color
        )

    elif args.command == 'video':
        processor = VideoStyleTransfer(
            method=args.method,
            device=args.device,
            checkpoint=args.checkpoint
        )

        processor.transfer_video(
            video_path=args.video,
            style_path=args.style,
            output_path=args.output,
            alpha=args.alpha,
            preserve_color=args.preserve_color,
            fps=args.fps,
            max_frames=args.max_frames
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
