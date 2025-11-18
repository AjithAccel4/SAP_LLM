"""
Data Augmentation Pipeline for SAP Document Training.

Applies realistic augmentations to increase model robustness:
- Image-level: rotation, brightness, noise, JPEG compression
- Document-level: simulate photocopy, fax, watermarks
- Text-level: OCR errors, character swaps

Follows PLAN_02.md Phase 2 augmentation strategy.
"""

import os
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install with: pip install pillow")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.info("albumentations not available (optional). Install with: pip install albumentations")


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Image-level augmentations
    random_rotation_degrees: float = 2.0
    random_brightness_factor: float = 0.3
    gaussian_noise_sigma: float = 5.0
    jpeg_compression_quality: Tuple[int, int] = (75, 95)

    # Document-level augmentations
    simulate_photocopy_prob: float = 0.1
    simulate_fax_prob: float = 0.05
    add_watermark_prob: float = 0.15

    # Text-level augmentations
    ocr_error_rate: float = 0.02
    swap_similar_chars_prob: float = 0.1

    # Augmentation probability
    augmentation_probability: float = 0.8


class DataAugmentor:
    """
    Document data augmentation pipeline.

    Applies various augmentations to increase training data diversity
    and model robustness to real-world document variations.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize data augmentor.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

        # Character swap mappings (commonly confused in OCR)
        self.char_swaps = {
            'O': '0', '0': 'O',
            'l': '1', '1': 'l',
            'I': '1',
            'S': '5', '5': 'S',
            'B': '8', '8': 'B',
            'Z': '2',
            'G': '6',
        }

        # Statistics
        self.stats = {
            "total_augmented": 0,
            "by_augmentation_type": {},
            "errors": 0
        }

        logger.info("DataAugmentor initialized")

    def augment_image(self,
                     image_path: str,
                     output_path: str,
                     augmentation_level: str = "medium") -> bool:
        """
        Apply augmentations to document image.

        Args:
            image_path: Input image path
            output_path: Output image path
            augmentation_level: Augmentation intensity (low, medium, high)

        Returns:
            Success status
        """
        if not PIL_AVAILABLE:
            logger.error("PIL is required for image augmentation")
            return False

        try:
            # Load image
            image = Image.open(image_path)

            # Decide whether to augment (based on probability)
            if random.random() > self.config.augmentation_probability:
                # No augmentation - just copy
                image.save(output_path)
                return True

            # Apply augmentations
            augmented = image.copy()

            # Image-level augmentations
            if random.random() < 0.7:
                augmented = self._apply_rotation(augmented)

            if random.random() < 0.6:
                augmented = self._apply_brightness(augmented)

            if random.random() < 0.5:
                augmented = self._apply_noise(augmented)

            if random.random() < 0.4:
                augmented = self._apply_jpeg_compression(augmented)

            # Document-level augmentations
            if random.random() < self.config.simulate_photocopy_prob:
                augmented = self._simulate_photocopy(augmented)
                self.stats["by_augmentation_type"]["photocopy"] = \
                    self.stats["by_augmentation_type"].get("photocopy", 0) + 1

            if random.random() < self.config.simulate_fax_prob:
                augmented = self._simulate_fax(augmented)
                self.stats["by_augmentation_type"]["fax"] = \
                    self.stats["by_augmentation_type"].get("fax", 0) + 1

            if random.random() < self.config.add_watermark_prob:
                augmented = self._add_watermark(augmented)
                self.stats["by_augmentation_type"]["watermark"] = \
                    self.stats["by_augmentation_type"].get("watermark", 0) + 1

            # Save augmented image
            augmented.save(output_path)

            self.stats["total_augmented"] += 1

            return True

        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {e}")
            self.stats["errors"] += 1
            return False

    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation."""
        angle = random.uniform(
            -self.config.random_rotation_degrees,
            self.config.random_rotation_degrees
        )
        return image.rotate(angle, fillcolor='white', expand=False)

    def _apply_brightness(self, image: Image.Image) -> Image.Image:
        """Apply random brightness adjustment."""
        factor = random.uniform(
            1.0 - self.config.random_brightness_factor,
            1.0 + self.config.random_brightness_factor
        )
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise."""
        # Convert to numpy
        img_array = np.array(image)

        # Generate Gaussian noise
        noise = np.random.normal(0, self.config.gaussian_noise_sigma, img_array.shape)

        # Add noise
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)

    def _apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        import io

        # Random quality
        quality = random.randint(*self.config.jpeg_compression_quality)

        # Save to bytes with compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Reload
        return Image.open(buffer)

    def _simulate_photocopy(self, image: Image.Image) -> Image.Image:
        """
        Simulate photocopied document appearance.

        Effects:
        - Increased contrast
        - Slight blur
        - Random speckles
        """
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        photocopied = enhancer.enhance(1.3)

        # Slight blur
        photocopied = photocopied.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Add speckles (noise)
        img_array = np.array(photocopied)

        # Random speckles
        speckle_density = 0.001
        mask = np.random.random(img_array.shape[:2]) < speckle_density

        if len(img_array.shape) == 3:
            # RGB image
            for c in range(img_array.shape[2]):
                channel = img_array[:, :, c]
                channel[mask] = np.random.choice([0, 255], size=mask.sum())
                img_array[:, :, c] = channel
        else:
            # Grayscale
            img_array[mask] = np.random.choice([0, 255], size=mask.sum())

        return Image.fromarray(img_array)

    def _simulate_fax(self, image: Image.Image) -> Image.Image:
        """
        Simulate faxed document appearance.

        Effects:
        - Convert to grayscale
        - Lower resolution
        - Horizontal scan lines
        - High contrast
        """
        # Convert to grayscale
        faxed = image.convert('L')

        # Resize down and up (lower resolution)
        original_size = faxed.size
        faxed = faxed.resize(
            (original_size[0] // 2, original_size[1] // 2),
            Image.Resampling.BILINEAR
        )
        faxed = faxed.resize(original_size, Image.Resampling.NEAREST)

        # Increase contrast
        enhancer = ImageEnhance.Contrast(faxed)
        faxed = enhancer.enhance(1.5)

        # Add horizontal scan lines
        img_array = np.array(faxed)

        # Every 3rd row, add slight darkening
        img_array[::3, :] = np.clip(img_array[::3, :] * 0.95, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def _add_watermark(self, image: Image.Image) -> Image.Image:
        """
        Add semi-transparent watermark.

        Simulates:
        - Company stamps
        - "COPY" watermarks
        - Date stamps
        """
        # Create watermark layer
        watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)

        # Random watermark text
        watermark_texts = [
            "COPY",
            "CONFIDENTIAL",
            "DRAFT",
            "INTERNAL USE ONLY"
        ]
        text = random.choice(watermark_texts)

        # Try to use a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
        except:
            font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position (center or diagonal)
        if random.random() < 0.5:
            # Center
            x = (image.width - text_width) // 2
            y = (image.height - text_height) // 2
        else:
            # Diagonal
            x = image.width // 4
            y = image.height // 3

        # Draw with low opacity
        draw.text(
            (x, y),
            text,
            fill=(128, 128, 128, 50),  # Low alpha
            font=font
        )

        # Composite watermark onto image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        watermarked = Image.alpha_composite(image, watermark)

        # Convert back to RGB
        return watermarked.convert('RGB')

    def augment_text(self, text: str) -> str:
        """
        Apply text-level augmentations (simulate OCR errors).

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        if random.random() > self.config.augmentation_probability:
            return text

        augmented = list(text)

        # Introduce random OCR errors
        for i in range(len(augmented)):
            # Random character swap
            if random.random() < self.config.swap_similar_chars_prob:
                char = augmented[i]
                if char in self.char_swaps:
                    augmented[i] = self.char_swaps[char]
                    self.stats["by_augmentation_type"]["char_swap"] = \
                        self.stats["by_augmentation_type"].get("char_swap", 0) + 1

            # Random deletion
            elif random.random() < self.config.ocr_error_rate:
                augmented[i] = ''
                self.stats["by_augmentation_type"]["char_deletion"] = \
                    self.stats["by_augmentation_type"].get("char_deletion", 0) + 1

        return ''.join(augmented)

    def augment_batch(self,
                     image_paths: List[str],
                     output_dir: str,
                     num_augmentations_per_image: int = 1) -> List[str]:
        """
        Augment a batch of images.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory for augmented images
            num_augmentations_per_image: Number of augmented versions per image

        Returns:
            List of augmented image paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        augmented_paths = []

        for img_path in image_paths:
            for aug_idx in range(num_augmentations_per_image):
                # Generate output path
                input_path = Path(img_path)
                output_file = output_path / f"{input_path.stem}_aug{aug_idx}{input_path.suffix}"

                # Augment
                success = self.augment_image(
                    image_path=str(img_path),
                    output_path=str(output_file)
                )

                if success:
                    augmented_paths.append(str(output_file))

        logger.info(f"Augmented {len(image_paths)} images -> {len(augmented_paths)} outputs")

        return augmented_paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        return self.stats.copy()


# CLI entrypoint
def main():
    """CLI for data augmentation."""
    import argparse

    parser = argparse.ArgumentParser(description="Augment Document Images")
    parser.add_argument("--input-dir", required=True, help="Input directory with images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--augmentations-per-image", type=int, default=2, help="Augmentations per image")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create augmentor
    augmentor = DataAugmentor()

    # Find all images
    input_path = Path(args.input_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_files)} images")

    # Augment
    augmented = augmentor.augment_batch(
        image_paths=[str(p) for p in image_files],
        output_dir=args.output_dir,
        num_augmentations_per_image=args.augmentations_per_image
    )

    print(f"\n{'=' * 80}")
    print(f"Augmentation Complete!")
    print(f"Input Images: {len(image_files)}")
    print(f"Augmented Images: {len(augmented)}")
    print(f"Statistics: {augmentor.get_statistics()}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
