import numpy as np
import torch
import cv2
from PIL import Image

class SatelliteEnhancer:
    """
    Advanced Satellite Image Post-Processing Enhancer.
    Translates raw reflectance tensors [0, 1] into visually crisp, high-contrast,
    and edge-sharpened 8-bit images for display and analysis.
    """
    
    @staticmethod
    def percentile_stretch(img_np, p_min=1.0, p_max=99.0):
        """Linearly stretch dynamic range using 1%-99% percentiles of satellite reflectance."""
        valid_pixels = img_np[img_np > 0] if np.any(img_np > 0) else img_np
        if len(valid_pixels) == 0:
            return img_np
        low = np.percentile(valid_pixels, p_min)
        high = np.percentile(valid_pixels, p_max)
        if high <= low:
            return img_np
        stretched = np.clip((img_np - low) / (high - low + 1e-8), 0.0, 1.0)
        return stretched

    @staticmethod
    def apply_clahe(img_uint8, clip_limit=2.5, tile_grid_size=(8, 8)):
        """Contrast Limited Adaptive Histogram Equalization for local terrain contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img_uint8)

    @staticmethod
    def unsharp_mask(img_uint8, sigma=1.0, strength=1.5):
        """Guided Unsharp Masking for crisp structural edge recovery."""
        blurred = cv2.GaussianBlur(img_uint8, (0, 0), sigma)
        sharpened = cv2.addWeighted(img_uint8, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @classmethod
    def enhance(cls, input_data, stretch=True, clahe=True, sharpen=True, p_min=1.0, p_max=99.0, sharp_strength=1.5):
        """
        Main enhancement function. Accepts torch Tensor (C, H, W) or numpy array (H, W).
        Returns a PIL Image and uint8 numpy array.
        """
        if isinstance(input_data, torch.Tensor):
            img_np = input_data.squeeze().detach().cpu().numpy()
        else:
            img_np = np.squeeze(input_data)
            
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)

        # 1. Percentile Dynamic Range Stretch
        if stretch:
            img_np = cls.percentile_stretch(img_np, p_min=p_min, p_max=p_max)

        # Convert to 8-bit scale [0, 255]
        img_uint8 = (img_np * 255.0).astype(np.uint8)

        # 2. CLAHE Adaptive Contrast
        if clahe:
            img_uint8 = cls.apply_clahe(img_uint8, clip_limit=2.0, tile_grid_size=(8, 8))

        # 3. Unsharp Masking Edge Enhancement
        if sharpen:
            img_uint8 = cls.unsharp_mask(img_uint8, sigma=1.0, strength=sharp_strength)

        pil_img = Image.fromarray(img_uint8)
        return pil_img, img_uint8

    @classmethod
    def enhance_color(cls, pil_rgb, stretch=True, clahe=True, sharpen=True, sharp_strength=1.2):
        """
        Enhance 3-channel RGB image: percentile stretch, LAB luminance CLAHE, and unsharp masking.
        Preserves natural hues without causing color distortion.
        """
        np_rgb = np.array(pil_rgb.convert('RGB'))

        if stretch:
            valid = np_rgb[np_rgb > 0] if np.any(np_rgb > 0) else np_rgb
            if len(valid) > 0:
                low = np.percentile(valid, 1.0)
                high = np.percentile(valid, 99.0)
                if high > low:
                    np_rgb = np.clip((np_rgb.astype(np.float32) - low) / (high - low + 1e-8) * 255.0, 0, 255).astype(np.uint8)

        if clahe:
            lab = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe_op.apply(l)
            lab = cv2.merge((l, a, b))
            np_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if sharpen:
            blurred = cv2.GaussianBlur(np_rgb, (0, 0), 1.0)
            sharpened = cv2.addWeighted(np_rgb, 1.0 + sharp_strength, blurred, -sharp_strength, 0)
            np_rgb = np.clip(sharpened, 0, 255).astype(np.uint8)

        return Image.fromarray(np_rgb), np_rgb

if __name__ == '__main__':
    print("SatelliteEnhancer module ready.")
