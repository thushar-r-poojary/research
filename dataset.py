import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np

class ProbavDataset(Dataset):
    """
    Dataset loader for the official PROBA-V Super-Resolution dataset.
    Supports RED and NIR bands, multi-frame stacking, quality map filtering,
    16-bit normalization, and patch cropping.
    """
    def __init__(self, root_dir='data/probav', split='train', band='RED', num_frames=9, is_train=True, patch_size=64, scale=3):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.band = band
        self.num_frames = num_frames
        self.is_train = is_train
        self.patch_size = patch_size
        self.scale = scale

        # Path to scene directories
        # Kelvins structure typically has data/probav/train/RED/imgsetXXXX/
        self.band_dir = os.path.join(root_dir, split, band)
        if not os.path.exists(self.band_dir):
            # Fallback if structure is slightly different (e.g. data/probav/RED/train)
            self.band_dir = os.path.join(root_dir, band, split)
        
        if os.path.exists(self.band_dir):
            self.scenes = sorted([d for d in os.listdir(self.band_dir) if d.startswith('imgset')])
        else:
            self.scenes = []
            print(f"Warning: PROBA-V directory {self.band_dir} not found. Please run download_probav.py first.")

    def __len__(self):
        return len(self.scenes)

    def _load_image(self, path):
        img = Image.open(path)
        img_np = np.array(img, dtype=np.float32)
        # Normalize: if max is > 255, it's 16-bit (divide by 65535), otherwise 8-bit (divide by 255)
        if img_np.max() > 255.0:
            img_np /= 65535.0
        else:
            img_np /= 255.0
        return torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)

    def _load_qm(self, path):
        img = Image.open(path)
        img_np = np.array(img, dtype=np.float32)
        # In PROBA-V, QM has 1 for clear pixels and 0 for bad pixels.
        qm = (img_np > 0.5).astype(np.float32)
        return torch.from_numpy(qm).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        scene_dir = os.path.join(self.band_dir, scene)

        # List all LR files and corresponding QMs
        all_files = os.listdir(scene_dir)
        lr_names = sorted([f for f in all_files if f.startswith('lr') and f.endswith('.png')])
        
        # Sort/select best frames using quality maps (highest clear pixel count)
        lr_scores = []
        for lr_name in lr_names:
            qm_name = 'qm_' + lr_name
            qm_path = os.path.join(scene_dir, qm_name)
            if os.path.exists(qm_path):
                try:
                    qm_img = np.array(Image.open(qm_path))
                    clear_pixels = np.sum(qm_img > 0)
                except Exception:
                    clear_pixels = 0
                lr_scores.append((lr_name, clear_pixels))
            else:
                lr_scores.append((lr_name, 1)) # dummy score

        # Sort by clear pixels descending
        lr_scores.sort(key=lambda x: x[1], reverse=True)
        best_lr_names = [x[0] for x in lr_scores]

        # Select top num_frames
        selected_lr_names = best_lr_names[:self.num_frames]
        # Pad if there are fewer frames than requested
        while len(selected_lr_names) < self.num_frames:
            if len(selected_lr_names) > 0:
                selected_lr_names.append(selected_lr_names[-1])
            else:
                raise RuntimeError(f"No LR frames found in scene {scene} at {scene_dir}")

        # Load selected LR images and their QMs
        lr_list = []
        qm_list = []
        for lr_name in selected_lr_names:
            lr_path = os.path.join(scene_dir, lr_name)
            qm_path = os.path.join(scene_dir, 'qm_' + lr_name)
            
            lr_list.append(self._load_image(lr_path))
            if os.path.exists(qm_path):
                qm_list.append(self._load_qm(qm_path))
            else:
                qm_list.append(torch.ones_like(lr_list[-1]))

        # Stack to (num_frames, H, W)
        lr_seq = torch.cat(lr_list, dim=0)
        qm_seq = torch.cat(qm_list, dim=0)

        # Load HR and its QM
        hr_path = os.path.join(scene_dir, 'hr.png')
        qm_hr_path = os.path.join(scene_dir, 'qm_hr.png')

        if os.path.exists(hr_path):
            hr = self._load_image(hr_path)
            if os.path.exists(qm_hr_path):
                qm_hr = self._load_qm(qm_hr_path)
            else:
                qm_hr = torch.ones_like(hr)
        else:
            # Test split doesn't have HR target, return zero tensors
            hr = torch.zeros(1, lr_seq.shape[1] * self.scale, lr_seq.shape[2] * self.scale)
            qm_hr = torch.zeros_like(hr)

        # Apply patch cropping during training
        if self.is_train and self.split == 'train':
            c, h, w = lr_seq.shape
            patch_lr = self.patch_size
            patch_hr = self.patch_size * self.scale
            
            if h > patch_lr and w > patch_lr:
                x_lr = random.randint(0, w - patch_lr)
                y_lr = random.randint(0, h - patch_lr)
                x_hr = x_lr * self.scale
                y_hr = y_lr * self.scale
                
                lr_seq = lr_seq[:, y_lr:y_lr+patch_lr, x_lr:x_lr+patch_lr]
                qm_seq = qm_seq[:, y_lr:y_lr+patch_lr, x_lr:x_lr+patch_lr]
                hr = hr[:, y_hr:y_hr+patch_hr, x_hr:x_hr+patch_hr]
                qm_hr = qm_hr[:, y_hr:y_hr+patch_hr, x_hr:x_hr+patch_hr]

        return {
            'lr_seq': lr_seq,      # (num_frames, H, W)
            'qm_seq': qm_seq,      # (num_frames, H, W)
            'lr1': lr_seq[0:1],    # Best frame (1, H, W) for dual-image
            'lr2': lr_seq[1:2] if self.num_frames > 1 else lr_seq[0:1], # Second-best frame (1, H, W)
            'hr': hr,              # (1, HR_H, HR_W)
            'qm_hr': qm_hr,        # (1, HR_H, HR_W)
            'filename': scene
        }

if __name__ == "__main__":
    print("PROBA-V dataset loader module ready.")
