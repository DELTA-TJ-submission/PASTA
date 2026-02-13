import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pasta.model_utils import get_disk_mask
import albumentations as A
import openslide
import os


def load_and_augment_images(batch_images):
    # define augmentation methods for Albumentations, keep original data range
    transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.RandomBrightnessContrast(),
        ], p=0.5),
        # A.OneOf([
        #     A.GaussNoise(var_limit=(10.0, 50.0)),
        #     A.Blur(blur_limit=3),
        # ], p=0.5),
    ])

    augmented_images = []
    for image in batch_images:
        image = image.astype(np.uint8)
        # apply augmentation
        augmented = transform(image=image)
        augmented_image = augmented['image']
        augmented_images.append(augmented_image)

    # convert list to tensor
    augmented_images = np.stack(augmented_images)

    return augmented_images


class H5TileDataset(Dataset):
    def __init__(self, h5_path, info_path, img_transform=None, chunk_size=1000, mask=False, sample_ratio=None, augment=False):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        self.mask = mask
        self.augment = augment
        # spot-level information (e.g., pathway scores) indexed by barcode
        self.info_df = pd.read_csv(info_path, index_col=0)
        self.info_df.index = self.info_df.index.astype(str)
        with h5py.File(h5_path, 'r') as f:
            self.n_chunks = int(np.ceil(len(f['barcode']) / chunk_size))
        self.sample_ratio = sample_ratio
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, 'r') as f:
            imgs = f['img'][start_idx:end_idx]
            barcodes = [barcode.decode('utf-8') for barcode in f['barcode'][start_idx:end_idx].flatten()]
            coords = f['coords'][start_idx:end_idx]
        info_values = torch.Tensor(self.info_df.loc[barcodes].values)
        if self.sample_ratio:
            selected = np.random.choice(range(len(imgs)), size=max(int(len(imgs)*self.sample_ratio),1), replace=False, p=None)
            imgs = imgs[selected]
            barcodes = [barcodes[i] for i in selected]
            coords = coords[selected]
            info_values = info_values[selected]
            
        if self.augment:
            imgs = load_and_augment_images(imgs)
        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
        if self.mask:
            # using circle spot mask, default false
            mask = get_disk_mask(radius=100)
            mask_tensor = torch.from_numpy(mask).expand(imgs.shape[0],3, -1, -1)

            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords, 'info_values': info_values, 'mask': mask_tensor}
        else:
            return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords, 'info_values': info_values}


class H5TileDataset_infer(Dataset):
    def __init__(self, h5_path, img_transform=None, chunk_size=64, mask=False, sample_ratio=None):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        self.mask = mask
        with h5py.File(h5_path, 'r') as f:
            self.n_chunks = int(np.ceil(len(f['img']) / chunk_size))
        self.sample_ratio = sample_ratio
        
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, 'r') as f:
            imgs = f['img'][start_idx:end_idx]
            coords = f['coords'][start_idx:end_idx]
        if self.sample_ratio:
            selected = np.random.choice(range(len(imgs)), size=max(int(len(imgs)*self.sample_ratio),1), replace=False, p=None)
            imgs = imgs[selected]
            coords = coords[selected]
            
        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
        if self.mask:
            mask = get_disk_mask(radius=220)
            mask_tensor = torch.from_numpy(mask).expand(imgs.shape[0],3, -1, -1)

            return {'imgs': imgs, 'coords': coords, 'mask': mask_tensor}
        else:
            return {'imgs': imgs, 'coords': coords,}


class H5TileDataset_runtime(Dataset):
    """Dataset that extracts images from WSI on-the-fly using coordinates from H5 file.
    
    This class is designed to be pickle-safe for multiprocessing by lazily opening
    the WSI file only when needed (not in __init__).
    """
    def __init__(self, h5_path, wsi_path, patch_size=224, img_transform=None, chunk_size=64, mask=False, sample_ratio=None):
        self.h5_path = h5_path
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        self.mask = mask
        self.sample_ratio = sample_ratio
        
        with h5py.File(h5_path, 'r') as f:
            if 'coords' not in f:
                raise KeyError(f"H5 file {h5_path} must contain 'coords' dataset")
            self.coords = f['coords'][:]
            self.n_chunks = int(np.ceil(len(self.coords) / chunk_size))
        
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
        
        # WSI slide will be opened lazily in __getitem__
        self._slide = None
        
    def _get_slide(self):
        """Lazy initialization of WSI slide object (for multiprocessing compatibility)."""
        if self._slide is None:
            self._slide = openslide.open_slide(self.wsi_path)
        return self._slide
        
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = min((idx + 1) * self.chunk_size, len(self.coords))
        chunk_coords = self.coords[start_idx:end_idx]
        
        if self.sample_ratio:
            n_samples = max(int(len(chunk_coords) * self.sample_ratio), 1)
            selected = np.random.choice(range(len(chunk_coords)), size=n_samples, replace=False)
            chunk_coords = chunk_coords[selected]

        # Get slide object (lazy initialization)
        slide = self._get_slide()
        
        imgs = []
        for coord in chunk_coords:
            x, y = coord
            top_left = (int(x - self.patch_size // 2), int(y - self.patch_size // 2))
            region = slide.read_region(top_left, 0, (self.patch_size, self.patch_size))
            img = region.convert('RGB')
            imgs.append(np.array(img))
        
        imgs = np.array(imgs)

        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
        
        if self.mask:
            mask = get_disk_mask(radius=220)
            mask_tensor = torch.from_numpy(mask).expand(imgs.shape[0], 3, -1, -1)
            return {'imgs': imgs, 'coords': chunk_coords, 'mask': mask_tensor}
        else:
            return {'imgs': imgs, 'coords': chunk_coords}
    
    def __del__(self):
        """Close WSI file when dataset is destroyed."""
        if hasattr(self, '_slide') and self._slide is not None:
            self._slide.close()