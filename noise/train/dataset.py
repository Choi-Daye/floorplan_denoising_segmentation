import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FloorplanDataset(Dataset):
    def __init__(self, image_dir, clean_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.clean_dir = clean_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        VALID_EXTENSIONS = ['.png', '.PNG']

        def is_image_file(filename):
            return any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS)

        # 정렬
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(image_dir, f)) and is_image_file(f)        # 숨김파일(점으로 시작하는) 제외, 폴더가 아닌 실제 파일만 포함, VALID_EXTENSIONS 안의 확자자만 포함
        ])

        self.cleans = sorted([
            f for f in os.listdir(clean_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(clean_dir, f)) and is_image_file(f)
        ])

        self.masks = sorted([
            f for f in os.listdir(mask_dir)
            if not f.startswith('.') and os.path.isfile(os.path.join(mask_dir, f)) and is_image_file(f)
        ])

        assert len(self.images) == len(self.cleans) and len(self.cleans) == len(self.masks), \
            f"Number of images ({len(self.images)}) and cleans ({len(self.cleans)}) and masks ({len(self.masks)}) do not match!"


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.image_dir, self.images[idx])).convert('L'))
        clean = np.array(Image.open(os.path.join(self.clean_dir, self.cleans[idx])).convert('L'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert('L'))

        if self.transform:
            augmented = self.transform(image=image, clean=clean, mask=mask)
            image = augmented['image']  # float32 tensor, 0~1 normalized
            clean = augmented['clean']  # float32 tensor, 0~1 normalized
            mask = augmented['mask'].long()    # long tensor with class IDs

        else:
            image = torch.from_numpy(image).float() / 255.0
            clean = torch.from_numpy(clean).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if image.dtype != torch.float32:
            image = image.float() / 255.0

        if clean.dtype != torch.float32:
            clean = clean.float() / 255.0

        return image, clean, mask


def get_loaders(image_dir, clean_dir, mask_dir, batch_size, shuffle=True):
    print(f"📂 Loading dataset...")
    print(f"   Image path: {image_dir}")
    print(f"   Clean path: {clean_dir}")
    print(f"   Mask path: {mask_dir}")
    
    transform = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),  # mask class id 보존
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
        ToTensorV2()
    ], additional_targets={'clean': 'image', 'mask': 'mask'}, is_check_shapes=False)
    
    dataset = FloorplanDataset(image_dir, clean_dir, mask_dir, transform=transform)
    
    print(f"   ✓ Total samples: {len(dataset)}")
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Shuffle: {shuffle}")
    
    loader = torch.utils.data.DataLoader(       # 배치 생성기
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0       # Recommended for MacBook MPS
    )
    
    print(f"   ✓ Total batches: {len(loader)}")
    
    return loader