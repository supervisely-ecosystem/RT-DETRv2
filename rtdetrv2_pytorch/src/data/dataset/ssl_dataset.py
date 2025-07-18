import os
import random
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as nn_F
from PIL import Image
from ...core import register
from .coco_dataset import CocoDetection
from ..dataloader import BaseCollateFunction


@register()
class SSLDataset(CocoDetection):
    
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False, **kwargs):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.sample_mode = kwargs.get('sample_mode', False)
        
        # Simple transform for unlabeled data (resize to 640x640)
        self.unlabeled_transform = lambda img: F.resize(img, (640, 640))
        
        if self.sample_mode:
            # Sample unlabeled data from the labeled dataset (for debugging)
            self.labeled_sample_size = kwargs['labeled_sample_size']
            self.unlabeled_sample_size = kwargs['unlabeled_sample_size']
            seed = kwargs.get('seed', 42)
            
            # Split the dataset indices
            total_size = super().__len__()
            assert self.labeled_sample_size + self.unlabeled_sample_size <= total_size, \
                f"Requested samples ({self.labeled_sample_size} + {self.unlabeled_sample_size}) exceed dataset size ({total_size})"
            
            # Create random split
            random.seed(seed)
            all_indices = list(range(total_size))
            random.shuffle(all_indices)
            
            self.labeled_indices = all_indices[:self.labeled_sample_size]
            self.unlabeled_indices = all_indices[self.labeled_sample_size:self.labeled_sample_size + self.unlabeled_sample_size]
            
        elif 'unlabeled_img_folder' in kwargs:
            self.unlabeled_img_folder = kwargs['unlabeled_img_folder']
            # List all image files in the unlabeled folder
            self.unlabeled_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                self.unlabeled_images.extend(Path(self.unlabeled_img_folder).glob(ext))
            self.unlabeled_images = sorted([str(p) for p in self.unlabeled_images])
            
            # In this mode, all labeled data is available
            self.labeled_indices = list(range(len(self.coco.imgs)))
            self.unlabeled_indices = list(range(len(self.unlabeled_images)))
            
        else:
            print("Warning: SSLDataset initialized without `sample_mode` or `unlabeled_img_folder`. "
                  "The dataset will work in non semi-supervised mode.")
            self.labeled_indices = list(range(len(self)))
            self.unlabeled_indices = []
        
        if len(self.unlabeled_indices) == 0:
            print("Warning: No unlabeled data used. The dataset will work in normal mode.")
    
    def __len__(self):
        return len(self.labeled_indices)
    
    def __getitem__(self, idx):
        # Get labeled sample
        labeled_idx = self.labeled_indices[idx]            
        img_labeled, target = self.load_item(labeled_idx)
        
        # Apply transforms to labeled data
        if self._transforms is not None:
            img_labeled, target, _ = self._transforms(img_labeled, target, self)
        
        # Get unlabeled sample
        if len(self.unlabeled_indices) > 0:
            # Randomly sample an unlabeled image
            unlabeled_idx = random.choice(self.unlabeled_indices)
            
            if self.sample_mode:
                # Load from the same COCO dataset
                img_unlabeled, _ = super(CocoDetection, self).__getitem__(unlabeled_idx)
            else:
                # Load from unlabeled folder
                img_path = self.unlabeled_images[unlabeled_idx]
                img_unlabeled = Image.open(img_path).convert('RGB')
            
            # Apply unlabeled transform
            img_unlabeled = self.unlabeled_transform(img_unlabeled)
            img_unlabeled = F.to_tensor(img_unlabeled)
        else:
            img_unlabeled = None
        
        return img_labeled, target, img_unlabeled


@register()
class SSLBatchImageCollateFunction(BaseCollateFunction):
    """Collate function for SSL dataset that returns labeled images, targets, and unlabeled images."""
    
    def __init__(self, scales=None, stop_epoch=None) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
    
    def __call__(self, items):
        # Separate the three components
        images_labeled = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        if items[0][2] is None:
            images_unlabeled = None
        else:
            images_unlabeled = torch.cat([x[2][None] for x in items], dim=0)
        
        # Apply random scaling if specified (only to labeled images and their targets)
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images_labeled = nn_F.interpolate(images_labeled, size=sz)
            
            # Note: Mask handling is not implemented as you mentioned we're focusing on detection only
            if 'masks' in targets[0]:
                raise NotImplementedError('Mask scaling in SSL setting not implemented')
        
        if images_unlabeled is not None:
            return images_labeled, targets, images_unlabeled
        else:
            return images_labeled, targets