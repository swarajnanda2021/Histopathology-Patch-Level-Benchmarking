import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Base dataset class for classification
class BaseClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Base classification dataset.
        
        Args:
            root_dir: Path to dataset
            split: 'train', 'test', 'val', or 'all' (combines available splits)
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # Populate samples and targets
        self._load_dataset()

    def _get_available_splits(self):
        """Determine which splits are available in the dataset."""
        possible_splits = {
            'train': ['Training', 'train'],
            'test': ['Test', 'test'],
            'val': ['Validation', 'validation', 'val']
        }
        
        available_splits = []
        for split_name, folders in possible_splits.items():
            for folder in folders:
                if (self.root_dir / folder).exists():
                    available_splits.append((split_name, folder))
                    break
        
        return available_splits

    def _load_dataset(self):
        """
        Load dataset - to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_dataset()")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

# Base dataset for segmentation
class BaseSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', magnification='40x', transform=None):
        """
        Base segmentation dataset.
        
        Args:
            root_dir: Path to dataset
            split: 'train' or 'test'
            magnification: Resolution ('20x' or '40x')
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split if split == 'train' else 'test'
        self.transform = transform
        self.magnification = magnification
        self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset - to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_dataset()")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get an item - to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __getitem__()")




########################################################
# Dataset Implementations
########################################################



# Implementation for each classification dataset

class BRACSDataset(BaseClassificationDataset):
    """BRACS dataset with train/test/val splits."""
    def __init__(self, root_dir, split='all', transform=None):
        """
        Args:
            root_dir: Path to BRACS dataset
            split: 'train', 'test', 'val', or 'all' (combines all splits)
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        # Define class mapping (0-6 folders)
        self.class_to_idx = {str(i): i for i in range(7)}
        
        # Determine which splits to load
        if self.split == 'all':
            splits_to_load = ['train', 'test', 'val']
        else:
            splits_to_load = [self.split]
        
        # Load from each split
        for split_name in splits_to_load:
            split_dir = self.root_dir / split_name
            if not split_dir.exists():
                print(f"Warning: Split directory {split_dir} does not exist")
                continue
                
            # Load each class
            for class_idx in range(7):
                class_dir = split_dir / str(class_idx)
                if not class_dir.exists():
                    continue
                    
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = class_dir / img_name
                        self.samples.append((str(img_path), class_idx))
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples from BRACS dataset (split: {self.split})")


class MiDOGDataset(BaseClassificationDataset):
    """MiDOG++ classification dataset with train/test splits."""
    def __init__(self, root_dir, split='all', transform=None):
        """
        Args:
            root_dir: Path to MiDOG++ classification dataset
            split: 'train', 'test', or 'all' (combines all splits)
            transform: Image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        # Define class mapping
        self.class_to_idx = {'hard_negative': 0, 'mitosis': 1}
        
        # Determine which splits to load
        if self.split == 'all':
            splits_to_load = ['train', 'test']
        else:
            splits_to_load = [self.split]
        
        # Load from each split
        for split_name in splits_to_load:
            split_dir = self.root_dir / split_name
            if not split_dir.exists():
                print(f"Warning: Split directory {split_dir} does not exist")
                continue
                
            # Load each class
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                    
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = class_dir / img_name
                        self.samples.append((str(img_path), class_idx))
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples from MiDOG++ dataset (split: {self.split})")





class MHISTDataset(BaseClassificationDataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to dataset
            split: 'train', 'test', or 'all' (combines all splits)
            transform: Image transformations
        """
        super().__init__(root_dir, split, transform)
        
    def _load_dataset(self):
        # Determine which splits to load
        if self.split == 'all':
            # Look for all possible split directories
            split_dirs = []
            for possible_split in ['Training', 'Test', 'Validation', 'train', 'test', 'val']:
                split_path = self.root_dir / possible_split
                if split_path.exists():
                    split_dirs.append(split_path)
            if not split_dirs:  # If no split directories, assume flat structure
                split_dirs = [self.root_dir]
        else:
            # Map split names to directory names
            split_map = {
                'train': 'Training',
                'test': 'Test'
            }
            actual_split = split_map.get(self.split, self.split)
            split_path = self.root_dir / actual_split
            if split_path.exists():
                split_dirs = [split_path]
            else:
                split_dirs = []
        
        if not split_dirs:
            print(f"Warning: No valid split directories found for {self.split}")
            return
        
        # Get all classes from the first split directory
        classes = sorted([d for d in os.listdir(split_dirs[0]) 
                         if os.path.isdir(os.path.join(split_dirs[0], d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # Load samples from all split directories
        for split_dir in split_dirs:
            for class_name in classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        img_path = class_dir / img_name
                        self.samples.append((str(img_path), class_idx))
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples from MHIST dataset (split: {self.split})")


class CRCDataset(BaseClassificationDataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to dataset
            split: 'train', 'test', or 'all' (combines all splits)
            transform: Image transformations
        """
        super().__init__(root_dir, split, transform)
        
    def _load_dataset(self):
        # Determine which splits to load
        if self.split == 'all':
            # Look for all possible split directories
            split_dirs = []
            for possible_split in ['Training', 'Test', 'Validation', 'train', 'test', 'val']:
                split_path = self.root_dir / possible_split
                if split_path.exists():
                    split_dirs.append(split_path)
            if not split_dirs:  # If no split directories, assume flat structure
                split_dirs = [self.root_dir]
        else:
            # Map split names to directory names
            split_map = {
                'train': 'Training',
                'test': 'Test'
            }
            actual_split = split_map.get(self.split, self.split)
            split_path = self.root_dir / actual_split
            if split_path.exists():
                split_dirs = [split_path]
            else:
                split_dirs = []
        
        if not split_dirs:
            print(f"Warning: No valid split directories found for {self.split}")
            return
        
        # Get all classes from the first split directory
        classes = sorted([d for d in os.listdir(split_dirs[0]) 
                         if os.path.isdir(os.path.join(split_dirs[0], d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # Load samples from all split directories
        for split_dir in split_dirs:
            for class_name in classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg')):
                        img_path = class_dir / img_name
                        self.samples.append((str(img_path), class_idx))
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples from CRC dataset (split: {self.split})")


class PCamDataset(BaseClassificationDataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to dataset
            split: 'train', 'test', or 'all' (combines all splits)
            transform: Image transformations
        """
        super().__init__(root_dir, split, transform)
        
    def _load_dataset(self):
        # Determine which splits to load
        if self.split == 'all':
            # Look for all possible split directories
            split_dirs = []
            for possible_split in ['Training', 'Test', 'Validation', 'train', 'test', 'val']:
                split_path = self.root_dir / possible_split
                if split_path.exists():
                    split_dirs.append(split_path)
            if not split_dirs:  # If no split directories, assume flat structure
                split_dirs = [self.root_dir]
        else:
            # Map split names to directory names
            split_map = {
                'train': 'Training',
                'test': 'Test'
            }
            actual_split = split_map.get(self.split, self.split)
            split_path = self.root_dir / actual_split
            if split_path.exists():
                split_dirs = [split_path]
            else:
                split_dirs = []
        
        if not split_dirs:
            print(f"Warning: No valid split directories found for {self.split}")
            return
        
        # Get all classes from the first split directory
        classes = sorted([d for d in os.listdir(split_dirs[0]) 
                         if os.path.isdir(os.path.join(split_dirs[0], d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # Load samples from all split directories
        for split_dir in split_dirs:
            for class_name in classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg')):
                        img_path = class_dir / img_name
                        self.samples.append((str(img_path), class_idx))
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples from PCam dataset (split: {self.split})")




# Implementation for segmentation datasets
class SynchronizedTransform:
    def __init__(self, transform_settings: dict, input_shape: int = 96):
        self.transform_settings = transform_settings
        self.input_shape = input_shape
        self.mean = self.transform_settings["normalize"]["mean"]
        self.std = self.transform_settings["normalize"]["std"]
        self.train_spatial_transforms, self.train_image_transforms = self.get_train_transforms()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.val_transforms = self.get_val_transforms()

    def get_train_transforms(self):
        spatial_transforms = [
            A.RandomRotate90(p=self.transform_settings.get("RandomRotate90", {}).get("p", 0)),
            A.HorizontalFlip(p=self.transform_settings.get("HorizontalFlip", {}).get("p", 0)),
            A.VerticalFlip(p=self.transform_settings.get("VerticalFlip", {}).get("p", 0)),
        ]

        image_transforms = [
            A.Downscale(
                scale_min=self.transform_settings.get("Downscale", {}).get("scale", 0.5),
                scale_max=self.transform_settings.get("Downscale", {}).get("scale", 0.5),
                p=self.transform_settings.get("Downscale", {}).get("p", 0)
            ),
            A.Blur(blur_limit=self.transform_settings.get("Blur", {}).get("blur_limit", 7), 
                  p=self.transform_settings.get("Blur", {}).get("p", 0)),
            A.ColorJitter(
                brightness=self.transform_settings.get("ColorJitter", {}).get("scale_setting", 0.25),
                contrast=self.transform_settings.get("ColorJitter", {}).get("scale_setting", 0.25),
                saturation=self.transform_settings.get("ColorJitter", {}).get("scale_color", 0.1),
                hue=self.transform_settings.get("ColorJitter", {}).get("scale_color", 0.05),
                p=self.transform_settings.get("ColorJitter", {}).get("p", 0)
            ),
            ToTensorV2()
        ]

        return A.Compose(spatial_transforms), A.Compose(image_transforms)

    def get_val_transforms(self):
        return A.Compose([
            ToTensorV2()
        ])
    
    def _ensure_range_0_to_1(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                return image.astype(np.float32) / 255.0
            elif image.dtype == np.float32 or image.dtype == np.float64:
                return np.clip(image, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.dtype == torch.uint8:
                return image.float() / 255.0
            elif image.dtype == torch.float32 or image.dtype == torch.float64:
                return torch.clamp(image, 0, 1)
        return image

    def __call__(self, image, mask, is_training=True):
        if is_training:
            combined = np.concatenate([image, mask], axis=2)
            transformed_combined = self.train_spatial_transforms(image=combined)["image"]
            # Apply spatial transforms
            image = transformed_combined[:, :, :3]
            mask = transformed_combined[:, :, 3:4]
            # Apply 
            image = self.train_image_transforms(image=image)["image"]
        else:
            image = self.val_transforms(image=image)["image"]
            mask = mask
        
        image = self._ensure_range_0_to_1(image)
        image = self.normalize(image)

        return image, mask

class HoverNetBasedDataset(Dataset):
    def __init__(self, data_dir, split='Training', magnification='20x', transform=None):
        self.data_dir = data_dir
        self.split = split if split == 'Training' else 'Test'
        self.transform = transform
        self.magnification = magnification
        self.transform_fin = A.Compose([ToTensorV2()])
        self.patch_names = self._get_patchnames()

    def _get_patchnames(self):
        patch_dir = os.path.join(self.data_dir, self.split, self.magnification, 'tissue_images')
        return [f.rsplit('.', 1)[0] for f in os.listdir(patch_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.patch_names)
    
    def _calculate_distance_maps(self, mask):
        h_map = np.zeros(mask.shape[:2], dtype=np.float32)
        v_map = np.zeros(mask.shape[:2], dtype=np.float32)

        mask = np.sum(mask, axis=-1) - 1

        unique_list = np.unique(mask)
        max_value = np.max(unique_list)

        # Remove the maximum value from the unique list (corresponds to background)
        unique_list = unique_list[unique_list < max_value]

        for nuclei in unique_list:
            if nuclei <= 0:  # Skip background
                continue

            nucleus_mask = mask == nuclei
            y_indices, x_indices = np.nonzero(nucleus_mask)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                centroid_y, centroid_x = np.mean(y_indices), np.mean(x_indices)
                
                h_distances = x_indices - centroid_x
                v_distances = y_indices - centroid_y
                
                # Normalize distances for this nucleus to [-1, 1]
                max_distance = max(np.max(np.abs(h_distances)), np.max(np.abs(v_distances)))
                if max_distance > 0:
                    h_distances = h_distances / max_distance
                    v_distances = v_distances / max_distance
                
                h_map[nucleus_mask] = h_distances
                v_map[nucleus_mask] = v_distances

        return h_map, v_map

    def _process_mask_binary(self, mask):
        # Create a single channel mask
        mask = np.squeeze(mask)
        single_channel_mask = np.zeros(mask.shape, dtype=np.uint8)
        # Find the maximum value in the mask
        max_value = np.max(mask)
        # Set all non-zero values to 1, except the maximum value
        single_channel_mask[(mask > 0) & (mask < max_value)] = 1
        return single_channel_mask

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        image_path = os.path.join(self.data_dir, self.split, self.magnification, 'tissue_images', f'{patch_name}.png')
        mask_path = os.path.join(self.data_dir, self.split, self.magnification, 'masks', f'{patch_name}.npy')
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Load mask
        mask = np.load(mask_path)
        mask = mask[..., None]
        mask = mask.astype(np.float32) 

        if self.transform:
            image, mask = self.transform(image=image, mask=mask)
        
        # Calculate the binary mask and the distance maps
        h_map, v_map = self._calculate_distance_maps(mask)
        binary_mask = self._process_mask_binary(mask) # change single channel mask to binary mask with nuclei as 1, and background as 0
        # Create a 2-channel mask: [nuclei, background]
        mask_2ch = np.stack([binary_mask, 1 - binary_mask], axis=2)
        # Stack h_map and v_map
        distance_map = np.stack([v_map, h_map], axis=2)

        mask_2ch = self.transform_fin(image=mask_2ch)["image"]
        distance_map = self.transform_fin(image=distance_map)["image"]
        instance_mask = self.transform_fin(image=mask)["image"]

        return image, mask_2ch, distance_map, instance_mask



# Dataset loaders for PanNuke and MonuSeg
class PanNukeDataset(HoverNetBasedDataset):
    def __init__(self, data_dir, split='Training', magnification='20x', transform=None):
        super().__init__(data_dir, split, magnification, transform)

class MonuSegDataset(HoverNetBasedDataset):
    def __init__(self, data_dir, split='Training', magnification='40x', transform=None):
        super().__init__(data_dir, split, magnification, transform)




class DinoTransforms(object):
    """Enhanced transformations for DINO with support for multiple augmentations per type."""
    def __init__(
        self,
        local_size=96,
        global_size=224,
        local_crop_scale=(0.05, 0.4),
        global_crop_scale=(0.4, 1.0),
        n_local_crops=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        augmentations_per_type=1,  # New parameter to control augmentations per type
    ):
        self.n_local_crops = n_local_crops
        self.global_size = global_size
        self.local_size = local_size
        self.local_crop_scale = local_crop_scale
        self.global_crop_scale = global_crop_scale
        self.mean = mean
        self.std = std
        self.augmentations_per_type = augmentations_per_type
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose([
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # DINO color augmentation
        self.flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.01),
        ])

        # Global view 1 - repeated for augmentations_per_type
        self.global_1 = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(3, (0.1, 0.15)),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        # Global view 2 - repeated for augmentations_per_type
        self.global_2 = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=global_size, scale=global_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.15)),
            transforms.RandomSolarize(threshold=64, p=0.5),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

        # Local crops - repeated for augmentations_per_type
        self.local = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.RandomResizedCrop(size=local_size, scale=local_crop_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(3, (0.1, 0.15)),
            self.to_tensor,
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x):
        """Generate multiple augmentations for each type."""
        crops = []
        
        # Generate augmentations_per_type for global_1
        for _ in range(self.augmentations_per_type):
            crops.append(self.global_1(x))
        
        # Generate augmentations_per_type for global_2
        for _ in range(self.augmentations_per_type):
            crops.append(self.global_2(x))
        
        # Generate augmentations_per_type for local views
        for _ in range(self.augmentations_per_type * self.n_local_crops):
            crops.append(self.local(x))
            
        return crops






class BCSSDataset(Dataset):
    """
    BCSS Multi-class Tissue Segmentation Dataset.

    Features:
    - Creates train/val/test splits on-the-fly with reproducibility
    - Handles large variable-sized images
    - Random crops during training with augmentation
    - Multiple crops per image for validation/test
    """

    # Class mapping
    CLASS_NAMES = {
        0: 'outside_roi',
        1: 'tumor',
        2: 'stroma',
        3: 'lymphocytic_infiltrate',
        4: 'necrosis_or_debris',
        5: 'glandular_secretions',
        6: 'blood',
        7: 'exclude',
        8: 'metaplasia_NOS',
        9: 'fat',
        10: 'plasma_cells',
        11: 'other_immune_infiltrate',
        12: 'mucoid_material',
        13: 'normal_acinus_or_duct',
        14: 'lymphatics',
        15: 'undetermined',
        16: 'nerve',
        17: 'skin_adnexa',
        18: 'blood_vessel',
        19: 'angioinvasion',
        20: 'dcis',
        21: 'other'
    }

    def __init__(
        self,
        data_dir,
        split='train',
        crop_size=224,
        transform=None,
        exclude_classes=[0, 7],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        crops_per_image_train=10,  # Random crops per image during training
        crops_per_image_val=5,     # Fixed crops per image during val/test
        min_foreground_ratio=0.1,  # Minimum ratio of foreground pixels in crop
    ):
        """
        Args:
            data_dir: Path to BCSS dataset root (contains 'rgbs_colorNormalized' and 'masks')
            split: 'train', 'val', or 'test'
            crop_size: Size to crop patches to (default 224)
            transform: SynchronizedTransform object for augmentation
            exclude_classes: Classes to ignore (default: outside_roi=0, exclude=7)
            train_ratio: Fraction of images for training
            val_ratio: Fraction of images for validation
            test_ratio: Fraction of images for testing
            random_seed: Random seed for split reproducibility
            crops_per_image_train: Number of random crops per image per epoch (training)
            crops_per_image_val: Number of fixed crops per image (val/test)
            min_foreground_ratio: Minimum foreground ratio to keep a crop
        """
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.crop_size = crop_size
        self.transform = transform
        self.transform_fin = A.Compose([ToTensorV2()])
        self.exclude_classes = exclude_classes
        self.num_classes = 22
        self.crops_per_image_train = crops_per_image_train
        self.crops_per_image_val = crops_per_image_val
        self.min_foreground_ratio = min_foreground_ratio
        self.random_seed = random_seed

        # Dataset structure: rgbs_colorNormalized/ and masks/
        self.image_dir = self.data_dir / 'rgbs_colorNormalized'
        self.mask_dir = self.data_dir / 'masks'

        # Get all image files
        all_image_files = sorted([f for f in os.listdir(self.image_dir)
                                  if f.endswith(('.png', '.tif', '.jpg'))])

        # Create reproducible train/val/test split
        np.random.seed(random_seed)
        n_images = len(all_image_files)
        indices = np.random.permutation(n_images)

        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]

        # Select images for this split
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        elif self.split == 'test':
            split_indices = test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.image_files = [all_image_files[i] for i in split_indices]

        print(f"BCSS {self.split} split: {len(self.image_files)} images")
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        # For val/test, pre-generate valid crop locations for each image
        if self.split in ['val', 'test']:
            self.crop_locations = self._generate_valid_crops()
            print(f"  Generated {len(self.crop_locations)} valid crops")
        else:
            self.crop_locations = None
            print(f"  Will generate {self.crops_per_image_train} random crops per image per epoch")

    def _is_valid_crop(self, mask_crop):
        """Check if crop has sufficient foreground content."""
        # Count non-excluded pixels
        valid_mask = np.ones_like(mask_crop, dtype=bool)
        for exc_class in self.exclude_classes:
            valid_mask &= (mask_crop != exc_class)

        foreground_ratio = valid_mask.sum() / mask_crop.size
        return foreground_ratio >= self.min_foreground_ratio

    def _generate_valid_crops(self):
        """Pre-generate valid crop locations for validation/test."""
        crop_locations = []

        for img_idx, img_name in enumerate(tqdm(self.image_files, desc=f"Generating {self.split} crops")):
            # Load mask to determine valid crop locations
            mask_name = img_name.replace('.png', '_mask.png')  # Adjust extension if needed
            mask_path = self.mask_dir / mask_name

            if not mask_path.exists():
                # Try without suffix
                mask_name = img_name
                mask_path = self.mask_dir / mask_name

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            img_h, img_w = mask.shape

            # Try to generate valid crops
            attempts = 0
            max_attempts = self.crops_per_image_val * 10
            crops_found = 0

            while crops_found < self.crops_per_image_val and attempts < max_attempts:
                # Random crop location
                if img_h > self.crop_size and img_w > self.crop_size:
                    top = np.random.randint(0, img_h - self.crop_size + 1)
                    left = np.random.randint(0, img_w - self.crop_size + 1)

                    # Extract crop
                    mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]

                    # Check if valid
                    if self._is_valid_crop(mask_crop):
                        crop_locations.append({
                            'img_idx': img_idx,
                            'img_name': img_name,
                            'top': top,
                            'left': left,
                        })
                        crops_found += 1

                attempts += 1

            if crops_found < self.crops_per_image_val:
                print(f"Warning: Only found {crops_found}/{self.crops_per_image_val} valid crops for {img_name}")

        return crop_locations

    def _get_random_valid_crop(self, image, mask):
        """Get a random crop with sufficient foreground content."""
        img_h, img_w = image.shape[:2]

        max_attempts = 50
        for attempt in range(max_attempts):
            if img_h > self.crop_size and img_w > self.crop_size:
                top = np.random.randint(0, img_h - self.crop_size + 1)
                left = np.random.randint(0, img_w - self.crop_size + 1)
            else:
                # Image smaller than crop size - need to pad
                top, left = 0, 0

            # Extract crop
            image_crop = image[top:top+self.crop_size, left:left+self.crop_size]
            mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]

            # Pad if necessary
            if image_crop.shape[0] < self.crop_size or image_crop.shape[1] < self.crop_size:
                pad_h = max(0, self.crop_size - image_crop.shape[0])
                pad_w = max(0, self.crop_size - image_crop.shape[1])
                image_crop = np.pad(image_crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            # Check if valid
            if self._is_valid_crop(mask_crop):
                return image_crop, mask_crop

        # If no valid crop found after max attempts, return last crop anyway
        print(f"Warning: Could not find valid crop after {max_attempts} attempts")
        return image_crop, mask_crop

    def __len__(self):
        if self.split == 'train':
            # In training, we generate multiple crops per image
            return len(self.image_files) * self.crops_per_image_train
        else:
            # In val/test, use pre-generated crops
            return len(self.crop_locations)

    def __getitem__(self, idx):
        if self.split == 'train':
            # Training: random crop from random image
            img_idx = idx // self.crops_per_image_train
            img_name = self.image_files[img_idx]

            # Load full image and mask
            img_path = self.image_dir / img_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

            # Load mask
            mask_name = img_name.replace('.png', '_mask.png')
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                mask_name = img_name
                mask_path = self.mask_dir / mask_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Get random valid crop
            image_crop, mask_crop = self._get_random_valid_crop(image, mask)

            # Apply augmentations (spatial transforms synchronized)
            if self.transform:
                # Spatial transforms (synchronized between image and mask)
                transformed = self.transform.train_spatial_transforms(
                    image=image_crop,
                    mask=mask_crop
                )
                image_crop = transformed['image']
                mask_crop = transformed['mask']

                # Image-only transforms (color augmentation)
                image_crop = self.transform.train_image_transforms(image=image_crop)['image']

        else:
            # Val/Test: use pre-generated crop
            crop_info = self.crop_locations[idx]
            img_name = crop_info['img_name']
            top = crop_info['top']
            left = crop_info['left']

            # Load full image and mask
            img_path = self.image_dir / img_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

            mask_name = img_name.replace('.png', '_mask.png')
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                mask_name = img_name
                mask_path = self.mask_dir / mask_name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Extract pre-determined crop
            image_crop = image[top:top+self.crop_size, left:left+self.crop_size]
            mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]

            # No augmentation for val/test, but apply val transforms (just to tensor)
            if self.transform:
                image_crop = self.transform.val_transforms(image=image_crop)['image']

        # Convert to tensors
        if not isinstance(image_crop, torch.Tensor):
            image_crop = self.transform_fin(image=image_crop)["image"]
            image_crop = image_crop.float() / 255.0 if image_crop.dtype == torch.uint8 else image_crop

            # Apply normalization
            if self.transform:
                image_crop = self.transform.normalize(image_crop)

        mask_tensor = torch.from_numpy(mask_crop).long()

        return image_crop, mask_tensor



class GLASDataset(Dataset):
    """
    GLAS Gland Segmentation Dataset with distance transform for instance segmentation.

    Training: Random 224x224 crops
    Validation/Test: Non-overlapping 224x224 grid crops
    """

    def __init__(self, data_dir, split='train', crop_size=224, transform=None):
        """
        Args:
            data_dir: Path to GLAS dataset root
            split: 'train' or 'test'
            crop_size: Size to crop patches to (default 224)
            transform: Optional additional transforms
        """
        from scipy.ndimage import distance_transform_edt

        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.crop_size = crop_size
        self.transform = transform
        self.transform_fin = A.Compose([ToTensorV2()])

        # Get image files
        split_dir = self.data_dir / self.split
        self.image_files = sorted([f for f in os.listdir(split_dir)
                                   if f.endswith('.bmp') and '_anno' not in f])

        # For test/validation, generate grid of non-overlapping crops
        if self.split == 'test':
            self.crop_metadata = self._generate_grid_crops()
        else:
            self.crop_metadata = None

        print(f"Loaded {len(self)} crops from GLAS {self.split} set ({len(self.image_files)} images)")

    def _generate_grid_crops(self):
        """Generate non-overlapping grid crops for validation/test."""
        crop_metadata = []

        for img_idx, img_name in enumerate(self.image_files):
            # Load image to get dimensions
            img_path = self.data_dir / self.split / img_name
            image = cv2.imread(str(img_path))
            img_h, img_w = image.shape[:2]

            # Calculate grid dimensions
            n_rows = img_h // self.crop_size
            n_cols = img_w // self.crop_size

            # Generate crop positions
            for row in range(n_rows):
                for col in range(n_cols):
                    top = row * self.crop_size
                    left = col * self.crop_size

                    crop_metadata.append({
                        'img_idx': img_idx,
                        'img_name': img_name,
                        'top': top,
                        'left': left,
                        'row': row,
                        'col': col
                    })

        return crop_metadata

    def _calculate_distance_transform(self, instance_mask):
        """
        Calculate distance transform maps for instance segmentation.

        Returns two channels:
        - Channel 1: Distance to nearest boundary (normalized)
        - Channel 2: Same as channel 1 (for compatibility with 2-channel expectation)
        """
        from scipy.ndimage import distance_transform_edt

        distance_map = np.zeros(instance_mask.shape[:2], dtype=np.float32)

        # Get unique instance IDs (excluding background=0)
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]

        for instance_id in unique_instances:
            # Get binary mask for this instance
            instance_binary = (instance_mask == instance_id).astype(np.uint8)

            # Calculate distance transform (distance to nearest boundary from inside)
            dist = distance_transform_edt(instance_binary)

            # Normalize to [0, 1] for this instance
            if dist.max() > 0:
                dist = dist / dist.max()

            # Store in distance map
            distance_map[instance_binary > 0] = dist[instance_binary > 0]

        return distance_map

    def _process_mask_binary(self, instance_mask):
        """Create binary mask from instance mask (foreground=1, background=0)."""
        binary_mask = (instance_mask > 0).astype(np.uint8)
        return binary_mask

    def _get_random_crop_params(self, img_h, img_w):
        """Get random crop parameters for training."""
        if img_h < self.crop_size or img_w < self.crop_size:
            # If image is smaller than crop size, we'll pad it
            pad_h = max(0, self.crop_size - img_h)
            pad_w = max(0, self.crop_size - img_w)
            return 0, 0, pad_h, pad_w

        # Random crop position
        top = np.random.randint(0, img_h - self.crop_size + 1)
        left = np.random.randint(0, img_w - self.crop_size + 1)

        return top, left, 0, 0

    def _apply_crop(self, image, mask, top, left, pad_h, pad_w):
        """Apply crop or padding to image and mask."""
        # Pad if necessary
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # Crop
        image_crop = image[top:top+self.crop_size, left:left+self.crop_size]
        mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]

        return image_crop, mask_crop

    def __len__(self):
        if self.crop_metadata is not None:
            return len(self.crop_metadata)
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        # Determine which image and crop position to use
        if self.crop_metadata is not None:
            # Test/validation: use grid crop
            metadata = self.crop_metadata[idx]
            img_name = metadata['img_name']
            top, left = metadata['top'], metadata['left']
            pad_h, pad_w = 0, 0
        else:
            # Training: random crop
            img_name = self.image_files[idx]
            top, left, pad_h, pad_w = None, None, None, None

        # Load image
        img_path = self.data_dir / self.split / img_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation (instance mask)
        anno_name = img_name.replace('.bmp', '_anno.bmp')
        anno_path = self.data_dir / self.split / anno_name
        instance_mask = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)

        # Convert image to float [0, 1]
        image = image.astype(np.float32) / 255.0

        # Get crop parameters if not already set
        if top is None:
            img_h, img_w = image.shape[:2]
            top, left, pad_h, pad_w = self._get_random_crop_params(img_h, img_w)

        # Apply crop to image and instance mask
        image_crop, instance_mask_crop = self._apply_crop(
            image, instance_mask, top, left, pad_h, pad_w
        )

        # Calculate distance transform from cropped instance mask
        distance_map = self._calculate_distance_transform(instance_mask_crop)

        # Create binary mask
        binary_mask = self._process_mask_binary(instance_mask_crop)

        # Apply optional transforms (spatial only, synchronized)
        if self.transform and self.split == 'train':
            # Only apply augmentations during training
            transformed = self.transform.train_spatial_transforms(
                image=image_crop,
                mask=np.stack([binary_mask, distance_map], axis=-1)
            )
            image_crop = transformed['image']
            binary_mask = transformed['mask'][..., 0]
            distance_map = transformed['mask'][..., 1]

            # Apply image-only transforms
            image_crop = self.transform.train_image_transforms(image=image_crop)['image']

        # Create 2-channel mask: [foreground, background]
        mask_2ch = np.stack([binary_mask, 1 - binary_mask], axis=2)

        # Stack distance maps: [distance, distance] (2 channels for compatibility)
        distance_map_2ch = np.stack([distance_map, distance_map], axis=2)

        # Convert to tensors
        if not isinstance(image_crop, torch.Tensor):
            image_crop = self.transform_fin(image=image_crop)["image"]
            image_crop = image_crop.float() / 255.0 if image_crop.dtype == torch.uint8 else image_crop
            # Apply normalization
            if self.transform:
                image_crop = self.transform.normalize(image_crop)

        mask_2ch = self.transform_fin(image=mask_2ch)["image"].float()
        distance_map_2ch = self.transform_fin(image=distance_map_2ch)["image"].float()
        instance_mask_tensor = self.transform_fin(image=instance_mask_crop[..., None])["image"].float()

        return image_crop, mask_2ch, distance_map_2ch, instance_mask_tensor
