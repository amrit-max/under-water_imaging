import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Root directory containing 'train' and 'val' folders
            mode: 'train' or 'val'
            transform: Optional transform to be applied
        """
        self.hazy_dir = os.path.join(root_dir, mode, 'hazy')
        self.clean_dir = os.path.join(root_dir, mode, 'clean')
        self.transform = transform
        
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        
        all_hazy = os.listdir(self.hazy_dir)
        all_clean = os.listdir(self.clean_dir)
        
        hazy_images = sorted([f for f in all_hazy 
                             if f.lower().endswith(valid_extensions)])
        clean_images = sorted([f for f in all_clean 
                              if f.lower().endswith(valid_extensions)])
        
        
        hazy_set = set(hazy_images)
        clean_set = set(clean_images)
        common_images = sorted(hazy_set & clean_set)
        
        self.hazy_images = common_images
        self.clean_images = common_images
        
        print(f"\n{mode.upper()} Dataset:")
        print(f"  Total hazy images: {len(hazy_images)}")
        print(f"  Total clean images: {len(clean_images)}")
        print(f"  Using {len(common_images)} matching pairs")
        
        if len(common_images) == 0:
            raise ValueError("No matching image pairs found!")
        
      
        skipped_hazy = len(hazy_images) - len(common_images)
        skipped_clean = len(clean_images) - len(common_images)
        if skipped_hazy > 0 or skipped_clean > 0:
            print(f"  Skipped {skipped_hazy} hazy and {skipped_clean} clean images without pairs\n")
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        hazy_img = Image.open(hazy_path).convert('RGB')
        
        
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        clean_img = Image.open(clean_path).convert('RGB')
        
       
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
        
        return hazy_img, clean_img

def get_transforms(img_size=256):
    """Get data transforms for training and validation"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return transform