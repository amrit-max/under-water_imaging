import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')

def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from {filepath}')
    return model, optimizer, epoch, loss

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11):
    """Calculate Structural Similarity Index (simplified version)"""
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim.item()

def visualize_results(hazy_img, clean_img, output_img, save_path=None):
    """Visualize comparison of hazy, clean, and predicted images"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    
    if torch.is_tensor(hazy_img):
        hazy_img = hazy_img.permute(1, 2, 0).cpu().numpy()
    if torch.is_tensor(clean_img):
        clean_img = clean_img.permute(1, 2, 0).cpu().numpy()
    if torch.is_tensor(output_img):
        output_img = output_img.permute(1, 2, 0).cpu().numpy()
    
    axes[0].imshow(hazy_img)
    axes[0].set_title('Hazy Image')
    axes[0].axis('off')
    
    axes[1].imshow(output_img)
    axes[1].set_title('Dehazed Image')
    axes[1].axis('off')
    
    axes[2].imshow(clean_img)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    return total_params, trainable_params

def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().detach()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0).numpy()
    
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

def create_comparison_grid(images, labels, save_path=None):
    """Create a grid of images for comparison"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_mb:.2f} MB')
    return size_mb