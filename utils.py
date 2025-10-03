import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from scipy import ndimage
from skimage import color, filters

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

def calculate_uiqm(img):
    """
    Calculate Underwater Image Quality Measure (UIQM)
    
    Args:
        img: PIL Image or numpy array (0-255 range)
    
    Returns:
        float: UIQM score
    """
    # Convert to numpy array if PIL Image
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Ensure image is in 0-255 range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Convert to float for calculations
    img = img.astype(np.float64)
    
    # Calculate UIQM components
    # 1. Colorfulness Measure (UICM)
    uicm = calculate_uicm(img)
    
    # 2. Sharpness Measure (UISM) 
    uism = calculate_uism(img)
    
    # 3. Contrast Measure (UIConM)
    uiconm = calculate_uiconm(img)
    
    # UIQM formula: 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    
    return uiqm

def calculate_uicm(img):
    """Calculate Underwater Image Colorfulness Measure"""
    # Convert to YUV color space
    yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YUV)
    y, u, v = yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]
    
    # Calculate colorfulness
    alpha = np.mean(u) - np.mean(v)
    beta = 0.5 * (np.mean(u) + np.mean(v)) - np.mean(y)
    
    uicm = np.sqrt(alpha**2 + beta**2)
    return uicm

def calculate_uism(img):
    """Calculate Underwater Image Sharpness Measure"""
    # Convert to grayscale
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel operator for edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate sharpness measure
    uism = np.mean(gradient_magnitude)
    return uism

def calculate_uiconm(img):
    """Calculate Underwater Image Contrast Measure"""
    # Convert to grayscale
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate local contrast using standard deviation
    # Apply Gaussian filter first
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate local standard deviation
    local_std = ndimage.generic_filter(gray, np.std, size=5)
    
    # Calculate contrast measure
    uiconm = np.mean(local_std)
    return uiconm

def calculate_image_metrics(original_img, enhanced_img):
    """
    Calculate PSNR, SSIM, and UIQM metrics for image comparison
    
    Args:
        original_img: PIL Image or numpy array of original image
        enhanced_img: PIL Image or numpy array of enhanced image
    
    Returns:
        dict: Dictionary containing PSNR, SSIM, and UIQM scores
    """
    # Convert PIL Images to numpy arrays if needed
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    if isinstance(enhanced_img, Image.Image):
        enhanced_img = np.array(enhanced_img)
    
    # Ensure images are in 0-1 range for PSNR and SSIM calculations
    if original_img.max() > 1.0:
        original_img = original_img.astype(np.float32) / 255.0
    if enhanced_img.max() > 1.0:
        enhanced_img = enhanced_img.astype(np.float32) / 255.0
    
    # Convert to tensors for PSNR and SSIM
    original_tensor = torch.from_numpy(original_img).permute(2, 0, 1).unsqueeze(0)
    enhanced_tensor = torch.from_numpy(enhanced_img).permute(2, 0, 1).unsqueeze(0)
    
    # Calculate PSNR
    psnr = calculate_psnr(enhanced_tensor, original_tensor)
    
    # Calculate SSIM
    ssim = calculate_ssim(enhanced_tensor, original_tensor)
    
    # Calculate UIQM for enhanced image (convert back to 0-255 range)
    enhanced_for_uiqm = (enhanced_img * 255).astype(np.uint8)
    uiqm = calculate_uiqm(enhanced_for_uiqm)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'uiqm': uiqm
    }

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