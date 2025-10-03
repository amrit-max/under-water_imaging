#!/usr/bin/env python3
"""
Test script to demonstrate PSNR, SSIM, and UIQM metrics calculation
"""

import os
import sys
from PIL import Image
import numpy as np
from utils import calculate_image_metrics

def test_metrics():
    """Test the metrics calculation with sample images"""
    
    # Check if we have sample images in uploads folder
    uploads_dir = 'uploads'
    outputs_dir = 'outputs'
    
    if not os.path.exists(uploads_dir) or not os.path.exists(outputs_dir):
        print("No sample images found. Please run the Flask app and process some images first.")
        return
    
    # Find sample images
    upload_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    output_files = [f for f in os.listdir(outputs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not upload_files or not output_files:
        print("No processed images found. Please run the Flask app and process some images first.")
        return
    
    print("Testing Image Quality Metrics")
    print("=" * 50)
    
    # Test with the first available pair
    original_file = upload_files[0]
    enhanced_file = output_files[0]
    
    original_path = os.path.join(uploads_dir, original_file)
    enhanced_path = os.path.join(outputs_dir, enhanced_file)
    
    print(f"Original image: {original_file}")
    print(f"Enhanced image: {enhanced_file}")
    print()
    
    try:
        # Load images
        original_img = Image.open(original_path).convert('RGB')
        enhanced_img = Image.open(enhanced_path).convert('RGB')
        
        # Calculate metrics
        metrics = calculate_image_metrics(original_img, enhanced_img)
        
        print("Image Quality Metrics:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  UIQM: {metrics['uiqm']:.4f}")
        print()
        
        # Interpretation
        print("Metric Interpretation:")
        print(f"  PSNR: {'Excellent' if metrics['psnr'] > 40 else 'Good' if metrics['psnr'] > 30 else 'Fair' if metrics['psnr'] > 20 else 'Poor'} quality")
        print(f"  SSIM: {'Excellent' if metrics['ssim'] > 0.9 else 'Good' if metrics['ssim'] > 0.8 else 'Fair' if metrics['ssim'] > 0.7 else 'Poor'} structural similarity")
        print(f"  UIQM: {'Excellent' if metrics['uiqm'] > 3.0 else 'Good' if metrics['uiqm'] > 2.0 else 'Fair' if metrics['uiqm'] > 1.0 else 'Poor'} underwater image quality")
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")

if __name__ == "__main__":
    test_metrics()
