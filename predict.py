import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import UNet
import argparse
from utils import calculate_image_metrics

class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
       
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.to_pil = transforms.ToPILImage()
    
    def predict(self, image_path, output_path, calculate_metrics=True):
        """
        Dehaze a single image
        
        Args:
            image_path: Path to input hazy image
            output_path: Path to save dehazed image
            calculate_metrics: Whether to calculate PSNR, SSIM, and UIQM metrics
        
        Returns:
            tuple: (enhanced_image, metrics_dict) if calculate_metrics=True, else enhanced_image
        """
        
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
       
        with torch.no_grad():
            output = self.model(img_tensor)
        
       
        output_img = self.to_pil(output.squeeze(0).cpu())
        output_img = output_img.resize(original_size, Image.LANCZOS)
        
        
        output_img.save(output_path)
        print(f'Saved dehazed image to {output_path}')
        
        if calculate_metrics:
            # Calculate metrics comparing original and enhanced images
            metrics = calculate_image_metrics(img, output_img)
            print(f'Metrics - PSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.4f}, UIQM: {metrics["uiqm"]:.4f}')
            return output_img, metrics
        else:
            return output_img
    
    def predict_batch(self, input_dir, output_dir, calculate_metrics=True):
        """
        Dehaze all images in a directory
        
        Args:
            input_dir: Directory containing hazy images
            output_dir: Directory to save dehazed images
            calculate_metrics: Whether to calculate metrics for each image
        
        Returns:
            list: List of metrics dictionaries if calculate_metrics=True, else None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f'Found {len(image_files)} images to process')
        
        all_metrics = []
        
        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f'dehazed_{img_file}')
            
            try:
                if calculate_metrics:
                    _, metrics = self.predict(input_path, output_path, calculate_metrics=True)
                    all_metrics.append(metrics)
                else:
                    self.predict(input_path, output_path, calculate_metrics=False)
            except Exception as e:
                print(f'Error processing {img_file}: {str(e)}')
        
        if calculate_metrics:
            return all_metrics
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Underwater Image Dehazing')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process entire directory')
    
    args = parser.parse_args()
    
    predictor = Predictor(args.model)
    
    if args.batch:
        predictor.predict_batch(args.input, args.output)
    else:
        predictor.predict(args.input, args.output)