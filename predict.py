import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import UNet
import argparse

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
    
    def predict(self, image_path, output_path):
        """
        Dehaze a single image
        
        Args:
            image_path: Path to input hazy image
            output_path: Path to save dehazed image
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
        
        return output_img
    
    def predict_batch(self, input_dir, output_dir):
        """
        Dehaze all images in a directory
        
        Args:
            input_dir: Directory containing hazy images
            output_dir: Directory to save dehazed images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f'Found {len(image_files)} images to process')
        
        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f'dehazed_{img_file}')
            
            try:
                self.predict(input_path, output_path)
            except Exception as e:
                print(f'Error processing {img_file}: {str(e)}')

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