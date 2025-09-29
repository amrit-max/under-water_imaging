import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import UnderwaterDataset, get_transforms
from utils import calculate_psnr, calculate_ssim, visualize_results
from tqdm import tqdm
import os
import argparse

class Evaluator:
    def __init__(self, model_path, data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        
        val_dataset = UnderwaterDataset(
            data_dir, 
            mode='val', 
            transform=get_transforms(256)
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        print(f'Loaded model from {model_path}')
        print(f'Validation set size: {len(val_dataset)}')
    
    def evaluate(self, save_samples=True, num_samples=5):
        """Evaluate model on validation set"""
        psnr_scores = []
        ssim_scores = []
        
        os.makedirs('evaluation_results', exist_ok=True)
        
        sample_count = 0
        
        with torch.no_grad():
            for i, (hazy, clean) in enumerate(tqdm(self.val_loader, desc='Evaluating')):
                hazy = hazy.to(self.device)
                clean = clean.to(self.device)
                
               
                output = self.model(hazy)
                
                
                psnr = calculate_psnr(output, clean)
                ssim = calculate_ssim(output, clean)
                
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)
                
                
                if save_samples and sample_count < num_samples:
                    save_path = f'evaluation_results/sample_{i+1}.png'
                    visualize_results(
                        hazy[0], 
                        clean[0], 
                        output[0], 
                        save_path=save_path
                    )
                    sample_count += 1
        
        
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        
        print('\n' + '='*50)
        print('EVALUATION RESULTS')
        print('='*50)
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')
        print(f'Min PSNR: {min(psnr_scores):.2f} dB')
        print(f'Max PSNR: {max(psnr_scores):.2f} dB')
        print(f'Min SSIM: {min(ssim_scores):.4f}')
        print(f'Max SSIM: {max(ssim_scores):.4f}')
        print('='*50)
        
        
        with open('evaluation_results/metrics.txt', 'w') as f:
            f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
            f.write(f'Average SSIM: {avg_ssim:.4f}\n')
            f.write(f'Min PSNR: {min(psnr_scores):.2f} dB\n')
            f.write(f'Max PSNR: {max(psnr_scores):.2f} dB\n')
            f.write(f'Min SSIM: {min(ssim_scores):.4f}\n')
            f.write(f'Max SSIM: {max(ssim_scores):.4f}\n')
        
        return {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'psnr_scores': psnr_scores,
            'ssim_scores': ssim_scores
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Underwater Image Dehazing Model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='datasets',
                       help='Path to dataset directory')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of sample visualizations to save')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(args.model, args.data)
    evaluator.evaluate(save_samples=True, num_samples=args.samples)