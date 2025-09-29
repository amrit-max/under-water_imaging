import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import UNet
from dataset import UnderwaterDataset, get_transforms
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.model = UNet().to(self.device)
        
       
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        
        train_dataset = UnderwaterDataset(
            config['data_dir'], 
            mode='train', 
            transform=get_transforms(config['img_size'])
        )
        val_dataset = UnderwaterDataset(
            config['data_dir'], 
            mode='val', 
            transform=get_transforms(config['img_size'])
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=config['num_workers']
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for hazy, clean in pbar:
            hazy, clean = hazy.to(self.device), clean.to(self.device)
            
            
            output = self.model(hazy)
            loss = self.criterion(output, clean)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for hazy, clean in tqdm(self.val_loader, desc='Validating'):
                hazy, clean = hazy.to(self.device), clean.to(self.device)
                
                output = self.model(hazy)
                loss = self.criterion(output, clean)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                )
                print('Best model saved!')
            
            
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.config['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
                )
        
       
        self.plot_history()
    
    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(os.path.join(self.config['checkpoint_dir'], 'training_history.png'))
        plt.close()

if __name__ == '__main__':
    config = {
        'data_dir': 'datasets',
        'checkpoint_dir': 'checkpoints',
        'img_size': 128,        
        'batch_size': 16,       
        'epochs': 20,           
        'lr': 0.0001,
        'num_workers': 4
    }
    
    trainer = Trainer(config)
    trainer.train()