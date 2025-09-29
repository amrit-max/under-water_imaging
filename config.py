import os

class Config:
    """Configuration class for underwater image dehazing project"""
    
  
    DATA_DIR = 'datasets'
    CHECKPOINT_DIR = 'checkpoints'
    UPLOAD_DIR = 'uploads'
    OUTPUT_DIR = 'outputs'
    
   
    IMG_SIZE = 128
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    
    
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    NUM_WORKERS = 4
    
   
    DEVICE = 'cuda'  
    
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
    @staticmethod
    def get_train_config():
        """Get training configuration dictionary"""
        return {
            'data_dir': Config.DATA_DIR,
            'checkpoint_dir': Config.CHECKPOINT_DIR,
            'img_size': Config.IMG_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'lr': Config.LEARNING_RATE,
            'num_workers': Config.NUM_WORKERS
        }