
import os

def check_dataset():
    paths = [
        ('datasets/train/clean', 'datasets/train/hazy'),
        ('datasets/val/clean', 'datasets/val/hazy')
    ]
    
    for clean_path, hazy_path in paths:
        print(f"\n{'='*60}")
        print(f"Checking: {clean_path} vs {hazy_path}")
        print('='*60)
        
        clean_images = sorted(os.listdir(clean_path))
        hazy_images = sorted(os.listdir(hazy_path))
        
        print(f"Clean images: {len(clean_images)}")
        print(f"Hazy images: {len(hazy_images)}")
        
        if len(clean_images) != len(hazy_images):
            print("❌ MISMATCH!")
            
            
            clean_set = set(clean_images)
            hazy_set = set(hazy_images)
            
            missing_in_hazy = clean_set - hazy_set
            missing_in_clean = hazy_set - clean_set
            
            if missing_in_hazy:
                print(f"\n⚠️ Files in clean but NOT in hazy:")
                for f in missing_in_hazy:
                    print(f"  - {f}")
            
            if missing_in_clean:
                print(f"\n⚠️ Files in hazy but NOT in clean:")
                for f in missing_in_clean:
                    print(f"  - {f}")
        else:
            print("✅ Counts match!")
            
           
            if clean_images == hazy_images:
                print("✅ Filenames match perfectly!")
            else:
                print("⚠️ Counts match but filenames are different")
                print("\nFirst 5 clean files:", clean_images[:5])
                print("First 5 hazy files:", hazy_images[:5])

if __name__ == '__main__':
    check_dataset()