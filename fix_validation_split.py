
import os
import shutil
import random

print("Creating proper train/val split...")


train_hazy = 'datasets/train/hazy'
train_clean = 'datasets/train/clean'
val_hazy = 'datasets/val/hazy'
val_clean = 'datasets/val/clean'


backup_dir = 'datasets/val_backup'
os.makedirs(backup_dir, exist_ok=True)

print("\n1. Backing up old validation data...")
for folder_name in ['hazy', 'clean']:
    src_folder = f'datasets/val/{folder_name}'
    dst_folder = f'{backup_dir}/{folder_name}'
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    shutil.copytree(src_folder, dst_folder)
    print(f"   Backed up val/{folder_name} to {dst_folder}")


print("\n2. Clearing validation folders...")
for f in os.listdir(val_hazy):
    os.remove(os.path.join(val_hazy, f))
for f in os.listdir(val_clean):
    os.remove(os.path.join(val_clean, f))


valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

train_hazy_files = [f for f in os.listdir(train_hazy) 
                    if f.lower().endswith(valid_extensions)]
train_clean_files = [f for f in os.listdir(train_clean) 
                     if f.lower().endswith(valid_extensions)]


hazy_set = set(train_hazy_files)
clean_set = set(train_clean_files)
matching_files = sorted(hazy_set & clean_set)

print(f"\n3. Found {len(matching_files)} matching pairs in training data")


random.seed(42)
random.shuffle(matching_files)

val_size = int(len(matching_files) * 0.2)  
val_files = matching_files[:val_size]


print(f"\n4. Moving {val_size} pairs to validation...")

moved_count = 0
for filename in val_files:
    try:
        
        src_hazy = os.path.join(train_hazy, filename)
        dst_hazy = os.path.join(val_hazy, filename)
        shutil.move(src_hazy, dst_hazy)
        
       
        src_clean = os.path.join(train_clean, filename)
        dst_clean = os.path.join(val_clean, filename)
        shutil.move(src_clean, dst_clean)
        
        moved_count += 1
        if moved_count % 50 == 0:
            print(f"   Moved {moved_count}/{val_size} pairs...")
    except Exception as e:
        print(f"   Error moving {filename}: {e}")

print(f"\nSplit complete!")
print(f"   Training set: {len(matching_files) - val_size} pairs")
print(f"   Validation set: {moved_count} pairs")
print(f"\n   Old validation data backed up to: {backup_dir}")