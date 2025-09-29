# check_val_mismatch.py
import os

hazy_dir = 'datasets/val/hazy'
clean_dir = 'datasets/val/clean'

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

hazy_files = sorted([f for f in os.listdir(hazy_dir) 
                     if f.lower().endswith(valid_extensions)])
clean_files = sorted([f for f in os.listdir(clean_dir) 
                      if f.lower().endswith(valid_extensions)])

print("First 10 HAZY files:")
for f in hazy_files[:10]:
    print(f"  {f}")

print("\nFirst 10 CLEAN files:")
for f in clean_files[:10]:
    print(f"  {f}")

print(f"\nTotal hazy: {len(hazy_files)}")
print(f"Total clean: {len(clean_files)}")

# Check for matches
hazy_set = set(hazy_files)
clean_set = set(clean_files)
matches = hazy_set & clean_set

print(f"\nMatching files: {len(matches)}")

if len(matches) == 0:
    print("\n❌ NO MATCHES! The filenames are completely different.")
    print("\nSample hazy names:", hazy_files[:3])
    print("Sample clean names:", clean_files[:3])