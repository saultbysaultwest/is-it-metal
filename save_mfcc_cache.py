# save_mfcc_cache.py

import os
import torch
import torchaudio
from preprocess_dataset import preprocess_audio

def find_max_length(raw_roots):
    """Find the maximum MFCC length across all files to determine padding target"""
    print("Finding maximum MFCC length across all files...")
    max_length = 0
    total_files = 0
    
    for split in raw_roots:
        for genre in ['non_metal', 'metal']:
            input_dir = os.path.join(split, genre)
            if not os.path.exists(input_dir):
                continue
                
            for fname in os.listdir(input_dir):
                if not fname.endswith('.wav'):
                    continue
                    
                path = os.path.join(input_dir, fname)
                try:
                    mfcc = preprocess_audio(path)
                    current_length = mfcc.shape[-1]  # Last dimension is time
                    max_length = max(max_length, current_length)
                    total_files += 1
                    
                    if total_files % 50 == 0:  # Progress update
                        print(f"Processed {total_files} files, current max length: {max_length}")
                        
                except Exception as e:
                    print(f"Error processing {path}: {e}")
    
    print(f"Found maximum length: {max_length} across {total_files} files")
    return max_length

def pad_mfcc(mfcc, target_length):
    """Pad MFCC to target length with zeros"""
    current_length = mfcc.shape[-1]
    if current_length < target_length:
        pad_amount = target_length - current_length
        # Pad the last dimension (time) with zeros
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
    elif current_length > target_length:
        # Truncate if somehow longer (shouldn't happen after finding max)
        mfcc = mfcc[:, :, :target_length]
    return mfcc

RAW_ROOTS = ['dataset', 'valset', 'testset']  # raw WAV folders
CACHE_ROOT = 'mfcc_cache'  # where to save the MFCC files

# First pass: find the maximum length
max_length = find_max_length(RAW_ROOTS)

# Add some buffer just in case
target_length = max_length + 50
print(f"-> Target length set to: {target_length}")

os.makedirs(CACHE_ROOT, exist_ok=True)

# Second pass: process and pad all files
for split in RAW_ROOTS:
    print(f'Processing: {split}')
    for genre in ['non_metal', 'metal']:
        input_dir = os.path.join(split, genre)
        output_dir = os.path.join(CACHE_ROOT, split, genre)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            continue

        for fname in os.listdir(input_dir):
            if not fname.endswith('.wav'):
                continue

            path = os.path.join(input_dir, fname)
            try:
                # Process original file
                mfcc = preprocess_audio(path)
                
                # Pad to target length
                mfcc_padded = pad_mfcc(mfcc, target_length)
                
                # Verify consistent shape
                assert mfcc_padded.shape[-1] == target_length, f"Padding failed for {fname}"
                
                # Save with consistent naming
                outname = os.path.splitext(fname)[0] + '.pt'
                torch.save({
                    'mfcc': mfcc_padded, 
                    'label': int(genre == 'metal'), 
                    'path': path,
                    'original_length': mfcc.shape[-1]  # Store original length for reference
                }, os.path.join(output_dir, outname))
                
                print(f"Saved: {outname} (padded from {mfcc.shape[-1]} to {target_length})")

            except Exception as e:
                print(f"Error processing {path}: {e}")

print("[Success] All files processed with consistent padding!")
