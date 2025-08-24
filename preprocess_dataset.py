# preprocess_dataset.py

import os
import torch
import torchaudio
from torch.utils.data import Dataset

# === Constants ===
SAMPLE_RATE = 22050
N_MFCC = 16
N_FFT = 512
HOP_LENGTH = 128
TARGET_DURATION = 15  # 15 seconds
TARGET_LEN = SAMPLE_RATE * TARGET_DURATION  # 15 seconds of audio samples

# === MFCC Transform ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': 40,
        'center': True,
        'power': 2.0,
    }
)

def calculate_mfcc_length(audio_samples):
    """Calculate expected MFCC time dimension given audio sample count"""
    return (audio_samples + HOP_LENGTH - 1) // HOP_LENGTH

# Expected MFCC time frames for 15 seconds
EXPECTED_MFCC_FRAMES = calculate_mfcc_length(TARGET_LEN)
print(f"Expected MFCC frames for {TARGET_DURATION}s audio: {EXPECTED_MFCC_FRAMES}")

def preprocess_audio(path=None, waveform=None):
    """
    Process audio to exactly 15 seconds
    
    Args:
        path: Audio file path
        waveform: Pre-loaded waveform (optional)
    """
    if waveform is None:
        waveform, sr = torchaudio.load(path)
    else:
        sr = SAMPLE_RATE

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    current_len = waveform.shape[1]
    
    if current_len > TARGET_LEN:
        # Crop to 15 seconds
        start_idx = 0
        waveform = waveform[:, start_idx:start_idx + TARGET_LEN]
        
    elif current_len < TARGET_LEN:
        # Pad shorter audio
        pad_amount = TARGET_LEN - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    
    # Ensure exact length
    waveform = waveform[:, :TARGET_LEN]
    
    # Convert to MFCC
    mfcc = mfcc_transform(waveform)
    
    # Verify expected shape
    expected_shape = (1, N_MFCC, EXPECTED_MFCC_FRAMES)
    if mfcc.shape != expected_shape:
        print(f"Warning: MFCC shape {mfcc.shape} != expected {expected_shape}")
    
    return mfcc

class MetalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        
        for label, genre in enumerate(['non_metal', 'metal']):
            genre_path = os.path.join(root_dir, genre)
            if not os.path.exists(genre_path):
                continue
            for fname in os.listdir(genre_path):
                if fname.endswith('.wav'):
                    path = os.path.join(genre_path, fname)
                    self.samples.append((path, label))
        
        print(f"Found {len(self.samples)} samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc = preprocess_audio(path)
        return mfcc, torch.tensor(label, dtype=torch.float32)