# process_directory.py

import os
import shutil
import torch
import torchaudio
import subprocess
import tempfile
from metal_classifier_optimized import MetalClassifier
from preprocess_dataset import preprocess_audio

# === Constants ===
SAMPLE_RATE = 22050
CLIP_LEN = 15 * SAMPLE_RATE
THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_mp3_to_waveform(mp3_path):
    """Convert MP3 to mono WAV using ffmpeg and return waveform."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", mp3_path,
        "-ac", "1",  # mono
        "-ar", str(SAMPLE_RATE),
        tmp_wav_path
    ]
    subprocess.run(command, check=True)
    waveform, sr = torchaudio.load(tmp_wav_path)
    os.remove(tmp_wav_path)
    return waveform, sr


def extract_three_clips(waveform):
    """Extract 3 clips from 25%, 50%, and 75% of the track length."""
    total_len = waveform.shape[1]
    if total_len < CLIP_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, CLIP_LEN - total_len))
        total_len = CLIP_LEN

    clips = []
    for pos in [0.25, 0.5, 0.75]:
        center = int(total_len * pos)
        start = max(0, center - CLIP_LEN // 2)
        end = start + CLIP_LEN
        if end > total_len:
            start = total_len - CLIP_LEN
            end = total_len
        clips.append(waveform[:, start:end])
    return clips


def predict_clip(model, waveform):
    """Run the model on a single waveform clip."""
    mfcc = preprocess_audio(waveform=waveform).to(DEVICE)
    with torch.no_grad():
        logits = model(mfcc.unsqueeze(0)).squeeze().item()
    return torch.sigmoid(torch.tensor(logits)).item()


def classify_mp3(model, mp3_path):
    """Classify one mp3 file using majority vote of 3 clips."""
    waveform, sr = convert_mp3_to_waveform(mp3_path)

    # Sanity: mono + resample already handled in ffmpeg
    clips = extract_three_clips(waveform)
    preds = [predict_clip(model, clip) > THRESHOLD for clip in clips]
    return sum(preds) >= 2


def process_directory(mp3_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'metal'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'non_metal'), exist_ok=True)

    model = MetalClassifier().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()

    for fname in os.listdir(mp3_dir):
        if not fname.lower().endswith('.mp3'):
            continue

        path = os.path.join(mp3_dir, fname)
        print(f"Classifying {fname}...")

        try:
            is_metal = classify_mp3(model, path)
            target = 'metal' if is_metal else 'non_metal'
            shutil.move(path, os.path.join(out_dir, target, fname))
            print(f"[Success] Moved to {target}")
        except Exception as e:
            print(f"[Error] Error with {fname}: {e}")


if __name__ == "__main__":
    process_directory("input_mp3", "sorted_mp3")
