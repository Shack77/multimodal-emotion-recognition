import os
import librosa
import numpy as np
import soundfile as sf

def extract_mfcc(file_path, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
    
    # Original
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])

    return pad_or_truncate(combined.T, max_len)

def augment_audio(y, sr):
    y_noise = y + 0.005 * np.random.randn(len(y))
    y_stretch = librosa.effects.time_stretch(y, rate=0.9)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    return [y_noise, y_stretch, y_pitch]

def pad_or_truncate(mfcc, max_len):
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

def process_dataset(audio_dir, output_dir, max_len=100):
    emotions = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    os.makedirs(output_dir, exist_ok=True)

    for emotion in emotions:
        emotion_path = os.path.join(audio_dir, emotion)
        out_emotion_path = os.path.join(output_dir, emotion)
        os.makedirs(out_emotion_path, exist_ok=True)

        for file in os.listdir(emotion_path):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(emotion_path, file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                # Original
                mfcc = extract_mfcc(file_path, max_len)
                np.save(os.path.join(out_emotion_path, file.replace(".wav", "_orig.npy")), mfcc)

                # Augmentations
                for i, aug_y in enumerate(augment_audio(y, sr)):
                    mfcc_aug = librosa.feature.mfcc(y=aug_y, sr=sr, n_mfcc=40)
                    delta = librosa.feature.delta(mfcc_aug)
                    delta2 = librosa.feature.delta(mfcc_aug, order=2)
                    combined = np.vstack([mfcc_aug, delta, delta2])
                    combined = pad_or_truncate(combined.T, max_len)
                    aug_name = file.replace(".wav", f"_aug{i}.npy")
                    np.save(os.path.join(out_emotion_path, aug_name), combined)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
