import os
import shutil

# Path where downloaded audio files are stored
SOURCE_DIR = "data/audio_raw"  # Your original CREMA-D folder
DEST_DIR = "data/audio"        # Where you want sorted files

# Emotion mapping (3-letter codes to folder names)
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# Create destination folders
for emotion in emotion_map.values():
    os.makedirs(os.path.join(DEST_DIR, emotion), exist_ok=True)

# Move files based on emotion
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".wav"):
        parts = filename.split("_")
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                src_path = os.path.join(SOURCE_DIR, filename)
                dest_path = os.path.join(DEST_DIR, emotion_map[emotion_code], filename)
                shutil.move(src_path, dest_path)

print("Dataset organized by emotion in 'data/audio/'")
