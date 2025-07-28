from src.audio_processing import process_dataset

audio_dir = "./data/audio"
output_dir = "./data/features/audio_mfcc"

process_dataset(audio_dir, output_dir)
