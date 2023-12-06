import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    return np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(contrast, axis=1), tempo

def write_features_to_csv(filename, features, output_file, header=False):
    with open(output_file, 'a') as f:
        if header:
            column_names = ['filename'] + [f'mfcc_{i}' for i in range(20)] + \
                           [f'chroma_{i}' for i in range(12)] + \
                           [f'contrast_{i}' for i in range(7)] + ['tempo']
            f.write(','.join(column_names) + '\n')

        line = ','.join(map(str, [filename] + features))
        f.write(line + '\n')

def process_directory(directory_path, output_file):
    first_file = True
    for filename in tqdm(os.listdir(directory_path), desc="Processing Files"):
        print("Processing: " + filename)
        if filename.endswith('.mp3'):
            file_path = os.path.join(directory_path, filename)
            mfccs, chroma, contrast, tempo = extract_features(file_path)
            combined_features = np.hstack((mfccs, chroma, contrast, [tempo])).tolist()

            write_features_to_csv(filename, combined_features, output_file, header=first_file)
            first_file = False

def main():
    if len(sys.argv) != 3:
        print("Usage: python mp3_featurizer.py <directory_path> <output_file>")
        sys.exit(1)

    directory_path = sys.argv[1]
    output_file = sys.argv[2]

    # Explanation of features
    print("Welcome!")
    print("This program extracts the following features from each MP3 file:")
    print("1. MFCCs: Mel-Frequency Cepstral Coefficients, representing the short-term power spectrum of sound.")
    print("2. Chroma: Pertains to the 12 different pitch classes.")
    print("3. Spectral Contrast: Difference in amplitude between peaks and valleys in the sound spectrum.")
    print("4. Tempo: The speed or pace of the given piece.")
    print("")
    print("By Nezu Life Sciences, November 2023")
    print("")

    # Ensure the output file is empty before starting
    open(output_file, 'w').close()

    process_directory(directory_path, output_file)

    print("Feature extraction complete. Data saved to", output_file)

if __name__ == "__main__":
    main()
