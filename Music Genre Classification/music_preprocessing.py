import os
import json
import librosa
import math
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

MUSIC_DIR = 'GTZAN'
JSON_PATH = 'music_data.json'
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def create_mfcc(src=MUSIC_DIR, dst=JSON_PATH, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10, save_file=True):
    data = {'mapping': [], 'mfcc': [], 'labels': []}
    n_samples_per_segment = SAMPLES_PER_TRACK // num_segments
    n_mfcc_per_segment = math.ceil(n_samples_per_segment / hop_length)
    for idx, (dir_path, dir_names, filenames) in enumerate(os.walk(src)):
        if dir_path is not src:
            print('Processing:', dir_path.split('\\')[-1])
            data['mapping'].append(dir_path.split('\\')[-1])

            for f in tqdm(filenames):
                try:
                    signal, sample_rate = librosa.load(os.path.join(dir_path, f), sr=SAMPLE_RATE)
                    for segment in range(num_segments):
                        start = n_samples_per_segment * segment
                        end = start + n_samples_per_segment
                        mfcc = librosa.feature.mfcc(signal[start:end],
                                                    sr=sample_rate,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length).T
                        if len(mfcc) == n_mfcc_per_segment:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(idx-1)
                except:
                    print('File loading failed:', f)
                    pass
    if save_file:
        with open(dst, 'w') as f:
            json.dump(data, f, indent=4)

    return data


def split_data(json_file=JSON_PATH, test_size=0.3, validation_set=True):
    with open(json_file) as f:
        data = json.load(f)
    X = np.array(data['mfcc'])
    y = np.array(data['labels']).reshape((-1, 1))
    y = OneHotEncoder().fit_transform(y).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    if validation_set:
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
        return X_train[..., np.newaxis], X_valid[..., np.newaxis], X_test[..., np.newaxis], y_train, y_valid, y_test
    else:
        return X_train[..., np.newaxis], X_test[..., np.newaxis], y_train, y_test