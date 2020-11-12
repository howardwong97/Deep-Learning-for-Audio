import json
import os
import librosa
import noisereduce as nr
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.models import Sequential
import numpy as np

SRC_DIR = 'train/audio'
LABELS = os.listdir(SRC_DIR)


SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def create_mfcc(src=SRC_DIR, dst='speech_data.json', n_mfcc=13, n_fft=2048, hop_length=512, pad=True, save_file=True):
    data = {'mappings': [], 'mfccs': [], 'labels': []}
    for idx, (dir_path, dir_names, filenames) in enumerate(os.walk(src)):
        if dir_path is not src:
            print('Processing:', dir_path.split('\\')[-1])
            data['mappings'].append(dir_path.split('\\')[-1])

            for f in filenames:
                try:
                    signal, sample_rate = librosa.load(os.path.join(dir_path, f), sr=SAMPLE_RATE)
                    signal_noise_reduced = nr.reduce_noise(audio_clip=signal, noise_clip=signal, verbose=False)
                    signal_trimmed, _ = librosa.effects.trim(signal_noise_reduced)
                    mfcc = librosa.feature.mfcc(
                        signal_trimmed.T,
                        sr=sample_rate,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length)
                    data['mfccs'].append(mfcc.tolist())
                    data['labels'].append(idx-1)
                except:
                    print('File loading failed:', f)
                    pass
    if save_file:
        with open(dst, 'w') as f:
            json.dump(data, f, indent=4)
    return data


def from_json(data, intended_shape=(13, 44)):
    padded_mfccs = []
    for i in data['mfccs']:
        padding_array = np.zeros(intended_shape)
        a = np.array(i)
        padding_array[:a.shape[0], :a.shape[1]] = a
        padded_mfccs.append(padding_array[np.newaxis, ...])
    X = np.vstack(padded_mfccs)
    X = X[..., np.newaxis]
    return X, np.array(data['labels']), np.array(data['mappings'])


def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(30, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
