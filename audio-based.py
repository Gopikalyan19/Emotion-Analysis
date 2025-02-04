import os
import sys
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_path):
    features, labels = [], []
    
    print(f"Searching for .wav files in: {data_path}")
    print(f"Directory contents: {os.listdir(data_path)}")
    
    # Recursively walk through directories
    for root, dirs, files in os.walk(data_path):
        print(f"\nChecking directory: {root}")
        print(f"Subdirectories: {dirs}")
        print(f"Files: {files}")
        
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                print(f"\nProcessing file: {file_path}")
                
                try:
                    # Print filename details for debugging
                    print(f"Filename parts: {filename.split('-')}")
                    
                    # RAVDESS dataset naming: Actor_03-01-03-01-01-01-01.wav
                    # Emotion is typically the 3rd element
                    emotion = filename.split('-')[2]
                    
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
                        print(f"Added feature for emotion: {emotion}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal features collected: {len(features)}")
    print(f"Total labels collected: {len(labels)}")
    
    return np.array(features), np.array(labels)

# Full path to dataset
DATA_PATH = r'C:\Users\gopib\OneDrive\Desktop\Final Project\RAVDESS_Dataset'

# Load data with extensive logging
X, y = load_data(DATA_PATH)

# If no data found, exit with detailed information
if len(X) == 0 or len(y) == 0:
    print("No valid audio files found!")
    sys.exit(1)

# Rest of the script remains the same as previous version
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

model.save('emotion_recognition_model.h5')
np.save('label_encoder_classes.npy', label_encoder.classes_)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy: {accuracy*100:.2f}%')
print("Emotion Labels:", list(label_encoder.classes_))