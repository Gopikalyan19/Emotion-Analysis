import speech_recognition as sr
import librosa
import numpy as np
from transformers import pipeline
import pyaudio
import traceback

# Load pre-trained emotion classification model
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Function to process and extract audio features using librosa
def extract_features_from_audio(audio_data):
    try:
        # Load audio with librosa
        audio, sampling_rate = librosa.load(audio_data, sr=None)
        
        # Extract MFCC features from the audio
        mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
        
        # Take the mean of the MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

# Function to detect emotion from text using pre-trained model
def detect_emotion_from_text(text):
    try:
        result = emotion_pipeline(text)
        emotion_results = [(res['label'], res['score']) for res in result]
        return emotion_results
    except Exception as e:
        print(f"Error detecting emotion from text: {e}")
        return []

# Function to recognize speech in real-time
def recognize_speech_from_microphone():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Capture microphone input
    with sr.Microphone() as source:
        print("Listening for your voice...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("No speech detected. Listening timed out.")
            return None
    
    try:
        # Recognize speech using Google's speech recognition
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Real-time emotion analysis loop
def main():
    print("Real-time Multilingual Emotion Recognition: Type 'exit' to quit.\n")
    try:
        while True:
            # Capture speech from the user
            user_input = recognize_speech_from_microphone()
            
            if user_input and user_input.lower() == "exit":
                print("Exiting the emotion recognition.")
                break
            
            # If speech is recognized, process the text to detect emotion
            if user_input:
                emotion_results = detect_emotion_from_text(user_input)
                
                if emotion_results:
                    print("Detected emotions and their confidence scores:")
                    for emotion, score in emotion_results:
                        print(f"{emotion}: {score:.4f}")
                else:
                    print("No emotions could be detected.")
                
            print("\n")  # Add space between iterations for better readability
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()