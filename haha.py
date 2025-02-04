import speech_recognition as sr
from langdetect import detect, DetectorFactory
from transformers import pipeline
from gtts import gTTS
import os
import matplotlib.pyplot as plt
import json
from googletrans import Translator

# Initialize Recognizer
recognizer = sr.Recognizer()

# Load Emotion Detection Model
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

# Initialize Translator for Language Fallback
translator = Translator()

# Supported Languages
SUPPORTED_LANGUAGES = {"en", "hi", "te"}

# Function to Record Speech and Convert to Text
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            if not text.strip():  # Check if text is empty or contains only whitespace
                raise sr.UnknownValueError
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Error connecting to speech service."

# Function to Detect Language with Fallback
def detect_language(text):
    try:
        lang = detect(text)
        if lang not in SUPPORTED_LANGUAGES:
            print(f"Unsupported language detected: {lang}. Defaulting to English.")
            return "en"  # Default to English if language is unsupported
        return lang
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Default to English if detection fails

# Function to Analyze Emotions
def analyze_emotion(text):
    try:
        emotions = emotion_model(text)
        emotion_scores = {e['label']: e['score'] for e in emotions[0]}
        return emotion_scores
    except Exception as e:
        print(f"Emotion analysis failed: {e}")
        return {"neutral": 1.0}  # Default to neutral emotion if analysis fails

# Function to Generate Response Based on Emotion
def generate_response(emotion_scores, lang):
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    
    responses = {
        "joy": {
            "en": "That's great! Keep enjoying your day! 😊",
            "hi": "बहुत बढ़िया! अपने दिन का आनंद लें! 😊",
            "te": "అద్భుతం! మీ రోజును ఆనందించండి! 😊"
        },
        "sadness": {
            "en": "I'm here for you. Want to talk about it?",
            "hi": "मैं आपके लिए हूँ। क्या आप इस बारे में बात करना चाहते हैं?",
            "te": "నేను మీకు తోడుగా ఉన్నాను. మీరు దీనిని గురించి మాట్లాడాలనుకుంటున్నారా?"
        },
        "anger": {
            "en": "I understand. Take a deep breath. Want some relaxation tips?",
            "hi": "मैं समझता हूँ। गहरी सांस लें। क्या आपको आराम की कोई सलाह चाहिए?",
            "te": "నేను అర్థం చేసుకున్నాను. లోతుగా శ్వాస తీసుకోండి. మీరు కొన్ని విశ్రాంతి చిట్కాలను కావాలా?"
        },
        "fear": {
            "en": "It's okay to feel afraid. You're not alone.",
            "hi": "डर लगना सामान्य है। आप अकेले नहीं हैं।",
            "te": "భయపడటం సర్వసాధారణం. మీరు ఒంటరిగా లేరు."
        }
    }
    
    # Default to English if language is not supported
    response = responses.get(dominant_emotion, {}).get(lang, responses.get(dominant_emotion, {}).get("en", "I'm here to help!"))
    
    return response

# Function to Speak Response
def speak_response(response_text, lang):
    try:
        tts = gTTS(text=response_text, lang=lang)
        tts.save("response.mp3")
        os.system("start response.mp3")  # Open audio file
    except Exception as e:
        print(f"Text-to-Speech failed: {e}")

# Function to Plot Emotions
def plot_emotion(emotion_scores):
    try:
        plt.figure(figsize=(6, 4))
        plt.bar(emotion_scores.keys(), emotion_scores.values(), color='skyblue')
        plt.xlabel("Emotions")
        plt.ylabel("Confidence Score")
        plt.title("Real-Time Emotion Detection")
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

# Function to Save Emotions
def save_emotions(text, emotion_scores):
    try:
        data = {"text": text, "emotions": emotion_scores}
        with open("emotion_log.json", "a") as file:
            json.dump(data, file)
            file.write("\n")
    except Exception as e:
        print(f"Saving emotions failed: {e}")

# Main Loop
def main():
    while True:
        text = recognize_speech()
        if text.lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break
        
        print(f"Detected Text: {text}")
        
        detected_lang = detect_language(text)
        print(f"Detected Language: {detected_lang}")
        
        emotion_results = analyze_emotion(text)
        print("Emotion Scores:", emotion_results)
        
        response_text = generate_response(emotion_results, detected_lang)
        print(f"Bot Response: {response_text}")
        
        speak_response(response_text, detected_lang)
        plot_emotion(emotion_results)
        save_emotions(text, emotion_results)

if __name__ == "__main__":
    main()