import os
import logging
import speech_recognition as sr
import pyttsx3
import threading
import queue
from transformers import pipeline
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon', quiet=True)

class EmotionAwareChatbot:
    def __init__(self):
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 160)
        self.tts_engine.setProperty("volume", 0.9)
        
        # Emotion Analysis Pipeline
        self.emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
        
        # Logging Setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Conversation Context
        self.conversation_history = []
        self.max_history = 5
        
        # Welcome Message
        self.welcome_message = (
            "Hi there! I'm an emotion-aware chatbot. "
            "I'm here to listen and understand your feelings. "
            "Speak freely, and I'll help you explore your emotions."
        )
    
    def recognize_speech(self):
        """Advanced speech recognition with error handling."""
        try:
            with self.microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio).strip()
            self.logger.info(f"Recognized Speech: {text}")
            return text
        except sr.WaitTimeoutError:
            self.logger.warning("No speech detected within timeout.")
            return None
        except sr.UnknownValueError:
            self.logger.error("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
    
    def analyze_emotions(self, text):
        """Simple emotion analysis method."""
        try:
            results = self.emotion_pipeline(text)
            emotions = {res["label"]: res["score"] for res in results}
            return emotions
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return {}
    
    def generate_response(self, emotions):
        """Generate a response based on detected emotions."""
        if not emotions:
            return "I'm having trouble understanding your emotions."
        
        dominant_emotion = max(emotions, key=emotions.get)
        
        response_map = {
            "joy": "Your happiness is contagious! What's making you so joyful?",
            "sadness": "I'm here to support you through difficult times.",
            "anger": "I understand you're feeling frustrated. Let's take a deep breath.",
            "fear": "Your feelings of fear are understandable. You're not alone.",
            "love": "Love is a beautiful emotion. Tell me more about what you're experiencing."
        }
        
        return response_map.get(dominant_emotion, "Thank you for sharing your feelings with me.")
    
    def speak_response(self, response):
        """Text-to-speech output."""
        try:
            self.logger.info(f"Response: {response}")
            print(f"Bot: {response}")
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Speech output error: {e}")
    
    def run(self):
        """Main chatbot interaction loop."""
        # Speak welcome message
        self.speak_response(self.welcome_message)
        
        print("Emotion-Aware Chatbot: Say 'exit' to quit.\n")
        
        try:
            while True:
                user_input = self.recognize_speech()
                
                if not user_input:
                    self.speak_response("I didn't catch that. Could you repeat?")
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    self.speak_response("Goodbye! Take care of yourself.")
                    break
                
                # Analyze emotions and generate response
                emotions = self.analyze_emotions(user_input)
                response = self.generate_response(emotions)
                self.speak_response(response)
        
        except KeyboardInterrupt:
            print("\nChatbot stopped by user.")
        finally:
            self.logger.info("Chatbot session ended.")

def main():
    chatbot = EmotionAwareChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()