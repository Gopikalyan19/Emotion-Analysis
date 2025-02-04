import os
import logging
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import numpy as np

class EmotionAwareChatbot:
    def __init__(self):
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 160)
        self.tts_engine.setProperty("volume", 0.9)
        
        # Multiple Emotion Analysis Pipelines
        self.emotion_pipelines = [
            pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion"),
            pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        ]
        
        # Expanded Emotion Mapping
        self.emotion_mapping = {
            "joy": ["happiness", "joy", "excitement"],
            "sadness": ["sadness", "grief", "melancholy"],
            "anger": ["anger", "frustration", "rage"],
            "fear": ["fear", "anxiety", "worry"],
            "love": ["love", "affection", "compassion"],
            "surprise": ["surprise", "shock", "amazement"],
            "disgust": ["disgust", "repulsion", "aversion"],
            "neutral": ["neutral", "calm", "indifference"]
        }
        
        # Logging Setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def analyze_emotions(self, text):
        """Advanced multi-model emotion analysis."""
        try:
            # Collect results from multiple emotion detection models
            all_results = []
            for pipeline in self.emotion_pipelines:
                all_results.extend(pipeline(text))
            
            # Aggregate and normalize emotions
            emotion_scores = {}
            for result in all_results:
                emotion = result['label']
                score = result['score']
                
                # Map to primary emotions
                primary_emotion = self._map_to_primary_emotion(emotion)
                emotion_scores[primary_emotion] = max(
                    emotion_scores.get(primary_emotion, 0), 
                    score
                )
            
            # Normalize percentages
            total_score = sum(emotion_scores.values())
            emotion_percentages = {
                emotion: round((score / total_score) * 100, 2)
                for emotion, score in emotion_scores.items()
            }
            
            # Sort and print emotion breakdown
            print("\n--- Emotion Analysis ---")
            for emotion, percentage in sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True):
                print(f"{emotion.capitalize()}: {percentage}%")
            
            # Identify dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_percentage = emotion_percentages[dominant_emotion]
            print(f"\nDominant Emotion: {dominant_emotion.capitalize()} ({dominant_percentage}%)")
            
            return {
                "emotions": emotion_scores,
                "percentages": emotion_percentages,
                "dominant_emotion": dominant_emotion,
                "dominant_percentage": dominant_percentage
            }
        
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return {}
    
    def _map_to_primary_emotion(self, emotion):
        """Map detected emotions to primary emotion categories."""
        for primary, variants in self.emotion_mapping.items():
            if emotion.lower() in variants:
                return primary
        return "neutral"
    
    def generate_response(self, emotion_analysis):
        """Generate nuanced response based on emotion analysis."""
        if not emotion_analysis:
            return "I couldn't understand your emotions."
        
        dominant_emotion = emotion_analysis['dominant_emotion']
        dominant_percentage = emotion_analysis['dominant_percentage']
        
        response_map = {
            "joy": "Your happiness radiates at {}%. What's lighting up your world?",
            "sadness": "I sense your emotional depth at {}%. Would you like to talk more?",
            "anger": "Your feelings of frustration are valid at {}%. How can we process this?",
            "fear": "Anxiety levels at {}%. Remember, you have inner strength.",
            "love": "Your capacity for love shines at {}%. Beautiful emotion.",
            "surprise": "Unexpected feelings at {}%. What's unexpected?",
            "disgust": "Complex emotions at {}%. Would you like to explore them?",
            "neutral": "A balanced state at {}%. Interesting emotional landscape."
        }
        
        return response_map.get(
            dominant_emotion, 
            "Thank you for sharing your feelings."
        ).format(dominant_percentage)
    
    def run(self):
        """Main chatbot interaction loop."""
        print("Emotion-Aware Chatbot: Type 'exit' to quit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Analyze emotions
            emotion_analysis = self.analyze_emotions(user_input)
            
            # Generate and print response
            response = self.generate_response(emotion_analysis)
            print("Bot:", response)

def main():
    chatbot = EmotionAwareChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()