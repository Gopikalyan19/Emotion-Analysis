import torch
import nltk
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class AdvancedEmotionDetector:
    def __init__(self, model_name="bhadresh-savani/bert-base-uncased-emotion"):
        """
        Initialize the emotion detection model with advanced configuration
        """
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt data...")
            nltk.download('punkt', quiet=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Comprehensive emotion labels
        self.emotion_labels = [
            # Primary emotions
            'joy', 'sadness', 'anger', 'fear', 
            'surprise', 'disgust', 'trust', 'anticipation',
            
            # Complex emotions
            'love', 'hope', 'pride', 'excitement', 
            'anxiety', 'loneliness', 'jealousy', 'guilt',
            'shame', 'relief', 'contentment', 'confusion',
            'nostalgia', 'frustration', 'enthusiasm', 'despair',
            
            # Nuanced emotional states
            'optimism', 'pessimism', 'compassion', 'curiosity',
            'wonder', 'gratitude', 'melancholy', 'contempt',
            'empathy', 'apathy', 'overwhelmed', 'serenity',
            
            # Neutral/Mixed states
            'neutral', 'uncertain', 'indifferent', 'conflicted'
        ]

    def preprocess_text(self, text):
        """
        Preprocess the input text with robust tokenization
        """
        text = text.strip()
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        return inputs

    def analyze_emotions(self, text, threshold=0.05):
        """
        Analyze emotions with nuanced detection and confidence scoring
        """
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = F.softmax(outputs.logits, dim=1)
        
        emotion_results = []
        for i, score in enumerate(predictions[0]):
            emotion = self.emotion_labels[i]
            confidence = score.item()
            
            if confidence >= threshold:
                emotion_results.append({
                    'emotion': emotion,
                    'confidence': confidence
                })
        
        emotion_results.sort(key=lambda x: x['confidence'], reverse=True)
        return emotion_results

    def analyze_multi_sentence(self, text, per_sentence=False, threshold=0.05):
        """
        Analyze emotions across multiple sentences
        """
        if per_sentence:
            try:
                sentences = nltk.sent_tokenize(text)
            except Exception:
                sentences = text.split('.')
        else:
            sentences = [text]
        
        results = {
            'overall_emotions': self.analyze_emotions(text, threshold),
            'sentence_emotions': []
        }
        
        if per_sentence:
            for sentence in sentences:
                if sentence.strip():
                    sentence_emotions = self.analyze_emotions(sentence, threshold)
                    results['sentence_emotions'].append({
                        'sentence': sentence,
                        'emotions': sentence_emotions
                    })
        
        return results

def main():
    """
    Main function for interactive emotion detection
    """
    print("Advanced Real-time Emotion Analysis: Type 'exit' to quit.\n")
    
    emotion_detector = AdvancedEmotionDetector()
    
    while True:
        user_input = input("Enter a text or paragraph for emotion detection: ")
        
        if user_input.lower() == "exit":
            print("Exiting the emotion analysis.")
            break
        
        try:
            results = emotion_detector.analyze_multi_sentence(
                user_input, 
                per_sentence=True, 
                threshold=0.05
            )
            
            print("\nOverall Detected Emotions:")
            for emotion in results['overall_emotions']:
                print(f"{emotion['emotion'].capitalize()}: {emotion['confidence']:.4f}")
            
            if results['sentence_emotions']:
                print("\nSentence-level Emotions:")
                for sentence_data in results['sentence_emotions']:
                    print(f"\nSentence: {sentence_data['sentence']}")
                    for emotion in sentence_data['emotions']:
                        print(f"  {emotion['emotion'].capitalize()}: {emotion['confidence']:.4f}")
            
            print("\n")
        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()