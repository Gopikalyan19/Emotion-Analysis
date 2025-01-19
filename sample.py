from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Simple emotion detection based on keyword matching
emotion_keywords = {
    'joy': ['happy', 'joy', 'delighted', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'awesome'],
    'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'hurt', 'disappointed', 'sorry', 'lost'],
    'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'hate'],
    'fear': ['afraid', 'scared', 'worried', 'nervous', 'terrified', 'anxious', 'fear'],
    'surprise': ['wow', 'surprised', 'amazed', 'shocked', 'unexpected', 'incredible']
}

def analyze_emotion(text):
    text = text.lower()
    scores = {emotion: 0 for emotion in emotion_keywords}
    
    # Count occurrences of emotion keywords
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', text))
            scores[emotion] += count
    
    # Find the emotion with highest score
    max_emotion = max(scores.items(), key=lambda x: x[1])
    
    # If no emotion detected, return neutral
    if max_emotion[1] == 0:
        return 'neutral', 0.5
    
    # Calculate simple confidence score (0.5 - 1.0)
    total_matches = sum(scores.values())
    confidence = 0.5 + min((max_emotion[1] / total_matches) * 0.5, 0.5)
    
    return max_emotion[0], confidence

@app.route('/')
def home():
    return render_template('text-based.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        emotion, confidence = analyze_emotion(text)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)