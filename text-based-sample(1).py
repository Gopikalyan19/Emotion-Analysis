from transformers import pipeline

# Load a pre-trained model for more extensive emotion detection
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Function to analyze emotions in text and return multiple emotions
def analyze_emotions(text):
    result = emotion_pipeline(text)
    emotions = []
    
    # Process the results to group the emotions
    for res in result:
        emotion = res['label']
        score = res['score']
        emotions.append((emotion, score))
        
    return emotions

# Prompt the user to enter the text for emotion detection
user_input = input("Enter a text or paragraph for emotion detection: ")

# Analyze the emotions in the user's text
emotions = analyze_emotions(user_input)

# Display the results
print("\nDetected emotions and their confidence scores:")
for emotion, score in emotions:
    print(f"{emotion}: {score:.4f}")
