from transformers import pipeline

# Initialize the emotion classification pipeline with the roberta-base-emotion model
emotion_pipeline = pipeline("text-classification", model="Dimi-G/roberta-base-emotion", tokenizer="Dimi-G/roberta-base-emotion")

def analyze_emotion(text):
    # Use the pipeline to classify the emotion in the text
    result = emotion_pipeline(text)
    
    # Extract and return the emotion with the highest confidence
    emotion = result[0]['label']
    confidence = result[0]['score']
    return emotion, confidence

# Input text for emotion analysis
input_text = input("Enter a text for emotion analysis: ")

# Perform emotion analysis
emotion, confidence = analyze_emotion(input_text)

# Display the result
print(f"The detected emotion is: {emotion} with a confidence of {confidence:.2f}")
