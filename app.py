from flask import Flask, request, jsonify, render_template
import joblib
import re

# Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input comment from JSON
    data = request.get_json()
    comment = data.get('comment', '')

    # Preprocess the input
    cleaned_comment = clean_text(comment)

    # Transform the text using the vectorizer
    vectorized_comment = vectorizer.transform([cleaned_comment])

    # Predict the sentiment
    prediction = model.predict(vectorized_comment)[0]

    # Map prediction to sentiment labels
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map.get(prediction, "Unknown")

    return jsonify({"predicted_sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)