from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import cv2
import re
import pytesseract
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load model and tokenizer
model = load_model("emotion_lstm_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
emotion_classes = ["anger", "fear", "joy", "love", "sadness", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(emotion_classes)

def perform_ocr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    extracted_text = pytesseract.image_to_string(gray)
    return re.sub(r'\s+', ' ', extracted_text).strip()

def predict_emotion(text, max_length=100):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_length)
    prediction = model.predict(padded_seq)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image_file.save(temp.name)
        image_path = temp.name

    try:
        text = perform_ocr(image_path)
        if not text:
            return jsonify({"error": "No text detected in the image"}), 200
        
        emotion = predict_emotion(text)
        return jsonify({"text": text, "emotion": emotion})
    finally:
        os.remove(image_path)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

