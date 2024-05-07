import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model  # type: ignore # Import load_model to load the Keras model
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the LSTM model and tokenizer
model = load_model("./BackEnd/Models/LSTM/lstm_model.keras")
with open("./BackEnd/Models/LSTM/lstm_tokenizer.pkl", "rb") as handle:
    tokenizer = joblib.load(handle)

# Define Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Handle POST API requests
@app.route("/api/predict_news", methods=["POST"])
def predict_news():
    # Receive data from client
    news_data = request.json
    news_text = news_data["news_text"]

    # Preprocess the text using the loaded tokenizer
    news_sequence = tokenizer.texts_to_sequences([news_text])
    news_padded = pad_sequences(news_sequence, maxlen=100)  # Assuming you used maxlen=100 during training

    # Make predictions
    probabilities = model.predict(news_padded)[0]
    predicted_class = int(probabilities > 0.5)  # Convert probabilities to binary class

    # Return the result as JSON
    if predicted_class == 1:
        return jsonify({"prediction": "fake", "probability": float(probabilities)})
    else:
        return jsonify({"prediction": "true", "probability": float(1 - probabilities)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
