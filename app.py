from flask import Flask, request, jsonify
from flask_cors import CORS
from model import train_model

app = Flask(__name__)
CORS(app)  # allow frontend requests

# Train model once when server starts
model, vectorizer = train_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Spam Detector Backend Running"})


@app.route("/predict", methods=["POST"])
def predict_spam():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"prediction": "not-spam"})

    message = data["message"].strip()

    if message == "":
        return jsonify({"prediction": "not-spam"})

    X_test = vectorizer.transform([message])
    prediction = model.predict(X_test)[0]

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
