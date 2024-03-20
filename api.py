import pickle

import nltk
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from flask import jsonify
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()
ps = PorterStemmer()
app.secret_key = "060312345"


# Load the trained model
model = load_model('img_classifier_model.h5')

# Target size for image preprocessing
target_size = (64, 64)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model('img_classifier_model.h5')

# Target size for image preprocessing
target_size = (64, 64)


class LoginRequest(BaseModel):
    email: str
    password: str


class ModelPredictionRequest(BaseModel):
    txt: str


class ImagePredictionRequest(BaseModel):
    image: str


@app.post("/login_validation")
def login_validation(request: LoginRequest):
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')
    return jsonify({"message": "Login successful"})


@app.post("/model")
def model_prediction(request: ModelPredictionRequest):
    data = request.get_json()
    txt = data.get("txt")

    # Mock model and vectorizer loading
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

    transformed_sms = transform_text(txt)
    vector_input = tfidf.transform([transformed_sms])

    # Mock model prediction
    result = model.predict(vector_input)[0]

    if result == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return jsonify({"result": result})


@app.post("/predicted_img")
def image_prediction(request: ImagePredictionRequest):
    data = request.get_json()

    # Mock image handling logic
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data['image']
    img_array = preprocess_image(image_data)

    # Mock model prediction
    result = model.predict(img_array)
    predicted_class = int(round(result[0][0]))
    res = "Spam" if predicted_class == 1 else "Not Spam"

    return jsonify({"result": res})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
