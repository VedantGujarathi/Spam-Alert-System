from flask import Flask, render_template, request, redirect, session
import sqlite3
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import nltk.data
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import sklearn

ps = PorterStemmer()
con = sqlite3.connect("database.db")
con.execute("create table if not exists login(username text,email text, password text)")
con.close()

app = Flask(__name__)
app.secret_key = "060312345"
# Load the trained model



# Target size for image preprocessing
target_size = (64, 64)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

@app.route('/')
def login():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template("homepage.html")


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from login where email=? and password=? ",(email,password))
    data = cur.fetchone()

    if data:
        session["email"] = data["email"]
        session["password"] = data["password"]
        return redirect("home")
    else:
        ab = "Incorrect Username and Password"
        return render_template('index.html', ab=ab)


@app.route('/registration',methods=['POST'])
def registration():
    try:
        name = request.form.get('txt')
        email = request.form.get('email')
        password = request.form.get('password')

        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("insert into login(username,email,password)values(?,?,?)", (name, email, password))
        con.commit()
        con.close()
        return redirect("home")

    except:
        return render_template('index.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect("/")


@app.route('/model',  methods=["POST"])
def model():
    txt = request.form.get("txt")
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    transformed_sms = transform_text(txt)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        result = "Spam"
    else:
        result = "Not Spam"
    return render_template('homepage.html', result=result)


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


@app.route('/predicted_img', methods=['POST'])
def predicted_img():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        model = load_model('img_classifier_model.h5')
        # Save the uploaded image
        img_path = 'static/uploaded_image.jpg'  # Choose a suitable location
        file.save(img_path)

        # Preprocess the image
        preprocessed_img = preprocess_image(img_path)

        # Make prediction
        result = model.predict(preprocessed_img)

        # Get the class label (spam or non-spam)
        predicted_class = int(round(result[0][0]))
        res=""
        if result == 1:
            res = "Spam"
        else:
            res = "Not Spam"
        print(result)
        return render_template('homepage.html', result=res)


if __name__ == "__main__":
    app.run(debug=True)