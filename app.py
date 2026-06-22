from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    emotion = label_encoder.inverse_transform(prediction)[0]

    return render_template(
        'index.html',
        prediction=emotion,
        user_text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
    