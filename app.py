from flask import Flask, render_template, request
import joblib
import re
import time

app = Flask(__name__)

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    word_count = 0
    character_count = 0
    reading_time = 0
    processing_time = 0
    confidence_level = ""
    model_name = "Logistic Regression"
    # FIX: separate flag for template logic (avoids string matching issues)
    is_fake = None

    if request.method == "POST":

        start = time.time()

        news = request.form["news"]

        word_count = len(news.split())
        character_count = len(news)
        reading_time = max(1, round(word_count / 200))

        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = round(max(probability) * 100, 2)

        if confidence >= 90:
            confidence_level = "High"
        elif confidence >= 75:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        if result == 0:
            prediction = "❌ Fake News"
            is_fake = True        # FIX: clean boolean flag
        else:
            prediction = "✅ Real News"
            is_fake = False

        end = time.time()
        processing_time = round(end - start, 3)

    return render_template(
        "index.html",
        prediction=prediction,
        is_fake=is_fake,          # FIX: pass flag to template
        confidence=confidence,
        word_count=word_count,
        character_count=character_count,
        reading_time=reading_time,
        processing_time=processing_time,
        confidence_level=confidence_level,
        model_name=model_name
    )


if __name__ == "__main__":
    # This only runs when you execute 'python app.py' locally
    app.run(host="127.0.0.1", port=5000, debug=False)