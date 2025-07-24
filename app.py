from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

DATA_FILE = "chat_data.json"

# Load data asli (belum di-lowercase)
with open(DATA_FILE, 'r', encoding="utf-8") as f:
    raw_data = json.load(f)

# Buat versi lowercase untuk pelatihan
def preprocess_data(data):
    return [{"question": item["question"].lower(), "answer": item["answer"].lower()} for item in data]

lower_data = preprocess_data(raw_data)

def train_bot(data):
    X_texts = [item["question"].lower() for item in data]
    y = [item["answer"] for item in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_texts)
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_bot(lower_data)

def get_response(user_input):
    vec_input = vectorizer.transform([user_input.lower()])
    corpus_vecs = vectorizer.transform([item["question"].lower() for item in raw_data])
    similarities = cosine_similarity(vec_input, corpus_vecs)
    max_sim = similarities.max()
    best_idx = np.argmax(similarities)
    
    if max_sim < 0.5:
        return "Maaf, aku belum mengerti maksudmu 😥. Seharusnya Ai-chan jawab apa ya?"
    return raw_data[best_idx]["answer"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message","").strip().lower()
    response = get_response(user_msg)
    if response:
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Ai-chan belum tahu jawaban itu 😥. Seharusnya jawab apa ya?"})

@app.route("/teach", methods=["POST"])
def teach():
    global raw_data, lower_data, model, vectorizer

    user_original = request.json.get("question", "").strip()
    reply_original = request.json.get("answer", "").strip()

    # Tambahkan versi asli ke data yang disimpan
    raw_data.append({
        "question": user_original,
        "answer": reply_original
    })

    # Simpan ke file (versi natural)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    # Latih ulang model dengan versi lowercase
    lower_data = preprocess_data(raw_data)
    model, vectorizer = train_bot(lower_data)

    return jsonify({"status": "ok", "message": "Noted. Ai-chan catet jawabannya! 💕"})

if __name__ == "__main__":
    app.run(debug=True)