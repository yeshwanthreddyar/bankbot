from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import os, csv, random, sqlite3, pandas as pd

app = Flask(__name__)
app.secret_key = "yesh_bank_secret_key"

# ================= SAFE SPACY LOADING =================
try:
    import spacy
    try:
        nlp_model = spacy.load("bank_nlu_model")
        print("✅ Custom spaCy model loaded")
    except:
        nlp_model = spacy.blank("en")
        print("⚠️ Using blank spaCy model")
except:
    nlp_model = None
    print("❌ spaCy not available")

# ================= LOAD RESPONSES =================
responses_dict = {}

def load_responses():
    if not os.path.exists("training_and_responses.csv"):
        return
    df = pd.read_csv(
        "training_and_responses.csv",
        header=None,
        names=["example", "intent", "response", "source"],
        on_bad_lines="skip"
    )
    for _, row in df.iterrows():
        responses_dict.setdefault(row["intent"], []).append(row["response"])

load_responses()

# ================= DATABASE LOGGING =================
def save_log(user_message, intent, bot_response):
    conn = sqlite3.connect("logs.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            intent TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute(
        "INSERT INTO logs (user_message, intent, bot_response) VALUES (?, ?, ?)",
        (user_message, intent, bot_response)
    )
    conn.commit()
    conn.close()

# ================= DUMMY DATA =================
users = {"yesh": "yesh123"}

account_profile = {
    "number": "96182240",
    "balance": 75000.00
}

def logged_in():
    return "user" in session

# ================= ROUTES =================
@app.route("/")
def home():
    return "✅ BankBot is running successfully"

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if users.get(u) == p:
            session["user"] = u
            return redirect(url_for("chatbot"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/chatbot")
def chatbot():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("chatbot.html")

# ================= CHAT API =================
@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not logged_in():
        return jsonify({"reply": "Authentication error"})

    message = request.get_json().get("message", "").strip()

    if "balance" in message.lower():
        session["state"] = "awaiting_account"
        reply = "💰 Please enter your account number"
        save_log(message, "check_balance", reply)
        return jsonify({"reply": reply})

    if session.get("state") == "awaiting_account":
        if message == account_profile["number"]:
            reply = f"💰 Your balance is ₹{account_profile['balance']:.2f}"
        else:
            reply = "⚠️ Invalid account number"
        session.pop("state", None)
        save_log(message, "check_balance", reply)
        return jsonify({"reply": reply})

    reply = "I can help with balance enquiries"
    save_log(message, "fallback", reply)
    return jsonify({"reply": reply})

# ================= RUN =================
if __name__ == "__main__":
    app.run()
