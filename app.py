from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import os, random, sqlite3, pandas as pd

app = Flask(__name__)
app.secret_key = "yesh_bank_secret_key"

# ================= SAFE SPACY LOADING =================
try:
    import spacy
    try:
        nlp_model = spacy.load("bank_nlu_model")
    except:
        nlp_model = spacy.blank("en")
except:
    nlp_model = None

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
users = {"yesh": "yesh123", "reddy": "bank123"}

account_profile = {
    "name": "Yesh",
    "number": "96182240",
    "balance": 75000.00
}

# ================= HELPERS =================
def logged_in():
    return "user" in session

# ================= ROUTES =================

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if users.get(u) == p:
            session["user"] = u
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ✅ FIXED: DASHBOARD ROUTE
@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template(
        "dashboard.html",
        user=session["user"],
        balance=account_profile["balance"]
    )

@app.route("/chatbot")
def chatbot():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template(
        "chatbot.html",
        now=datetime.now().strftime("%d %b %Y, %I:%M %p")
    )

# ================= CHAT API =================
@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not logged_in():
        return jsonify({"reply": "Authentication error", "intent": "error"})

    message = request.get_json().get("message", "").strip().lower()

    # ---- BALANCE FLOW ----
    if session.get("state") == "awaiting_account":
        if message == account_profile["number"]:
            reply = f"💰 Your account balance is ₹{account_profile['balance']:.2f}"
        else:
            reply = "⚠️ Invalid account number. Try again."
        session.pop("state", None)
        save_log(message, "check_balance", reply)
        return jsonify({"reply": reply, "intent": "check_balance"})

    if "balance" in message:
        session["state"] = "awaiting_account"
        reply = "💰 Please enter your account number"
        save_log(message, "check_balance", reply)
        return jsonify({"reply": reply, "intent": "check_balance"})

    reply = "I can help you check your balance."
    save_log(message, "fallback", reply)
    return jsonify({"reply": reply, "intent": "fallback"})

# ================= RUN =================
if __name__ == "__main__":
    app.run()
