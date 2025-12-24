from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import spacy, os, csv, random, sqlite3, pandas as pd, subprocess

app = Flask(__name__)
app.secret_key = "yesh_bank_secret_key"

# ---------------- LOAD AI MODEL ----------------
def load_model():
    global nlp_model
    try:
        nlp_model = spacy.load("bank_nlu_model")
        print("✅ NLU model loaded successfully.")
    except IOError:
        nlp_model = None
        print("❌ NLU model not found. Run train.py first.")

load_model()

# ---------------- LOAD RESPONSES ----------------
responses_dict = {}

def load_responses():
    global responses_dict
    responses_dict = {}
    file_path = "training_and_responses.csv"
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path, header=None,
                     names=["example", "intent", "response", "source"],
                     on_bad_lines="skip")
    for _, row in df.iterrows():
        responses_dict.setdefault(row["intent"], []).append(row["response"])

load_responses()

# ---------------- LOGGING ----------------
def save_log(user_message, intent, entities, bot_response):
    conn = sqlite3.connect("logs.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            intent TEXT,
            entities TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute(
        "INSERT INTO logs (user_message,intent,entities,bot_response) VALUES (?,?,?,?)",
        (user_message, intent, str(entities), bot_response)
    )
    conn.commit()
    conn.close()

# ---------------- DUMMY DATA ----------------
users = {"yesh": "yesh123", "reddy": "bank123", "admin": "admin123"}

account_profile = {
    "name": "Yesh",
    "number": "96182240",
    "type": "Savings",
    "balance": 75000.00
}

transactions = [
    {"date": "2025-08-20", "desc": "Zomato Order", "amount": -450.00},
    {"date": "2025-08-18", "desc": "Amazon Purchase", "amount": -2999.00},
    {"date": "2025-08-15", "desc": "Flipkart Refund", "amount": 1500.00},
    {"date": "2025-08-10", "desc": "Rent Payment", "amount": -15000.00},
]

cards_info = {
    "debit": {"status": "Active", "last4": "4321"},
    "credit": {"status": "Active", "last4": "9988"}
}

loans_catalog = [
    {"type": "Personal Loan", "rate": "11.25% p.a."},
    {"type": "Home Loan", "rate": "8.50% p.a."}
]

branches = [
    {"city": "Hyderabad", "name": "Yesh Bank - HiTech City", "address": "Plot 21, Cyber Towers", "ifsc": "YESHB0000123"},
    {"city": "Bengaluru", "name": "Yesh Bank - Indiranagar", "address": "100ft Rd, HAL 2nd Stage", "ifsc": "YESHB0000456"},
    {"city": "Mumbai", "name": "Yesh Bank - BKC", "address": "G Block, Bandra Kurla Complex", "ifsc": "YESHB0000789"},
]

def logged_in():
    return "user" in session and session["user"] != "admin"

def is_admin():
    return session.get("user") == "admin"

# ---------------- CHAT API ----------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not logged_in():
        return jsonify({"reply": "Authentication error.", "intent": "error"})

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()

    state = session.get("conversation_state")
    transfer = session.get("transfer_details", {})

    # ===== BALANCE ACCOUNT NUMBER FLOW =====
    if state == "awaiting_account_number":
        acc = message.replace(" ", "")
        if acc == account_profile["number"]:
            reply = f"💰 Your account balance is ₹{account_profile['balance']:.2f}."
        else:
            reply = "⚠️ Invalid account number. Please try again."

        session.pop("conversation_state", None)
        save_log(message, "check_balance", [("ACCOUNT_NUMBER", acc)], reply)
        return jsonify({"reply": reply, "intent": "check_balance"})

    # ===== TRANSFER FLOW =====
    if state == "awaiting_recipient":
        session["transfer_details"] = {"recipient": message}
        session["conversation_state"] = "awaiting_amount"
        reply = f"💸 How much would you like to send to {message}?"
        save_log(message, "transfer_money", [], reply)
        return jsonify({"reply": reply, "intent": "transfer_money"})

    if state == "awaiting_amount":
        try:
            amount = float(message.replace("₹", "").replace(",", ""))
            if amount <= 0:
                reply = "⚠️ Enter a valid amount."
            elif amount > account_profile["balance"]:
                reply = f"⚠️ Insufficient balance. Current balance ₹{account_profile['balance']:.2f}"
            else:
                transfer["amount"] = amount
                session["transfer_details"] = transfer
                session["conversation_state"] = "awaiting_confirmation"
                reply = f"💡 Confirm transfer of ₹{amount:.2f} to {transfer['recipient']}? (yes/no)"
                return jsonify({"reply": reply, "intent": "transfer_money"})
        except:
            reply = "⚠️ Enter numeric amount only."

        save_log(message, "transfer_money", [], reply)
        return jsonify({"reply": reply, "intent": "transfer_money"})

    if state == "awaiting_confirmation":
        recipient = transfer["recipient"]
        amount = transfer["amount"]

        if message.lower() in ["yes", "y"]:
            account_profile["balance"] -= amount
            transactions.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "desc": f"Transfer to {recipient}",
                "amount": -amount
            })
            reply = f"✅ ₹{amount:.2f} sent to {recipient}. New balance ₹{account_profile['balance']:.2f}"
        else:
            reply = "❌ Transfer cancelled."

        session.pop("conversation_state", None)
        session.pop("transfer_details", None)
        save_log(message, "transfer_money", [], reply)
        return jsonify({"reply": reply, "intent": "transfer_money"})

    # ===== NLU INTENT HANDLING (NO CONFIDENCE FOR BALANCE) =====
    doc = nlp_model(message)
    intent = max(doc.cats, key=doc.cats.get) if doc.cats else "n/a"

    if intent == "check_balance":
        session["conversation_state"] = "awaiting_account_number"
        reply = "💰 Please enter your account number."
    elif intent == "transfer_money":
        session["conversation_state"] = "awaiting_recipient"
        session["transfer_details"] = {}
        reply = "💸 Who should I send money to?"
    elif doc.cats.get(intent, 0) > 0.65:
        reply = random.choice(responses_dict.get(intent, ["I can help with banking queries."]))
    else:
        intent = "out_of_scope"
        reply = random.choice(responses_dict.get(intent, ["I can assist with banking services only."]))

    entities = [(e.text, e.label_) for e in doc.ents]
    save_log(message, intent, entities, reply)
    return jsonify({"reply": reply, "intent": intent})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
