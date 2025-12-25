import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import spacy, csv, random, sqlite3, pandas as pd, subprocess

# ================= PATH FIX (THIS IS THE KEY) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

app.secret_key = "yesh_bank_secret_key"

# ================= LOAD AI MODEL =================
def load_model():
    global nlp_model
    try:
        nlp_model = spacy.load("bank_nlu_model")
        print("✅ NLU model loaded")
    except Exception as e:
        nlp_model = None
        print("❌ NLU model not found:", e)

load_model()

# ================= LOAD RESPONSES =================
responses_dict = {}

def load_responses():
    global responses_dict
    responses_dict = {}
    path = os.path.join(BASE_DIR, "training_and_responses.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, header=None,
                     names=["example", "intent", "response", "source"],
                     on_bad_lines="skip")
    for _, row in df.iterrows():
        responses_dict.setdefault(row["intent"], []).append(row["response"])

load_responses()

# ================= DATABASE LOGGING =================
def save_log(user_message, intent, entities, bot_response):
    conn = sqlite3.connect(os.path.join(BASE_DIR, "logs.db"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            intent TEXT,
            entities TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute(
        "INSERT INTO logs (user_message, intent, entities, bot_response) VALUES (?, ?, ?, ?)",
        (user_message, intent, str(entities), bot_response)
    )
    conn.commit()
    conn.close()

# ================= USERS =================
users = {
    "yesh": "yesh123",
    "reddy": "bank123",
    "admin": "admin123"
}

# ================= DUMMY DATA =================
# ================= USER-SPECIFIC DATA =================
user_accounts = {
    "yesh": {
        "profile": {
            "name": "Yesh",
            "number": "96182240",
            "type": "Savings",
            "balance": 75000.00
        },
        "transactions": [
            {"date": "2025-08-20", "desc": "Zomato Order", "amount": -450.00},
            {"date": "2025-08-18", "desc": "Amazon Purchase", "amount": -2999.00},
            {"date": "2025-08-15", "desc": "Flipkart Refund", "amount": 1500.00},
            {"date": "2025-08-10", "desc": "Rent Payment", "amount": -15000.00},
        ],
        "cards": {
            "debit": {
                "last4": "4321",
                "status": "Active",
                "limit": "₹50,000/day"
            },
            "credit": {
                "last4": "9988",
                "status": "Active",
                "limit": "₹1,50,000"
            }
        }
    },

    "reddy": {
        "profile": {
            "name": "Reddy",
            "number": "96182241",
            "type": "Savings",
            "balance": 50000.00
        },
        "transactions": [],
        "cards": {
            "debit": {
                "last4": "1122",
                "status": "Active",
                "limit": "₹30,000/day"
            },
            "credit": {
                "last4": "3344",
                "status": "Inactive",
                "limit": "₹75,000"
            }
        }
    }
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
    return "user" in session

def is_admin():
    return session.get("user") == "admin"
def current_user_data():
    username = session.get("user")
    if not username:
        return None
    return user_accounts.get(username)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()

        if u in users and users[u] == p:
            session["user"] = u
            if u == "admin":
                return redirect(url_for("admin_home"))
            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/balance")
def balance():
    if not logged_in():
        return redirect(url_for("login"))

    data = current_user_data()
    if not data:
        flash("User data not found", "danger")
        return redirect(url_for("dashboard"))

    return render_template("balance.html", profile=data["profile"])


@app.route("/transactions")
def transactions_page():
    if not logged_in():
        return redirect(url_for("login"))

    data = current_user_data()
    if not data:
        flash("User data not found", "danger")
        return redirect(url_for("dashboard"))

    txns_with_balance = []
    running_balance = data["profile"]["balance"]

    # reverse to calculate correctly
    for txn in reversed(data["transactions"]):
        running_balance -= txn["amount"]
        t = txn.copy()
        t["balance"] = running_balance
        txns_with_balance.append(t)

    # restore original order
    txns_with_balance.reverse()

    return render_template("transactions.html", txns=txns_with_balance)



@app.route("/loans")
def loans():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("loans.html", loans=loans_catalog)

@app.route("/cards")
def cards():
    if not logged_in():
        return redirect(url_for("login"))

    data = current_user_data()
    if not data or "cards" not in data:
        flash("Card data not available", "danger")
        return redirect(url_for("dashboard"))

    return render_template("cards.html", cards=data["cards"])


@app.route("/branches")
def branches_page():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("branches.html", branches=branches)

@app.route("/chatbot")
def chatbot():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("chatbot.html",
                           now=datetime.now().strftime("%d %b %Y, %I:%M %p"))
@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not logged_in():
        return jsonify({"reply": "Please login to continue."})

    data = request.get_json() or {}
    message = data.get("message", "").strip().lower()

    # --- Conversation state ---
    state = session.get("chat_state")
    transfer_data = session.get("transfer_data", {})
    user_data = current_user_data()
    if not user_data:
        return jsonify({"reply": "User data not found."})

    profile = user_data["profile"]
    transactions = user_data["transactions"]


    # ================= TRANSFER FLOW =================
    if state == "awaiting_account":
        transfer_data["account"] = message
        session["transfer_data"] = transfer_data
        session["chat_state"] = "awaiting_amount"
        return jsonify({"reply": "💰 Please enter the amount to transfer."})

    if state == "awaiting_amount":
        try:
            amount = float(message)
        except ValueError:
            return jsonify({"reply": "⚠️ Please enter a valid numeric amount."})

        if amount <= 0:
            return jsonify({"reply": "⚠️ Amount must be greater than zero."})

        if amount > account_profile["balance"]:
            session.pop("chat_state", None)
            session.pop("transfer_data", None)
            return jsonify({"reply": "❌ Insufficient balance for this transfer."})

        # Perform transfer
       if amount > profile["balance"]:
            session.pop("chat_state", None)
            session.pop("transfer_data", None)
            return jsonify({"reply": "❌ Insufficient balance for this transfer."})

       profile["balance"] -= amount
       transactions.insert(0, {
       "date": datetime.now().strftime("%Y-%m-%d"),
       "desc": f"Transfer to A/C {transfer_data['account']}",
       "amount": -amount
       })


        session.pop("chat_state", None)
        session.pop("transfer_data", None)

        reply = (
            f"✅ Successfully transferred ₹{amount:.2f} "
            f"to account {transfer_data['account']}.\n"
            f"💰 New balance: ₹{account_profile['balance']:.2f}"
        )

        save_log(message, "transfer_money", [], reply)
        return jsonify({"reply": reply})

    # ================= NORMAL CHAT =================
    if "transfer" in message or "send money" in message:
        session["chat_state"] = "awaiting_account"
        session["transfer_data"] = {}
        return jsonify({"reply": "💸 Please enter the recipient account number."})

    elif "balance" in message:
        reply = f"💰 Your current balance is ₹{account_profile['balance']:.2f}"

    elif "transaction" in message:
        if not transactions:
            reply = "📭 You have no recent transactions."
        else:
            reply = "🧾 Recent Transactions:\n"
            for t in transactions[:5]:
                sign = "+" if t["amount"] > 0 else "-"
                reply += f"{t['date']} | {t['desc']} | {sign}₹{abs(t['amount']):.2f}\n"

    elif "loan" in message:
        reply = "🏦 We offer Personal and Home loans. Visit the Loans page for details."

    elif "card" in message:
        reply = "💳 Your debit and credit cards are currently active."

    elif "branch" in message:
        reply = "📍 We have branches in Hyderabad, Bengaluru, and Mumbai."

    else:
        reply = (
            "🤖 I can help with:\n"
            "- Balance\n"
            "- Transactions\n"
            "- Transfer money\n"
            "- Loans\n"
            "- Cards\n"
            "- Branch details"
        )

    save_log(message, "chatbot", [], reply)
    return jsonify({"reply": reply})


@app.route("/admin")
def admin_home():
    if not is_admin():
        return redirect(url_for("login"))

    return render_template(
        "admin_home.html",
        user=session.get("user", "admin"),
        session=session
    )

# ================= ADMIN LOGS =================
@app.route("/admin/logs")
def view_logs():
    if not is_admin():
        return redirect(url_for("login"))

    conn = sqlite3.connect(os.path.join(BASE_DIR, "logs.db"))
    cur = conn.cursor()
    cur.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50")
    logs = cur.fetchall()
    conn.close()

    return render_template("admin_logs.html", logs=logs)


# ================= ADMIN TRAINING =================
@app.route("/admin/training", methods=["GET", "POST"])
def admin_training():
    if not is_admin():
        return redirect(url_for("login"))

    file_path = os.path.join(BASE_DIR, "training_and_responses.csv")

    if request.method == "POST":
        intent = request.form.get("intent")
        example = request.form.get("example")
        response = request.form.get("response")

        if intent and example and response:
            with open(file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([example, intent, response, "admin"])
            flash("Training data added", "success")

    rows = []
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)
        rows = df.values.tolist()

    return render_template("admin_training.html", rows=rows)


@app.route("/admin/training/delete/<int:row_index>", methods=["POST"])
def delete_training_row(row_index):
    if not is_admin():
        return redirect(url_for("login"))

    file_path = os.path.join(BASE_DIR, "training_and_responses.csv")
    df = pd.read_csv(file_path, header=None)
    df = df.drop(row_index).reset_index(drop=True)
    df.to_csv(file_path, index=False, header=False)

    flash("Row deleted", "success")
    return redirect(url_for("admin_training"))


@app.route("/admin/retrain", methods=["POST"])
def retrain_model():
    if not is_admin():
        return redirect(url_for("login"))

    try:
        subprocess.Popen(["python", "train.py"])
        flash("🔄 Model retraining started in background", "success")
    except Exception as e:
        flash(f"❌ Failed to start retraining: {e}", "danger")

    return redirect(url_for("admin_home"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
