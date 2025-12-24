from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime

# =================================================
# APP CONFIG — IMPORTANT
# =================================================
app = Flask(
    __name__,
    template_folder="bankbot/templates",
    static_folder="bankbot/static"
)
app.secret_key = "yesh_bank_secret_key"

# =================================================
# DUMMY USERS
# =================================================
users = {
    "yesh": "yesh123",
    "reddy": "bank123",
    "admin": "admin123"
}

# =================================================
# DATA
# =================================================
account_profile = {
    "name": "Yesh",
    "number": "96182240",
    "type": "Savings",
    "balance": 75000.00
}

transactions = [
    {"date": "2025-08-20", "desc": "Zomato Order", "amount": -450},
    {"date": "2025-08-18", "desc": "Amazon Purchase", "amount": -2999},
    {"date": "2025-08-15", "desc": "Flipkart Refund", "amount": 1500},
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
    {"city": "Hyderabad", "name": "Yesh Bank - HiTech City", "ifsc": "YESHB0000123"},
    {"city": "Bengaluru", "name": "Yesh Bank - Indiranagar", "ifsc": "YESHB0000456"},
]

# =================================================
# HELPERS
# =================================================
def logged_in():
    return "user" in session and session["user"] != "admin"

def is_admin():
    return session.get("user") == "admin"

# =================================================
# ROUTES — PUBLIC
# =================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username")
        p = request.form.get("password")
        if users.get(u) == p:
            session["user"] = u
            return redirect(url_for("dashboard" if u != "admin" else "admin_home"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =================================================
# ROUTES — USER PAGES
# =================================================
@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/balance")
def balance():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("balance.html", profile=account_profile)

@app.route("/transactions")
def transactions_page():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("transactions.html", txns=transactions)

@app.route("/cards")
def cards():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("cards.html", cards=cards_info)

@app.route("/loans")
def loans():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("loans.html", loans=loans_catalog)

@app.route("/branches")
def branches_page():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("branches.html", branches=branches)

@app.route("/chatbot")
def chatbot():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template(
        "chatbot.html",
        now=datetime.now().strftime("%d %b %Y, %I:%M %p")
    )

# =================================================
# ROUTES — ADMIN
# =================================================
@app.route("/admin")
def admin_home():
    if not is_admin():
        return redirect(url_for("login"))
    return render_template("admin_home.html")

@app.route("/admin/logs")
def admin_logs():
    if not is_admin():
        return redirect(url_for("login"))
    return render_template("admin_logs.html")

@app.route("/admin/training")
def admin_training():
    if not is_admin():
        return redirect(url_for("login"))
    return render_template("admin_training.html")

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    app.run()
