import os
import openai
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_jwt_extended import JWTManager
from pymongo import MongoClient
import bcrypt
from dotenv import load_dotenv

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['JWT_SECRET_KEY'] = 'supersecret'
app.config['SECRET_KEY'] = 'anothersecret'
jwt = JWTManager(app)

load_dotenv()  # Загружаем переменные из .env

openai.api_key = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["E-commerce"]
users = db["users"]

load_dotenv()  # Загружаем переменные из .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(message: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Ты — финансовый консультант. Отвечай понятно, кратко(максимум 100 токенов), если тебе отправляют запрос 'Калькуляторы', спрашивай что пользователь хочет рассчитать,если 'Финансовое планирование' ты спрашивай какого вида  ."},
                {"role": "user", "content": message}
            ],
            max_tokens=300,
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка API: {e}"

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        hashed_pw = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        users.insert_one({"username": data['username'], "password": hashed_pw})
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    user = users.find_one({"username": data['username']})
    if user and bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        session['username'] = data['username']
        return redirect(url_for('dashboard'))
    return render_template('login.html', error="Неверные данные")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_message = data.get("message", "")

    try:
        bot_reply = get_openai_response(user_message)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": f"Ошибка: {e}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Берем порт из переменной окружения, Railway сам его задаст
    app.run(host="0.0.0.0", port=port)



