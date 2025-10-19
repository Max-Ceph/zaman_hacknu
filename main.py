import os
import bcrypt
import certifi
import json
import numpy as np
import pytz
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from openai import OpenAI
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from bson.decimal128 import Decimal128
from bson.json_util import dumps
from datetime import datetime, timedelta
import pytz
from bson.decimal128 import Decimal128
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sessions'

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY не найден в .env файле!")

openai_client = OpenAI(api_key=openai_api_key)
print(f"✓ OpenAI API инициализирован успешно")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("❌ MONGO_URI не найден в .env файле!")

ca = certifi.where()
mongo_client = MongoClient(MONGO_URI, tlsCAFile=ca)
db = mongo_client["E-commerce"]
if "transactions" not in db.list_collection_names():
    db.create_collection("transactions")
    print("✓ Коллекция transactions создана")

print(f"✓ MongoDB подключена успешно")
print(f"✓ База данных: E-commerce")
print(f"✓ Все системы готовы к работе!\n")

vector_database_ru = []
vector_database_kk = []


def load_vector_databases():
    global vector_database_ru, vector_database_kk

    try:
        with open("vector_database.json", "r", encoding="utf-8") as f:
            vector_database_ru = json.load(f)
            for item in vector_database_ru:
                item['vector'] = np.array(item['vector'])
        print(f"✓ Русская база знаний успешно загружена. Записей: {len(vector_database_ru)}")
    except FileNotFoundError:
        print("⚠ ПРЕДУПРЕЖДЕНИЕ: Файл vector_database.json (RU) не найден.")
        print("  Запустите prepare_data.py для создания русской базы знаний.")

    try:
        with open("vector_database_kk.json", "r", encoding="utf-8") as f:
            vector_database_kk = json.load(f)
            for item in vector_database_kk:
                item['vector'] = np.array(item['vector'])
        print(f"✓ Казахская база знаний успешно загружена. Записей: {len(vector_database_kk)}")
    except FileNotFoundError:
        print("⚠ ПРЕДУПРЕЖДЕНИЕ: Файл vector_database_kk.json (KK) не найден.")
        print("  Запустите prepare_data.py для создания казахской базы знаний.")


def get_embedding(text, model="text-embedding-3-small"):
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_relevant_chunk(user_question_vector, top_k=2):
    if not vector_database:
        return []

    similarities = [
        (cosine_similarity(user_question_vector, item['vector']), item['content'], item['source'])
        for item in vector_database
    ]
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [(content, source) for sim, content, source in similarities[:top_k]]


def find_most_relevant_chunk(user_question_vector, vector_db, top_k=2):
    if not vector_db:
        return []

    similarities = [
        (cosine_similarity(user_question_vector, item['vector']), item['content'], item['source'])
        for item in vector_db
    ]
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [(content, source) for sim, content, source in similarities[:top_k]]


def detect_language(text):
    kk_chars = set('ӘәІіҢңҒғҮүҰұҚқӨөҺһ')

    if any(char in kk_chars for char in text):
        return 'kk'

    ru_markers = [
        'хочу', 'нужно', 'нужен', 'нужна', 'можно', 'скажи', 'расскажи',
        'открыть', 'закрыть', 'получить', 'взять', 'оформить',
        'какой', 'какая', 'какие', 'который', 'где', 'когда', 'почему',
        'это', 'что', 'как', 'мне', 'меня', 'тебя', 'вас',
        'банк', 'счет', 'карту', 'кредит', 'займ'
    ]

    kk_markers = [
        'қалай', 'неше', 'қандай', 'маған', 'саған', 'сізге',
        'керек', 'тиіс', 'үшін', 'туралы', 'арқылы', 'бойынша',
        'қайда', 'қашан', 'неге', 'себебі',
        'алайын', 'берейін', 'жасайын', 'ашайын',
        'банкте', 'шот', 'аламын', 'қаржы', 'несие'
    ]

    text_lower = text.lower()

    ru_score = sum(1 for marker in ru_markers if marker in text_lower)
    kk_score = sum(1 for marker in kk_markers if marker in text_lower)

    if ru_score >= 2 and kk_score == 0:
        return 'ru'

    if kk_score >= 1:
        return 'kk'

    if ru_score >= 1:
        return 'ru'

    try:
        detected = detect(text)
        return detected if detected in ['ru', 'kk'] else 'ru'
    except:
        return 'ru'


def detect_intent_to_open_product(message: str, lang: str) -> bool:
    message_lower = message.lower()

    kk_triggers = [
        'карта аш', 'карта алайын', 'карта керек', 'карта ашу', 'карта алғым',
        'депозит аш', 'депозит ашайын', 'депозит керек', 'депозит ашу', 'депозит алғым',
        'несие ал', 'несие алайын', 'несие керек', 'несие алу', 'несие алғым',
        'кредит ал', 'кредит алайын', 'кредит керек', 'кредит алу', 'кредит алғым',
        'шот аш', 'шот ашайын', 'шот керек', 'шот ашу', 'шот алғым',
        'өтінім беру', 'өтінім жасау', 'рәсімдеу',
        'тіркелу', 'тіркелгім келеді', 'тіркелгім'
    ]

    ru_triggers = [
        'открыть карт', 'открою карт', 'хочу карт', 'нужна карт', 'оформить карт', 'карту открыть',
        'открыть депозит', 'открою депозит', 'хочу депозит', 'нужен депозит', 'оформить депозит', 'депозит открыть',
        'взять кредит', 'возьму кредит', 'хочу кредит', 'нужен кредит', 'оформить кредит', 'кредит взять',
        'взять заем', 'взять займ', 'получить заем', 'оформить заем',
        'открыть счет', 'открою счет', 'хочу счет', 'нужен счет', 'счет открыть',
        'подать заявку', 'оформить заявку', 'оформить продукт',
        'зарегистрироваться', 'регистрация', 'стать клиентом',
        'давай откро', 'давай оформ', 'давай возьм', 'давай созда',
        'помоги открыть', 'помоги оформить', 'помоги получить'
    ]

    triggers = kk_triggers if lang == 'kk' else ru_triggers

    match_found = any(trigger in message_lower for trigger in triggers)

    if match_found:
        print(f"✓ Найден триггер для открытия продукта в сообщении: '{message_lower[:50]}...'")

    return match_found


def get_rag_response(user_id: str, message: str) -> dict:
    lang = 'ru'

    try:
        detected_lang = detect_language(message)
        print(f"🌍 Обнаружен язык: {detected_lang}")

        lang = 'kk' if detected_lang == 'kk' else 'ru'
        print(f"📝 Используется логика для: '{lang}'")
        print(f"📨 Сообщение: '{message}'")

        should_open_bank_site = detect_intent_to_open_product(message, lang)
        print(f"🔗 Намерение открыть продукт: {should_open_bank_site}")

        wants_analytics = any(keyword in message.lower() for keyword in [
            'анализ', 'расход', 'трат', 'статистик', 'аналитик', 'где я трачу',
            'на что уходит', 'сколько трачу', 'мои траты'
        ])

        vector_database = vector_database_kk if lang == 'kk' else vector_database_ru

        print(f"🤖 Обработка RAG для пользователя {user_id} на языке '{lang}'")

        user = db.users.find_one({"_id": ObjectId(user_id)})
        user_accounts = list(db.accounts.find({"userId": ObjectId(user_id)}))
        user_goals = list(db.goals.find({"userId": ObjectId(user_id)}))
        first_name = user.get('profile', {}).get('firstName', 'друг') if user else 'друг'
        print(f"👤 Имя пользователя: {first_name}")

        analytics_context = ""
        if wants_analytics:
            print("📊 Клиент запросил аналитику расходов")
            analysis = analyze_spending_habits(user_id)
            if analysis:
                recommendations = generate_personalized_recommendations(user_id, analysis, user_goals)
                if lang == 'kk':
                    analytics_context = f"\n\n### КЛИЕНТТІҢ ШЫҒЫСТАРЫ ТАЛДАУЫ:\n"
                    analytics_context += f"Жалпы шығыстар: {analysis['total_expenses']:,.0f}₸\n"
                    analytics_context += f"Ең көп шығын категориясы: {analysis['top_category']}\n"
                    if recommendations:
                        analytics_context += f"\nПерсоналды ұсыныстар:\n{recommendations}\n"
                else:
                    analytics_context = f"\n\n### АНАЛИЗ РАСХОДОВ КЛИЕНТА:\n"
                    analytics_context += f"Общие расходы за месяц: {analysis['total_expenses']:,.0f}₸\n"
                    analytics_context += f"Топ категория: {analysis['top_category']}\n"
                    if recommendations:
                        analytics_context += f"\nПерсональные рекомендации:\n{recommendations}\n"

        astana_tz = pytz.timezone('Asia/Almaty')
        current_time = datetime.now(astana_tz)
        hour = current_time.hour

        if lang == 'kk':
            if 5 <= hour < 12:
                time_greeting = "Қайырлы таң"
            elif 12 <= hour < 17:
                time_greeting = "Қайырлы күн"
            elif 17 <= hour < 22:
                time_greeting = "Қайырлы кеш"
            else:
                time_greeting = "Қайырлы түн"
            accounts_header = "\nКлиенттің шоттары:\n"
            goals_header = "\nКлиенттің қаржылық мақсаттары:\n"
            no_data_msg = "Клиентте әлі шоттар немесе мақсаттар жоқ.\n"
        else:
            if 5 <= hour < 12:
                time_greeting = "Доброе утро"
            elif 12 <= hour < 17:
                time_greeting = "Добрый день"
            elif 17 <= hour < 22:
                time_greeting = "Добрый вечер"
            else:
                time_greeting = "Доброй ночи"
            accounts_header = "\nСчета клиента:\n"
            goals_header = "\nФинансовые цели клиента:\n"
            no_data_msg = "У клиента пока нет счетов или целей.\n"

        personal_context = f"### Персональные данные клиента:\nИмя: {first_name}\n"
        if not user_accounts and not user_goals:
            personal_context += no_data_msg
        if user_accounts:
            personal_context += accounts_header
            for acc in user_accounts:
                balance_value = acc.get('balance')
                balance_str = str(balance_value) if isinstance(balance_value, Decimal128) else '0'
                personal_context += f"- '{acc.get('accountName', 'N/A')}' шоты, баланс: {balance_str} {acc.get('currency', 'KZT')}\n"
        if user_goals:
            personal_context += goals_header
            for goal in user_goals:
                current_str = str(goal.get('currentAmount')) if isinstance(goal.get('currentAmount'),
                                                                           Decimal128) else '0'
                target_str = str(goal.get('targetAmount')) if isinstance(goal.get('targetAmount'), Decimal128) else '0'
                personal_context += f"- '{goal.get('goalName', 'N/A')}' мақсаты. Жиналған: {current_str} / {target_str} KZT\n"
        personal_context += "###\n\n"

        personal_context += analytics_context

        chat_history = list(db.chat_history.find({"userId": ObjectId(user_id)}).sort("timestamp", -1).limit(3))
        chat_history.reverse()
        is_first_message = len(chat_history) == 0


        history_context = ""
        if chat_history:
            history_context = "### История предыдущих сообщений:\n"
            for msg in chat_history:
                role = "Клиент" if msg.get('role') == 'user' else "Амир"
                history_context += f"{role}: {msg.get('message', '')}\n"
            history_context += "###\n\n"

        print(f"🔍 Поиск в базе знаний ('{lang}')...")
        question_vector = get_embedding(message)
        chunks_with_sources = find_most_relevant_chunk(np.array(question_vector), vector_db=vector_database, top_k=2)

        context_str = "Білім базасында деректер жоқ." if lang == 'kk' else "Нет данных в базе знаний."
        if chunks_with_sources:
            context_chunks = [item[0] for item in chunks_with_sources]
            context_str = "\n\n".join(context_chunks)
            print(f"✓ Найдено релевантных фрагментов: {len(chunks_with_sources)}")

        if should_open_bank_site:
            if lang == 'kk':
                product_instruction = "\n\nМАҢЫЗДЫ: Клиент өнім ашқысы келеді. Оған өнім туралы қысқаша айтып, содан кейін: 'Мен сізді Zaman Bank сайтына бағыттап жатырмын, онда сіз өтінім жасай аласыз!' деп жаз."
            else:
                product_instruction = "\n\nВАЖНО: Клиент хочет открыть продукт. Расскажи ему кратко о продукте, а затем скажи: 'Сейчас я перенаправлю вас на сайт Zaman Bank, где вы сможете оформить заявку!'"
        else:
            product_instruction = ""

        if lang == 'kk':
            system_prompt = f"""Сенің рөлің: Сен — Әмір, Zaman Bank-тің жеке қаржы тілімгері және психологиялық қолдаушы.

МАҢЫЗДЫ: Жауапты ТӘУЕЛСІЗ қазақ тілінде беру керек. Ешбір орыс сөздерін қолданба!

Сенің мақсатың:
1. Клиенттің қаржылық мақсаттарына жетуге көмектесу
2. Стресспен күресу үшін альтернативалар ұсыну (тек сатып алу емес!)
3. Денсаулыққа шығындар туралы ескерту
4. Пайдалы әдеттерді дамытуға ынталандыру

Қарым-қатынас стилің:
- Жылы, адамгершілікпен сөйле, досыңдай.
- Клиентке "сіз" деп құрметпен жүгін.
- ТӘУЕЛСІЗ қарапайым, түсінікті қазақ тілін қолдан.

ПСИХОЛОГИЯЛЫҚ ҚОЛДАУ:
- Егер клиент түнде (23:00-06:00) көп шығындайтын болса, импульсивті сатып алулар туралы ескерт
- Стресспен күресудің тиімді жолдарын ұсын: серуендеу, медитация, досқа қоңырау шалу, спорт
- Клиенттің жетістіктерін атап өт және мадақта

Қалай жауап беру керек:
✓ Жеке деректерді пайдалан
✓ Білім базасына сүйен
✓ Күрделі терминдерді қарапайым сөзбен түсіндір
Әр хабарламада сәлемдеспе! Сәлемдесу тек алғашқы байланыста ғана.
✗ Білім базасында жоқ ақпаратты ойдан шығарма
✗ ЕШБІР ** немесе * белгілерін қолданба!
✗ Тек қарапайым мәтінмен жауап бер!{product_instruction}"""
            user_prompt = f"{personal_context}{history_context}БІЛІМ БАЗАСЫНАН АЛЫНҒАН КОНТЕКСТ:\n{context_str}\n\nКЛИЕНТТІҢ АҒЫМДАҒЫ СҰРАҒЫ:\n{message}\n\nНҰСҚАУ: Жауапты қазақ тілінде бер."
        else:
            system_prompt = f"""Твоя роль: Ты — Амир, персональный финансовый наставник и психолог поддержки от Zaman Bank.

ВАЖНО: Отвечай ТОЛЬКО на русском языке!

Твои цели:
1. Помогать клиентам достигать финансовых целей
2. Предлагать альтернативы стресс-шопингу и импульсивным тратам
3. Предупреждать о вредных финансовых привычках
4. Мотивировать к развитию полезных привычек

Твой стиль общения:
- Общайся тепло, по-человечески, как заботливый друг.
- Обращайся к клиенту на "вы".
- Используй простой, понятный язык.

ПСИХОЛОГИЧЕСКАЯ ПОДДЕРЖКА:
- Если клиент тратит ночью (23:00-06:00), мягко предупреди об импульсивных покупках
- Предложи здоровые способы борьбы со стрессом: прогулки, медитация, звонок другу, спорт, хобби
- Отмечай успехи клиента и хвали прогресс
- Если видишь вредную привычку (много трат на развлечения), деликатно предложи альтернативу

Как отвечать:
✓ Используй персональные данные клиента для индивидуальных советов.
✓ Ссылайся на базу знаний для точной информации.
✓ Объясняй сложные термины простыми словами.
✓ Будь эмпатичным и поддерживающим
- НЕ здоровайся в каждом сообщении! Приветствие только при первом контакте.
✗ Не придумывай информацию, которой нет в базе знаний.
✗ НЕ используй ** или * символы!
✗ Отвечай только обычным текстом!{product_instruction}"""
            user_prompt = f"{personal_context}{history_context}КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context_str}\n\nТЕКУЩИЙ ВОПРОС КЛИЕНТА:\n{message}\n\nИНСТРУКЦИЯ: Ответь на русском языке."

        print("🚀 Отправка запроса к AI...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        bot_reply = response.choices[0].message.content
        bot_reply = bot_reply.replace('**', '').replace('*', '')
        print(f"✓ Получен ответ от AI")

        db.chat_history.insert_one({
            "userId": ObjectId(user_id),
            "role": "user",
            "message": message,
            "timestamp": datetime.utcnow()
        })
        db.chat_history.insert_one({
            "userId": ObjectId(user_id),
            "role": "assistant",
            "message": bot_reply,
            "timestamp": datetime.utcnow()
        })
        print("✓ История сохранена")

        return {
            "reply": bot_reply,
            "open_bank_site": should_open_bank_site,
            "bank_url": "https://www.zamanbank.kz/" if should_open_bank_site else None
        }

    except Exception as e:
        print(f"✗ Ошибка в get_rag_response: {e}")
        import traceback
        traceback.print_exc()

        error_msg = "Кешіріңіз, қате пайда болды. Қайта көріңізші." if lang == 'kk' else "Извините, произошла ошибка. Пожалуйста, попробуйте еще раз."

        return {
            "reply": error_msg,
            "open_bank_site": False,
            "bank_url": None
        }


def categorize_transaction(description: str) -> str:
    categories = {
        'Продукты': ['магазин', 'супермаркет', 'grocery', 'мегамарт', 'small', 'продукты'],
        'Транспорт': ['бензин', 'заправка', 'такси', 'яндекс', 'uber', 'автобус', 'метро'],
        'Развлечения': ['кино', 'кафе', 'ресторан', 'бар', 'клуб', 'концерт', 'игры'],
        'Одежда': ['zara', 'h&m', 'одежда', 'обувь', 'магазин одежды'],
        'Здоровье': ['аптека', 'клиника', 'больница', 'врач', 'лекарства'],
        'Связь': ['beeline', 'kcell', 'altel', 'интернет', 'телефон'],
        'Образование': ['курс', 'обучение', 'книга', 'университет'],
        'Переводы': ['перевод', 'transfer', 'другу', 'родителям']
    }

    desc_lower = description.lower()
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return 'Прочее'


def analyze_spending_habits(user_id: str) -> dict:
    try:
        print(f"📊 Начало анализа для пользователя: {user_id}")

        one_month_ago = datetime.utcnow() - timedelta(days=30)

        transactions = list(db.transactions.find({
            "userId": ObjectId(user_id),
            "createdAt": {"$gte": one_month_ago},
            "type": "expense"
        }))

        print(f"🔍 Найдено транзакций: {len(transactions)}")

        if len(transactions) == 0:
            print("⚠️ Нет транзакций для анализа")
            return None

        category_totals = {}
        night_spending = 0
        total_expenses = 0

        for tx in transactions:
            amount_field = tx.get('amount')

            if amount_field is None:
                print(f"⚠️ Транзакция без amount: {tx.get('_id')}")
                continue

            if isinstance(amount_field, Decimal128):
                amount = float(amount_field.to_decimal())
            elif isinstance(amount_field, dict) and '$numberDecimal' in amount_field:
                amount = float(amount_field['$numberDecimal'])
            elif isinstance(amount_field, (int, float)):
                amount = float(amount_field)
            else:
                print(f"⚠️ Неизвестный формат amount: {type(amount_field)}, значение: {amount_field}")
                continue

            total_expenses += amount

            category = categorize_transaction(tx.get('description', ''))
            category_totals[category] = category_totals.get(category, 0) + amount

            tx_date = tx.get('createdAt')
            if tx_date and isinstance(tx_date, datetime):
                tx_hour = tx_date.hour
                if tx_hour >= 23 or tx_hour < 6:
                    night_spending += amount

        print(f"💰 Общая сумма расходов: {total_expenses:,.0f}₸")
        print(f"📊 Категории: {category_totals}")

        if category_totals:
            top_category = max(category_totals.items(), key=lambda x: x[1])
        else:
            top_category = ('Прочее', 0)

        result = {
            "total_expenses": total_expenses,
            "categories": category_totals,
            "top_category": top_category[0],
            "top_category_amount": top_category[1],
            "night_spending": night_spending,
            "night_spending_percentage": (night_spending / total_expenses * 100) if total_expenses > 0 else 0
        }

        print(f"✅ Анализ завершен успешно")
        print(f"📈 Результат: {result}")
        return result

    except Exception as e:
        print(f"❌ Ошибка в analyze_spending_habits: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_personalized_recommendations(user_id: str, analysis: dict, user_goals: list) -> str:
    if not analysis:
        return ""

    recommendations = []

    if analysis['top_category_amount'] > 50000:
        recommendations.append(
            f"💡 Вы тратите {analysis['top_category_amount']:,.0f}₸ в месяц на {analysis['top_category']}. "
            f"Если сократить расходы на 20%, можно сэкономить {analysis['top_category_amount'] * 0.2:,.0f}₸!"
        )

    if analysis['night_spending_percentage'] > 15:
        recommendations.append(
            f"⚠️ {analysis['night_spending_percentage']:.0f}% ваших трат происходит ночью. "
            f"Это может быть признаком импульсивных покупок. Попробуйте перед покупкой подождать 24 часа "
            f"или заменить шопинг на прогулку, медитацию или звонок другу."
        )

    if user_goals and analysis['total_expenses'] > 0:
        active_goal = next((g for g in user_goals if g.get('status') == 'active'), None)
        if active_goal:
            target_field = active_goal.get('targetAmount')
            current_field = active_goal.get('currentAmount')

            if isinstance(target_field, Decimal128):
                target = float(target_field.to_decimal())
            elif isinstance(target_field, dict) and '$numberDecimal' in target_field:
                target = float(target_field['$numberDecimal'])
            else:
                target = float(target_field) if target_field else 0

            if isinstance(current_field, Decimal128):
                current = float(current_field.to_decimal())
            elif isinstance(current_field, dict) and '$numberDecimal' in current_field:
                current = float(current_field['$numberDecimal'])
            else:
                current = float(current_field) if current_field else 0

            remaining = target - current

            if remaining > 0:
                monthly_potential_savings = analysis['total_expenses'] * 0.15
                months_to_goal = remaining / monthly_potential_savings if monthly_potential_savings > 0 else 999

                recommendations.append(
                    f"🎯 До вашей цели '{active_goal.get('goalName')}' осталось {remaining:,.0f}₸. "
                    f"Если откладывать {monthly_potential_savings:,.0f}₸ в месяц (15% от расходов), "
                    f"достигнете цели за {int(months_to_goal)} месяцев!"
                )

    return "\n\n".join(recommendations) if recommendations else ""


@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        period = request.args.get('period', 'month')

        now = datetime.utcnow()
        if period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'year':
            start_date = now - timedelta(days=365)
        else:
            start_date = now - timedelta(days=30)

        transactions = list(db.transactions.find({
            "userId": ObjectId(user_id),
            "createdAt": {"$gte": start_date}
        }).sort("createdAt", -1))

        return dumps(transactions), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Ошибка получения транзакций: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        print(f"🔍 Запрос аналитики для пользователя: {user_id}")

        analysis = analyze_spending_habits(user_id)

        if not analysis:
            print("⚠️ Недостаточно данных для анализа")
            return jsonify({"error": "Недостаточно данных для анализа"}), 400

        user_goals = list(db.goals.find({"userId": ObjectId(user_id)}))

        recommendations = generate_personalized_recommendations(user_id, analysis, user_goals)

        response_data = {
            "analysis": {
                "total_expenses": analysis['total_expenses'],
                "categories": analysis['categories'],
                "top_category": analysis['top_category'],
                "night_spending_percentage": analysis['night_spending_percentage']
            },
            "recommendations": recommendations
        }

        print(f"✅ Аналитика успешно сформирована. Расходы: {analysis['total_expenses']:,.0f}₸")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Критическая ошибка в /api/analytics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/transactions', methods=['POST'])
def add_transaction():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        data = request.get_json()

        new_transaction = {
            "userId": ObjectId(user_id),
            "type": data.get('type', 'expense'),
            "amount": Decimal128(str(data['amount'])),
            "description": data['description'],
            "category": categorize_transaction(data['description']),
            "createdAt": datetime.utcnow()
        }

        result = db.transactions.insert_one(new_transaction)

        return jsonify({
            "success": True,
            "transactionId": str(result.inserted_id)
        }), 200
    except Exception as e:
        print(f"Ошибка добавления транзакции: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-demo-data', methods=['POST'])
def generate_demo_data():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        print(f"🎲 Генерация демо-данных для пользователя: {user_id}")

        delete_result = db.transactions.delete_many({"userId": ObjectId(user_id)})
        print(f"🗑️ Удалено старых транзакций: {delete_result.deleted_count}")

        demo_transactions = [
            {"desc": "Магазин продуктов Small", "amount": 15000, "days_ago": 1},
            {"desc": "Яндекс.Такси", "amount": 2500, "days_ago": 2},
            {"desc": "Кофейня Starbucks", "amount": 3000, "days_ago": 2},
            {"desc": "Заправка KazMunayGas", "amount": 12000, "days_ago": 3},
            {"desc": "Магазин одежды Zara", "amount": 35000, "days_ago": 4, "hour": 23},
            {"desc": "Ресторан Burger King", "amount": 4500, "days_ago": 5},
            {"desc": "Аптека", "amount": 5000, "days_ago": 6},
            {"desc": "Beeline оплата", "amount": 3500, "days_ago": 7},
            {"desc": "Кино Chaplin", "amount": 4000, "days_ago": 8},
            {"desc": "Магазин продуктов", "amount": 18000, "days_ago": 9},
            {"desc": "Uber поездка", "amount": 1800, "days_ago": 10},
            {"desc": "Книжный магазин", "amount": 8000, "days_ago": 11},
            {"desc": "Кафе Coffee Room", "amount": 2500, "days_ago": 12, "hour": 1},
            {"desc": "Заправка", "amount": 11000, "days_ago": 13},
            {"desc": "Магазин H&M", "amount": 28000, "days_ago": 14},
            {"desc": "Доставка еды Chocofood", "amount": 5500, "days_ago": 15},
            {"desc": "Аптека лекарства", "amount": 7000, "days_ago": 16},
            {"desc": "Магазин продуктов", "amount": 16000, "days_ago": 17},
            {"desc": "Такси", "amount": 2000, "days_ago": 18},
            {"desc": "Кофейня", "amount": 2800, "days_ago": 19},
            {"desc": "Онлайн покупки Wildberries", "amount": 42000, "days_ago": 20, "hour": 0},
        ]

        transactions_to_insert = []
        for tx in demo_transactions:
            tx_date = datetime.utcnow() - timedelta(days=tx['days_ago'])
            if 'hour' in tx:
                tx_date = tx_date.replace(hour=tx['hour'])

            transactions_to_insert.append({
                "userId": ObjectId(user_id),
                "type": "expense",
                "amount": Decimal128(str(tx['amount'])),
                "description": tx['desc'],
                "category": categorize_transaction(tx['desc']),
                "createdAt": tx_date
            })

        result = db.transactions.insert_many(transactions_to_insert)
        print(f"✅ Вставлено транзакций: {len(result.inserted_ids)}")

        return jsonify({
            "success": True,
            "message": f"Создано {len(transactions_to_insert)} тестовых транзакций",
            "count": len(transactions_to_insert)
        }), 200

    except Exception as e:
        print(f"❌ Ошибка генерации данных: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        required_fields = ['firstName', 'lastName', 'email', 'password']
        if not all(field in data and data[field] for field in required_fields):
            return render_template('register.html', error="Пожалуйста, заполните все поля.")

        email = data['email'].lower()
        if db.users.find_one({"username": email}):
            return render_template('register.html', error="Пользователь с таким email уже существует")

        hashed_pw = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        new_user = {
            "username": email,
            "password": hashed_pw,
            "profile": {
                "firstName": data['firstName'],
                "lastName": data['lastName'],
                "currency": "KZT"
            },
            "preferences": {"notifications": {"email": True, "push": False}},
            "createdAt": datetime.utcnow()
        }
        db.users.insert_one(new_user)
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        if 'email' not in data:
            return "Bad Request: Email field is missing.", 400

        email = data['email'].lower()
        user = db.users.find_one({"username": email})

        if user and 'password' in user and bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
            if 'profile' in user:
                session['user_id'] = str(user['_id'])
                session['username'] = user['profile']['firstName']
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Аккаунт устарел. Пожалуйста, зарегистрируйтесь заново.")
        return render_template('login.html', error="Неверный email или пароль.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({
            "reply": "Пожалуйста, напишите что-нибудь.",
            "open_bank_site": False,
            "bank_url": None
        })

    print(f"💬 Получено сообщение от {user_id}: {user_message}")

    try:
        response_data = get_rag_response(user_id, user_message)

        if not isinstance(response_data, dict):
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: get_rag_response вернул не словарь: {type(response_data)}")
            response_data = {
                "reply": str(response_data),
                "open_bank_site": False,
                "bank_url": None
            }

        print(f"✓ Ответ отправлен: {response_data.get('reply', '')[:50]}...")

        if response_data.get('open_bank_site'):
            print(f"🔗 Перенаправление на: {response_data.get('bank_url')}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Критическая ошибка в эндпоинте /chat: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "reply": "Извините, произошла ошибка на сервере. Пожалуйста, попробуйте еще раз.",
            "open_bank_site": False,
            "bank_url": None
        }), 200


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files: return jsonify({"error": "Аудиофайл не найден"}), 400
    audio_file = request.files['audio']
    temp_audio_path = "temp_audio_file.webm"
    audio_file.save(temp_audio_path)
    try:
        with open(temp_audio_path, "rb") as f:
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return jsonify({'transcribed_text': transcript.text})
    except Exception as e:
        print(f"Ошибка при распознавании речи: {e}")
        return jsonify({"error": "Не удалось распознать речь"}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.route('/api/accounts')
def get_accounts():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        print(f"💰 Запрос счетов для пользователя: {user_id}")

        user_accounts = list(db.accounts.find({"userId": ObjectId(user_id)}))
        print(f"✓ Найдено счетов: {len(user_accounts)}")

        return dumps(user_accounts), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"❌ Ошибка при получении счетов: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/goals', methods=['GET'])
def get_goals():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session['user_id']
        print(f"📊 Запрос целей для пользователя: {user_id}")

        user_goals = list(db.goals.find({"userId": ObjectId(user_id)}))
        print(f"✓ Найдено целей: {len(user_goals)}")

        return dumps(user_goals), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"❌ Ошибка при получении целей: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/goals', methods=['POST'])
def add_goal():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_id']
    data = request.get_json()

    print(f"📝 Получены данные для новой цели: {data}")

    if not data or not data.get('goalName') or not data.get('targetAmount'):
        return jsonify({"error": "Отсутствуют обязательные поля"}), 400

    try:
        new_goal = {
            "userId": ObjectId(user_id),
            "goalType": data.get('goalType', 'other'),
            "goalName": data['goalName'],
            "targetAmount": Decimal128(str(data['targetAmount'])),
            "currentAmount": Decimal128("0"),
            "status": "active",
            "createdAt": datetime.utcnow()
        }

        result = db.goals.insert_one(new_goal)
        print(f"✓ Цель успешно добавлена: {result.inserted_id}")

        return jsonify({
            "success": True,
            "message": "Цель успешно добавлена!",
            "goalId": str(result.inserted_id)
        }), 200
    except Exception as e:
        print(f"✗ Ошибка при добавлении цели: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_vector_databases()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)