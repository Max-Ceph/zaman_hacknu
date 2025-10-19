"""
Скрипт для добавления тестовых данных в MongoDB
Запустите: python add_test_data.py
"""

import os
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from bson.decimal128 import Decimal128
from datetime import datetime

load_dotenv()

# Подключение к MongoDB
MONGO_URI = os.getenv("MONGO_URI")
ca = certifi.where()
mongo_client = MongoClient(MONGO_URI, tlsCAFile=ca)
db = mongo_client["E-commerce"]

print("\n" + "=" * 60)
print("ДОБАВЛЕНИЕ ТЕСТОВЫХ ДАННЫХ")
print("=" * 60)

# Найдем последнего зарегистрированного пользователя
user = db.users.find_one(sort=[("createdAt", -1)])

if not user:
    print("\n❌ Пользователи не найдены! Сначала зарегистрируйтесь через интерфейс.")
    exit()

user_id = user['_id']
user_name = user['profile']['firstName']

print(f"\n👤 Найден пользователь: {user_name}")
print(f"   ID: {user_id}")

# Проверяем существующие данные
existing_accounts = list(db.accounts.find({"userId": user_id}))
existing_goals = list(db.goals.find({"userId": user_id}))

print(f"\n📊 Текущие данные:")
print(f"   Счетов: {len(existing_accounts)}")
print(f"   Целей: {len(existing_goals)}")

# Добавляем тестовые счета, если их нет
if len(existing_accounts) == 0:
    print("\n💰 Добавляю тестовые счета...")

    test_accounts = [
        {
            "userId": user_id,
            "accountName": "Основной счет",
            "accountType": "checking",
            "balance": Decimal128("150000.50"),
            "currency": "KZT",
            "isActive": True,
            "createdAt": datetime.utcnow()
        },
        {
            "userId": user_id,
            "accountName": "Накопительный счет",
            "accountType": "savings",
            "balance": Decimal128("75000.00"),
            "currency": "KZT",
            "isActive": True,
            "createdAt": datetime.utcnow()
        }
    ]

    result = db.accounts.insert_many(test_accounts)
    print(f"   ✓ Добавлено счетов: {len(result.inserted_ids)}")
else:
    print("\n   ℹ Счета уже существуют, пропускаю...")

# Добавляем тестовые цели, если их нет
if len(existing_goals) < 2:
    print("\n🎯 Добавляю тестовые цели...")

    test_goals = [
        {
            "userId": user_id,
            "goalName": "Отпуск в Турции",
            "targetAmount": Decimal128("500000"),
            "currentAmount": Decimal128("125000"),
            "status": "active",
            "createdAt": datetime.utcnow()
        },
        {
            "userId": user_id,
            "goalName": "Новый ноутбук",
            "targetAmount": Decimal128("300000"),
            "currentAmount": Decimal128("50000"),
            "status": "active",
            "createdAt": datetime.utcnow()
        }
    ]

    result = db.goals.insert_many(test_goals)
    print(f"   ✓ Добавлено целей: {len(result.inserted_ids)}")
else:
    print("\n   ℹ Цели уже существуют, пропускаю...")

# Итоговая статистика
print("\n" + "=" * 60)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 60)

final_accounts = list(db.accounts.find({"userId": user_id}))
final_goals = list(db.goals.find({"userId": user_id}))

print(f"\n📊 Всего данных для {user_name}:")
print(f"   💰 Счетов: {len(final_accounts)}")
for acc in final_accounts:
    balance = str(acc['balance'])
    print(f"      - {acc['accountName']}: {balance} {acc['currency']}")

print(f"\n   🎯 Целей: {len(final_goals)}")
for goal in final_goals:
    current = str(goal['currentAmount'])
    target = str(goal['targetAmount'])
    progress = (float(current) / float(target)) * 100
    print(f"      - {goal['goalName']}: {progress:.1f}% ({current}/{target} KZT)")

print("\n✅ Готово! Обновите страницу в браузере.\n")