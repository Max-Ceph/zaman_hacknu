"""
Тестовый скрипт для проверки подключения к API
Запустите: python test_api.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def test_zaman_api():
    """Тестирует подключение к Zaman Bank API"""
    print("\n" + "=" * 60)
    print("ТЕСТ ПОДКЛЮЧЕНИЯ К ZAMAN BANK API")
    print("=" * 60)

    # Загружаем ключи
    api_key = os.getenv("ZAMAN_BANK_API_KEY")
    base_url = os.getenv("ZAMAN_BANK_BASE_URL")

    # Убираем кавычки
    if api_key:
        api_key = api_key.strip('"').strip("'")
    if base_url:
        base_url = base_url.strip('"').strip("'")

    print(f"\n📋 Конфигурация:")
    print(f"  API Key: {api_key[:20]}... (первые 20 символов)")
    print(f"  Base URL: {base_url}")

    if not api_key or not base_url:
        print("\n❌ ОШИБКА: Не найдены ключи в .env файле!")
        return False

    # Пробуем подключиться
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Тест 1: Простой чат
        print("\n🔄 Тест 1: Отправка простого запроса в чат...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Привет! Ответь одним словом: работает?"}
            ],
            max_tokens=10
        )
        print(f"✅ Ответ получен: {response.choices[0].message.content}")

        # Тест 2: Эмбеддинги
        print("\n🔄 Тест 2: Получение эмбеддинга...")
        embedding_response = client.embeddings.create(
            input=["Тестовая фраза"],
            model="text-embedding-3-small"
        )
        vector_length = len(embedding_response.data[0].embedding)
        print(f"✅ Эмбеддинг получен (размерность: {vector_length})")

        print("\n" + "=" * 60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ ОШИБКА при подключении:")
        print(f"  {str(e)}")
        print("\n💡 Возможные решения:")
        print("  1. Проверьте правильность API ключа")
        print("  2. Убедитесь, что в .env нет кавычек вокруг ключей")
        print("  3. Проверьте доступность URL: " + base_url)
        return False


def test_openai_api():
    """Тестирует подключение к стандартному OpenAI API (fallback)"""
    print("\n" + "=" * 60)
    print("ТЕСТ ПОДКЛЮЧЕНИЯ К OPENAI API (FALLBACK)")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\n⚠ OpenAI API ключ не найден (это нормально, если работает Zaman API)")
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Привет!"}],
            max_tokens=10
        )
        print(f"✅ OpenAI API работает: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API не работает: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 Запуск тестов API...")

    # Тестируем Zaman Bank API
    zaman_works = test_zaman_api()

    # Если Zaman не работает, тестируем OpenAI
    if not zaman_works:
        test_openai_api()

    print("\n✅ Тестирование завершено\n")