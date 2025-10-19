import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

# --- 1. НАСТРОЙКА API-КЛИЕНТА ---
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY не найден в .env файле!")

openai_client = OpenAI(api_key=openai_api_key)
print("✓ OpenAI API инициализирован")

# --- 2. КОНФИГУРАЦИЯ ЯЗЫКОВ ---
# Здесь мы указываем, какие файлы обрабатывать.
# Вы можете добавить новые языки, просто добавив новую строку.
LANGUAGES = [
    {
        "name": "Русский",
        "input_file": "knowledge_base.json",
        "output_file": "vector_database.json"
    },
    {
        "name": "Казахский",
        "input_file": "knowledge_base_kk.json",
        "output_file": "vector_database_kk.json"
    }
]


# --- 3. ФУНКЦИИ ДЛЯ ОБРАБОТКИ ДАННЫХ ---

def load_knowledge_base(filename):
    """Загружает сырые данные из указанного файла."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {filename} не найден.")
        print(f"   Убедитесь, что файл существует в текущей директории.")
        return None


def get_embedding(text_chunk, model="text-embedding-3-small"):
    """Получает вектор для одного фрагмента текста с помощью API."""
    try:
        text_chunk = text_chunk.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text_chunk], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"  ❌ Ошибка при векторизации: {e}")
        time.sleep(5)
        return None


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Разбивает большой текст на пересекающиеся чанки."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


# --- 4. ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ ---

def process_language_files(lang_config):
    """
    Выполняет полный цикл обработки для одного языка:
    загрузка -> чанкинг -> векторизация -> сохранение.
    """
    input_file = lang_config["input_file"]
    output_file = lang_config["output_file"]
    lang_name = lang_config["name"]

    print("\n" + "=" * 60)
    print(f"НАЧАТА ОБРАБОТКА ДЛЯ ЯЗЫКА: {lang_name.upper()}")
    print(f"Входной файл: {input_file}")
    print("=" * 60 + "\n")

    knowledge_data = load_knowledge_base(input_file)
    if not knowledge_data:
        return  # Переходим к следующему языку, если файл не найден

    vector_database = []
    total_chunks = 0

    print(f"📚 Найдено источников: {len(knowledge_data)}")
    print("🔄 Начинаю обработку...\n")

    for idx, entry in enumerate(knowledge_data, 1):
        source = entry["source_url"]
        content = entry["content"]
        chunks = chunk_text(content)

        print(f"[{idx}/{len(knowledge_data)}] Обработка: {source}")
        print(f"           Чанков: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"           └─ Векторизация {i + 1}/{len(chunks)}...", end=" ")
            vector = get_embedding(chunk)

            if vector:
                vector_database.append({
                    "source": source,
                    "content": chunk,
                    "vector": vector
                })
                total_chunks += 1
                print("✓")
            else:
                print("❌")

            time.sleep(0.5)

        print()

    if not vector_database:
        print(f"\n❌ Не удалось создать векторы для {lang_name}. Файл не будет сохранен.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vector_database, f, ensure_ascii=False, indent=4)

    print(f"✅ ОБРАБОТКА ДЛЯ '{lang_name}' ЗАВЕРШЕНА!")
    print(f"📦 Файл сохранен: {output_file}")
    print(f"📊 Всего векторов: {total_chunks}")


# --- 5. ЗАПУСК СКРИПТА ---

def main():
    """Главная функция: запускает обработку для всех языков, указанных в LANGUAGES."""
    print("\n" + "#" * 60)
    print("ЗАПУСК СКРИПТА ПОДГОТОВКИ ВЕКТОРНЫХ БАЗ ДАННЫХ")
    print("#" * 60)

    for lang_config in LANGUAGES:
        process_language_files(lang_config)

    print("\n" + "=" * 60)
    print("🎉 ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()