import requests
from bs4 import BeautifulSoup
import json

# URL страницы с вопросами и ответами
URL = "https://www.zamanbank.kz/ru/islamic-finance/otvety-na-voprosy"

# Заголовки, чтобы сайт думал, что мы - обычный браузер
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def parse_zaman_faq():
    """
    Загружает страницу FAQ Zaman Bank, извлекает все вопросы и ответы.
    Возвращает список словарей, где каждый словарь - это {'question': ..., 'answer': ...}.
    """
    print(f"Начинаю парсинг страницы: {URL}")

    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        print("Страница успешно загружена.")

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Ищем все родительские блоки по новому, более надежному признаку.
        # У каждого блока есть ID, начинающийся с 'question-faq-'.
        faq_items = soup.find_all('div', id=lambda x: x and x.startswith('question-faq-'))

        if not faq_items:
            # Если по ID не нашлось, пробуем по классам (запасной вариант)
            faq_items = soup.find_all('div', class_='py-6 border-b border-gray-300')

        if not faq_items:
            print("Не найдено ни одного элемента с вопросом-ответом. Возможно, структура сайта снова изменилась.")
            return []

        print(f"Найдено {len(faq_items)} вопросов и ответов. Начинаю извлечение текста...")

        parsed_data = []
        for item in faq_items:
            # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
            # Находим заголовок (вопрос) по его набору классов
            question_tag = item.find('div', class_='text-2xl font-semibold')
            # Находим контент (ответ) по его уникальному классу
            answer_tag = item.find('div', class_='_cie-faq')

            if question_tag and answer_tag:
                question_text = question_tag.get_text(strip=True)
                # Внутри ответа могут быть теги <p>, get_text() их хорошо обработает
                answer_text = answer_tag.get_text(strip=True)

                parsed_data.append({
                    "question": question_text,
                    "answer": answer_text
                })

        print("Извлечение текста завершено.")
        return parsed_data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке страницы: {e}")
        return None


def save_to_json(data, filename="zaman_faq.json"):
    """Сохраняет данные в JSON файл."""
    if not data:
        print("Нет данных для сохранения.")
        return

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Данные успешно сохранены в файл: {filename}")


if __name__ == "__main__":
    faq_data = parse_zaman_faq()
    save_to_json(faq_data)