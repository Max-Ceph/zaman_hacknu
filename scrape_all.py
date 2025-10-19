import requests
from bs4 import BeautifulSoup
import json
import time

# Список ключевых страниц сайта для создания базы знаний
TARGET_URLS = [
    "https://www.zamanbank.kz/ru/islamic-finance/otvety-na-voprosy",  # Эту страницу будем обрабатывать особо
    "https://www.zamanbank.kz/ru/islamic-finance/glossarij",
    "https://www.zamanbank.kz/ru/islamic-finance/islamskie-finansy",
    "https://www.zamanbank.kz/ru/personal/agentskij-depozit-vakala",
    "https://www.zamanbank.kz/ru/personal/onlajn-finansirovanie-bez-zaloga",
    "https://www.zamanbank.kz/ru/business/business"
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


def parse_faq_page(soup):
    """Специализированный парсер для страницы FAQ, сохраняет структуру Вопрос-Ответ."""
    print("  - Обнаружена страница FAQ. Использую специализированный парсер...")

    # Третья, самая надежная попытка найти блоки. Ищем по ID.
    faq_items = soup.find_all('div', id=lambda x: x and x.startswith('question-faq-'))

    if not faq_items:
        print("  - Не удалось найти блоки вопросов-ответов на странице FAQ.")
        return []

    chunks = []
    for item in faq_items:
        question_tag = item.find('div', class_='text-2xl')
        answer_tag = item.find('div', class_='_cie-faq')

        if question_tag and answer_tag:
            question = question_tag.get_text(strip=True)
            answer = answer_tag.get_text(strip=True)
            # Сохраняем структуру в виде единой строки. Это идеально для поиска.
            chunks.append(f"Вопрос: {question} Ответ: {answer}")

    return chunks


def parse_generic_page(soup):
    """Универсальный парсер для всех остальных страниц."""
    print("  - Использую универсальный парсер для основного контента...")
    content_block = soup.find('main', class_='content')
    if content_block:
        return [content_block.get_text(separator=' ', strip=True)]
    return []


def main():
    knowledge_base = []

    for url in TARGET_URLS:
        print(f"\nОбработка страницы: {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Выбираем парсер в зависимости от URL
            if "/otvety-na-voprosy" in url:
                content_chunks = parse_faq_page(soup)
            else:
                content_chunks = parse_generic_page(soup)

            if not content_chunks:
                print("  - Не удалось извлечь контент.")
                continue

            for chunk in content_chunks:
                knowledge_base.append({
                    "source_url": url,
                    "content": chunk
                })

            print(f"  - Успешно! Добавлено {len(content_chunks)} фрагментов данных.")

        except requests.exceptions.RequestException as e:
            print(f"  - Ошибка при загрузке страницы: {e}")

        time.sleep(1)  # Пауза между запросами

    with open("knowledge_base.json", 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

    print(f"\nГотово! База знаний сохранена в 'knowledge_base.json'. Всего фрагментов: {len(knowledge_base)}")


if __name__ == "__main__":
    main()