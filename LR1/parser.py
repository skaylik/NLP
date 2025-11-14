import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
from urllib.parse import urljoin
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorRUParser:
    def __init__(self):
        self.base_url = "https://indicator.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.collected_articles = []
        self.total_words = 0

    def get_categories(self):
        """Категории для парсинга"""
        return {
            "Открытия российских ученых": "https://indicator.ru/label/otkrytiya-rossijskih-uchenyh",
            "Дискуссионный клуб": "https://indicator.ru/label/debate-club",
            "Медицина": "https://indicator.ru/medicine",
            "Астрономия": "https://indicator.ru/astronomy",
            "Биология": "https://indicator.ru/biology",
            "Гуманитарные науки": "https://indicator.ru/humanitarian-science",
            "Математика и Computer Science": "https://indicator.ru/mathematics",
            "Науки о Земле": "https://indicator.ru/earth-science",
            "Сельское хозяйство": "https://indicator.ru/agriculture",
            "Технические науки": "https://indicator.ru/engineering-science",
            "Физика": "https://indicator.ru/physics",
            "Химия и науки о материалах": "https://indicator.ru/chemistry-and-materials"
        }

    def get_article_links_from_category(self, category_url):
        """Получаем ссылки на статьи из категории"""
        try:
            logger.info(f"Загружаем категорию: {category_url}")
            response = self.session.get(category_url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Страница недоступна: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            article_links = []

            # Ищем ВСЕ ссылки на странице
            all_links = soup.find_all('a', href=True)

            for link in all_links:
                href = link.get('href')
                if not href:
                    continue

                # Преобразуем в полный URL
                full_url = urljoin(self.base_url, href)

                # Критерии для статьи:
                # 1. Должна содержать дату в формате dd-mm-yyyy
                # 2. Должна заканчиваться на .htm
                # 3. Должна быть с indicator.ru
                if (full_url.startswith('https://indicator.ru/') and
                    re.search(r'\d{2}-\d{2}-\d{4}\.htm$', full_url) and
                    full_url not in article_links):

                    article_links.append(full_url)
                    logger.info(f"Найдена статья: {full_url}")

            logger.info(f"Всего найдено статей: {len(article_links)}")
            return article_links

        except Exception as e:
            logger.error(f"Ошибка при загрузке категории: {e}")
            return []

    def parse_article(self, url, category):
        """Парсим статью"""
        try:
            logger.info(f"Парсим статью: {url}")
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Заголовок
            title = self.extract_title(soup)
            if not title or title == "Заголовок не найден":
                logger.warning(f"Не найден заголовок: {url}")
                return None

            # Текст
            text = self.extract_text(soup)
            if not text or text == "Текст статьи не найден":
                logger.warning(f"Не найден текст: {url}")
                return None

            # Дата из URL
            date = self.extract_date_from_url(url)

            # Считаем слова
            word_count = len(text.split())

            article_data = {
                "title": title,
                "text": text,
                "publication_date": date,
                "url": url,
                "category": category,
                "word_count": word_count,
                "source": "indicator.ru"
            }

            logger.info(f"✓ {title[:70]}... ({word_count} слов)")
            return article_data

        except Exception as e:
            logger.error(f"Ошибка парсинга статьи: {e}")
            return None

    def extract_title(self, soup):
        """Извлекаем заголовок"""
        # Пробуем h1
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
            if title and len(title) > 5:
                return title

        # Пробуем title страницы
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            clean_title = re.sub(r'\s*[|-]\s*Indicator.*$', '', title)
            if len(clean_title) > 5:
                return clean_title

        return "Заголовок не найден"

    def extract_text(self, soup):
        """Извлекаем текст статьи"""
        # Основной контент
        content = soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post'))

        if content:
            # Удаляем ненужные элементы
            for elem in content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                elem.decompose()

            text = content.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n\n', text)

            if len(text) > 200:
                return text

        # Альтернатива: собираем все параграфы
        paragraphs = soup.find_all('p')
        text_parts = []

        for p in paragraphs:
            p_text = p.get_text(strip=True)
            if len(p_text) > 30 and not p_text.startswith(('Фото:', 'Источник:', 'Читайте также:')):
                text_parts.append(p_text)

        if text_parts:
            text = '\n\n'.join(text_parts)
            if len(text) > 200:
                return text

        return "Текст статьи не найден"

    def extract_date_from_url(self, url):
        """Извлекаем дату из URL"""
        match = re.search(r'(\d{2})-(\d{2})-(\d{4})\.htm$', url)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month}-{day}"

        return datetime.now().strftime('%Y-%m-%d')

    def parse_category(self, category_name, category_url):
        """Парсим все статьи в категории"""
        logger.info(f"Парсим категорию: {category_name}")

        # Получаем ссылки на статьи
        article_links = self.get_article_links_from_category(category_url)

        if not article_links:
            logger.warning(f"Не найдено статей в категории: {category_name}")
            return []

        category_articles = []

        # Парсим каждую статью
        for i, link in enumerate(article_links):
            if self.total_words >= 50000:
                break

            # Проверяем дубликаты
            if any(article['url'] == link for article in self.collected_articles):
                continue

            article = self.parse_article(link, category_name)
            if article and article['word_count'] > 100:
                category_articles.append(article)
                self.total_words += article['word_count']

            # Задержка чтобы не блокировали
            time.sleep(0.5)

        return category_articles

    def save_to_jsonl(self, filename):
        """Сохраняем в JSONL"""
        with open(filename, 'w', encoding='utf-8') as f:
            for article in self.collected_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        logger.info(f"Сохранено в {filename}")

    def run(self):
        """Запуск парсера"""
        logger.info("=== ЗАПУСК ПАРСЕРА INDICATOR.RU ===")

        categories = self.get_categories()

        print("\n" + "="*60)
        print("КАТЕГОРИИ ДЛЯ ПАРСИНГА:")
        for i, (name, url) in enumerate(categories.items(), 1):
            print(f"{i:2d}. {name}")
        print("="*60)

        # Парсим все категории
        for category_name, category_url in categories.items():
            if self.total_words >= 50000:
                logger.info(f"Достигнута цель: {self.total_words} слов")
                break

            logger.info(f"\n>>> Обрабатываем: {category_name}")

            articles = self.parse_category(category_name, category_url)
            self.collected_articles.extend(articles)

            logger.info(f"Прогресс: {self.total_words} слов из 50000")

        # Сохраняем результаты
        if self.collected_articles:
            self.save_to_jsonl('indicator_ru_corpus.jsonl')
        else:
            logger.error("Не собрано ни одной статьи!")

        return self.collected_articles

# Запуск
if __name__ == "__main__":
    parser = IndicatorRUParser()
    articles = parser.run()

    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    print(f"Всего статей: {len(articles)}")
    print(f"Всего слов: {parser.total_words}")
    print(f"Файл: indicator_ru_corpus.jsonl")

    if articles:
        # Статистика по категориям
        categories_stats = {}
        for article in articles:
            cat = article['category']
            categories_stats[cat] = categories_stats.get(cat, 0) + 1

        print("\nСтатистика по категориям:")
        for category, count in sorted(categories_stats.items(), key=lambda x: x[1], reverse=True):
            words = sum(a['word_count'] for a in articles if a['category'] == category)
            print(f"  {category}: {count} статей, {words} слов")
    else:
        print("\n❌ Не собрано ни одной статьи!")