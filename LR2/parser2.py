import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
from urllib.parse import urljoin
import logging
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiaParser:
    def __init__(self):
        self.base_url = "https://ria.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.collected_articles = []
        self.total_words = 0
        self.target_articles = 1000  # Увеличили цель до 1000
        self.processed_urls = set()  # Для отслеживания обработанных URL

    def get_categories_from_main_page(self):
        """Получаем категории с главной страницы РИА Новости"""
        try:
            logger.info("Загружаем главную страницу для получения категорий...")
            response = self.session.get(self.base_url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Главная страница недоступна: {response.status_code}")
                return self.get_fallback_categories()

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Расширенный поиск категорий
            categories = {}
            
            # Поиск в различных блоках
            category_selectors = [
                'div.cell-extension a.cell-extension__item-link',
                'nav a',
                '.menu__item a',
                '.b-menu a',
                'a[href*="/category/"]',
                'a[href*="/section/"]'
            ]
            
            for selector in category_selectors:
                category_links = soup.select(selector)
                for link in category_links:
                    href = link.get('href')
                    category_name = link.get_text(strip=True)
                    
                    if href and category_name and len(category_name) > 2:
                        full_url = urljoin(self.base_url, href)
                        
                        # Более гибкая фильтрация категорий
                        if (full_url.startswith('https://ria.ru/') and 
                            len(category_name) < 50 and
                            not any(x in full_url.lower() for x in [
                                '/longread/', '/spetsialnaya-voennaya-operatsiya-na-ukraine/', 
                                '/photolents/', '/video/', '/infografika/', '/photo/', 
                                '/lenta/', '/search/', '/subscribe/', '/auth/'
                            ])):
                            
                            # Нормализуем имя категории
                            category_name = re.sub(r'\s+', ' ', category_name).strip()
                            categories[category_name] = full_url

            # Удаляем дубликаты по URL
            unique_categories = {}
            for name, url in categories.items():
                if url not in unique_categories.values():
                    unique_categories[name] = url

            if not unique_categories:
                logger.warning("Не найдено категорий на главной странице")
                return self.get_fallback_categories()

            logger.info(f"Всего найдено категорий: {len(unique_categories)}")
            return dict(list(unique_categories.items())[:15])  # Берем первые 15 категорий

        except Exception as e:
            logger.error(f"Ошибка при получении категорий с главной страницы: {e}")
            return self.get_fallback_categories()

    def get_fallback_categories(self):
        """Расширенный список резервных категорий"""
        return {
            "Политика": "https://ria.ru/politics/",
            "Экономика": "https://ria.ru/economy/",
            "Общество": "https://ria.ru/society/",
            "Происшествия": "https://ria.ru/incidents/",
            "Армия": "https://ria.ru/defense_safety/",
            "Наука": "https://ria.ru/science/",
            "Культура": "https://ria.ru/culture/",
            "Спорт": "https://ria.ru/sport/",
            "Мир": "https://ria.ru/world/",
            "Технологии": "https://ria.ru/technology/",
            "Туризм": "https://ria.ru/tourism/",
            "Религия": "https://ria.ru/religion/",
            "Авто": "https://ria.ru/auto/",
            "Недвижимость": "https://ria.ru/realty/",
            "Здоровье": "https://ria.ru/health/",
        }

    def get_article_links_from_category(self, category_url, pages=8):  # Увеличили до 8 страниц
        """Получаем ссылки на статьи из категории"""
        all_links = []
        
        for page in range(1, pages + 1):
            try:
                if page == 1:
                    url = category_url
                else:
                    # Формируем URL для следующих страниц
                    if category_url.endswith('/'):
                        url = f"{category_url}page{page}/"
                    else:
                        url = f"{category_url}/page{page}/"
                
                logger.info(f"Загружаем страницу {page}: {url}")
                response = self.session.get(url, timeout=15)

                if response.status_code != 200:
                    logger.warning(f"Страница недоступна: {response.status_code}")
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Расширенные селекторы для статей
                article_selectors = [
                    'a[href*="/20"]',
                    'div.list-item__content a',
                    'div.media__title a',
                    'a.list-item__title',
                    'div.cell-news__title a',
                    'div.cell-main-photo a',
                    'div.cell-list__item a',
                    'div.cell-author__item a',
                    'article a',
                    '.news-feed__item a',
                    '.b-news__list a',
                    '.content__list a',
                ]
                
                found_links = set()
                
                for selector in article_selectors:
                    links = soup.select(selector)
                    for link in links:
                        href = link.get('href')
                        if not href:
                            continue
                            
                        full_url = urljoin(self.base_url, href)
                        
                        # Более гибкая фильтрация статей
                        if (full_url.startswith('https://ria.ru/') and
                            re.search(r'/\d{6,}', full_url) and  # Содержит хотя бы 6 цифр
                            not any(x in full_url.lower() for x in [
                                '/video/', '/photo/', '/infographics/', '/longread/',
                                '/photolents/', '/gallery/', '/audio/', '/live/'
                            ]) and
                            full_url not in found_links and
                            full_url not in all_links and
                            full_url not in self.processed_urls):
                            
                            found_links.add(full_url)
                            all_links.append(full_url)

                logger.info(f"На странице {page} найдено {len(found_links)} статей")
                
                # Если на странице нет статей, прекращаем
                if not found_links and page > 2:
                    break

                # Случайная задержка между 0.5 и 1.5 секунд
                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                logger.error(f"Ошибка при загрузке страницы {page}: {e}")
                continue

        logger.info(f"Всего найдено статей в категории: {len(all_links)}")
        return all_links

    def parse_article(self, url, category):
        """Парсим статью с улучшенной обработкой"""
        try:
            if url in self.processed_urls:
                return None
                
            logger.info(f"Парсим статью: {url}")
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Заголовок
            title = self.extract_title(soup)
            if not title or title == "Заголовок не найден":
                return None

            # Текст
            text = self.extract_text(soup)
            if not text or text == "Текст статьи не найден":
                return None

            # Дата
            date = self.extract_date(soup, url)

            # Считаем слова
            word_count = len(text.split())

            # Проверяем минимальное количество слов
            if word_count < 50:  # Уменьшили минимальный порог
                return None

            article_data = {
                "title": title,
                "text": text,
                "publication_date": date,
                "url": url,
                "category": category,
                "word_count": word_count,
                "source": "ria.ru"
            }

            self.processed_urls.add(url)
            logger.info(f"✓ {title[:60]}... ({word_count} слов)")
            return article_data

        except Exception as e:
            logger.error(f"Ошибка парсинга статьи: {e}")
            return None

    def extract_title(self, soup):
        """Извлекаем заголовок с улучшенными селекторами"""
        title_selectors = [
            'h1.article__title',
            'h1.m-article__title',
            'h1.news__header__title',
            'h1.article-header__title',
            'h1.b-article__title',
            'h1',
            '.article-header h1',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 10:
                    clean_title = re.sub(r'\s*[|-]\s*РИА Новости.*$', '', title)
                    clean_title = re.sub(r'\s*[|-]\s*Ria\.ru.*$', '', clean_title)
                    return clean_title.strip()

        return "Заголовок не найден"

    def extract_text(self, soup):
        """Извлекаем текст статьи с улучшенными методами"""
        content_selectors = [
            'div.article__body',
            'div.article__text',
            'div.m-article__body',
            'div.news__text',
            'article',
            'div.b-article__body',
            'div[class*="content"]',
            'div[class*="article"]',
            'div[class*="text"]',
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                # Удаляем ненужные элементы
                unwanted_elements = content.find_all([
                    'script', 'style', 'nav', 'aside', 'footer', 'header',
                    'div.article__info', 'div.article__tags', 'div.m-article__info',
                    'div.news__header', 'div.share', 'div.comments', 'div.social',
                    'div.article__meta', 'div.b-article__meta', 'div.advertisement'
                ])
                
                for elem in unwanted_elements:
                    elem.decompose()

                # Извлекаем параграфы
                paragraphs = content.find_all(['p', 'div'])
                text_parts = []
                
                for p in paragraphs:
                    p_text = p.get_text(strip=True)
                    if (len(p_text) > 20 and 
                        not p_text.startswith(('Фото:', 'Источник:', 'Читайте также:', 
                                             'РИА Новости', '©', 'Подпишитесь', 'Смотрите также',
                                             'Читать далее', 'В тему:', 'Ранее:')) and
                        not re.match(r'^\d{1,2}:\d{2}', p_text)):
                        text_parts.append(p_text)
                
                if text_parts:
                    text = '\n\n'.join(text_parts)
                    if len(text) > 150:  # Уменьшили минимальную длину
                        return text

        # Альтернативный метод - собираем все подходящие параграфы
        all_paragraphs = soup.find_all('p')
        text_parts = []
        
        for p in all_paragraphs:
            p_text = p.get_text(strip=True)
            if (len(p_text) > 30 and 
                not any(x in p_text for x in ['Фото:', 'Источник:', 'Читайте также:']) and
                not p_text.startswith('©') and
                'Подпишитесь' not in p_text and
                not re.match(r'^\d{1,2}:\d{2}', p_text)):
                text_parts.append(p_text)
        
        if text_parts:
            text = '\n\n'.join(text_parts)
            if len(text) > 150:
                return text

        return "Текст статьи не найден"

    def extract_date(self, soup, url):
        """Извлекаем дату"""
        # Из meta-тегов
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish_date"]',
            'meta[name="pubdate"]',
            'meta[itemprop="datePublished"]'
        ]
        
        for selector in meta_selectors:
            meta_date = soup.select_one(selector)
            if meta_date and meta_date.get('content'):
                date_str = meta_date['content']
                try:
                    if 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    return date_str
                except:
                    pass

        # Из времени в статье
        time_element = soup.find('time')
        if time_element and time_element.get('datetime'):
            date_str = time_element['datetime']
            try:
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                return date_str
            except:
                pass

        # Из URL
        match = re.search(r'/(\d{4})(\d{2})(\d{2})/', url)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"

        return datetime.now().strftime('%Y-%m-%d')

    def parse_category(self, category_name, category_url):
        """Парсим все статьи в категории"""
        logger.info(f"Парсим категорию: {category_name}")

        # Получаем ссылки на статьи (8 страниц)
        article_links = self.get_article_links_from_category(category_url, pages=8)

        if not article_links:
            logger.warning(f"Не найдено статей в категории: {category_name}")
            return []

        category_articles = []
        parsed_count = 0

        # Парсим каждую статью
        for i, link in enumerate(article_links):
            # Проверяем, не достигли ли целевого количества
            if len(self.collected_articles) >= self.target_articles:
                break

            # Проверяем дубликаты
            if any(article['url'] == link for article in self.collected_articles):
                continue

            article = self.parse_article(link, category_name)
            if article:
                category_articles.append(article)
                self.collected_articles.append(article)
                self.total_words += article['word_count']
                parsed_count += 1

                # Выводим прогресс каждые 10 статей
                if len(self.collected_articles) % 10 == 0:
                    logger.info(f"Прогресс: {len(self.collected_articles)}/{self.target_articles} статей")

            # Случайная задержка между 0.3 и 0.8 секунд
            time.sleep(random.uniform(0.3, 0.8))

        logger.info(f"В категории '{category_name}' успешно спарсено {parsed_count}/{len(article_links)} статей")
        return category_articles

    def save_to_jsonl(self, filename):
        """Сохраняем в JSONL"""
        with open(filename, 'w', encoding='utf-8') as f:
            for article in self.collected_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        logger.info(f"Сохранено в {filename}")

    def run(self):
        """Запуск парсера"""
        logger.info("=== ЗАПУСК ПАРСЕРА RIA.RU ===")
        logger.info(f"Цель: собрать {self.target_articles} статей")

        # Получаем категории с главной страницы
        categories = self.get_categories_from_main_page()

        print("\n" + "="*60)
        print("НАЙДЕННЫЕ КАТЕГОРИИ:")
        for i, (name, url) in enumerate(categories.items(), 1):
            print(f"{i:2d}. {name}")
        print("="*60)

        # Парсим все категории по порядку
        for category_name, category_url in categories.items():
            # Проверяем, не достигли ли целевого количества
            if len(self.collected_articles) >= self.target_articles:
                logger.info(f"Достигнута цель: {len(self.collected_articles)} статей")
                break

            logger.info(f"\n>>> Обрабатываем: {category_name}")
            logger.info(f"Осталось собрать: {self.target_articles - len(self.collected_articles)} статей")

            articles = self.parse_category(category_name, category_url)
            logger.info(f"В категории '{category_name}' собрано {len(articles)} статей")

            # Небольшая пауза между категориями
            time.sleep(2)

        # Сохраняем результаты
        if self.collected_articles:
            filename = f'ria_ru_{len(self.collected_articles)}_articles.jsonl'
            self.save_to_jsonl(filename)
        else:
            logger.error("Не собрано ни одной статьи!")

        return self.collected_articles

# Запуск
if __name__ == "__main__":
    parser = RiaParser()
    articles = parser.run()

    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    print(f"Всего статей: {len(articles)}")
    print(f"Всего слов: {parser.total_words}")
    print(f"Файл: ria_ru_{len(articles)}_articles.jsonl")

    if articles:
        # Статистика по категориям
        categories_stats = {}
        for article in articles:
            cat = article['category']
            categories_stats[cat] = categories_stats.get(cat, 0) + 1

        print("\nСтатистика по категориям:")
        for category, count in sorted(categories_stats.items(), key=lambda x: x[1], reverse=True):
            words = sum(a['word_count'] for a in articles if a['category'] == category)
            avg_words = words // count if count > 0 else 0
            print(f"  {category}: {count} статей, {words} слов (в среднем {avg_words} слов/статья)")
        
        # Общая статистика
        avg_words_total = parser.total_words // len(articles) if articles else 0
        print(f"\nОбщая статистика:")
        print(f"  Средняя длина статьи: {avg_words_total} слов")
        print(f"  Минимальная цель достигнута: {'Да' if len(articles) >= 500 else 'Нет'}")
        print(f"  Идеальная цель достигнута: {'Да' if len(articles) >= 800 else 'Нет'}")
        print(f"  Максимальная цель достигнута: {'Да' if len(articles) >= 1000 else 'Нет'}")
    else:
        print("\n❌ Не собрано ни одной статьи!")