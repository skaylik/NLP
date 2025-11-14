import json
import time
import re
import random
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urlparse, parse_qs

class AdvancedSevenDaysParser:
    def __init__(self):
        # Настройки для Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
        
        # Расширенный список категорий с архивными страницами
        self.categories = {
            'Новости/Звезды': [
                'https://7days.ru/news/stars/',
                'https://7days.ru/news/stars/2.htm',
                'https://7days.ru/news/stars/3.htm',
                'https://7days.ru/news/stars/4.htm',
                'https://7days.ru/news/stars/5.htm'
            ],
            'Новости/Монархи': [
                'https://7days.ru/news/royals/',
                'https://7days.ru/news/royals/2.htm',
                'https://7days.ru/news/royals/3.htm'
            ],
            'Новости/Спорт': [
                'https://7days.ru/news/sports/',
                'https://7days.ru/news/sports/2.htm',
                'https://7days.ru/news/sports/3.htm'
            ],
            'Новости/В мире': [
                'https://7days.ru/news/world/',
                'https://7days.ru/news/world/2.htm',
                'https://7days.ru/news/world/3.htm'
            ],
            'Звезды/Светская хроника': [
                'https://7days.ru/stars/chronic/',
                'https://7days.ru/stars/chronic/2.htm',
                'https://7days.ru/stars/chronic/3.htm'
            ],
            'Звезды/Частная жизнь': [
                'https://7days.ru/stars/privatelife/',
                'https://7days.ru/stars/privatelife/2.htm',
                'https://7days.ru/stars/privatelife/3.htm'
            ],
            'Звезды/Рейтинги': [
                'https://7days.ru/stars/ratings/',
                'https://7days.ru/stars/ratings/2.htm'
            ],
            'Звезды/Энциклопедия звёзд': [
                'https://7days.ru/stars/bio/',
                'https://7days.ru/stars/bio/2.htm',
                'https://7days.ru/stars/bio/3.htm'
            ],
            'Стиль/Мода': [
                'https://7days.ru/style/fashion/',
                'https://7days.ru/style/fashion/2.htm',
                'https://7days.ru/style/fashion/3.htm'
            ],
            'Стиль/Красота': [
                'https://7days.ru/style/beauty/',
                'https://7days.ru/style/beauty/2.htm',
                'https://7days.ru/style/beauty/3.htm'
            ],
            'Стиль/Секреты красоты звезд': [
                'https://7days.ru/style/star-beauty-secrets/',
                'https://7days.ru/style/star-beauty-secrets/2.htm'
            ],
            'Стиль/Звездный стиль': [
                'https://7days.ru/style/star-style/',
                'https://7days.ru/style/star-style/2.htm'
            ],
            'Мой дом/Готовим вместе': [
                'https://7days.ru/lifestyle/food/',
                'https://7days.ru/lifestyle/food/2.htm',
                'https://7days.ru/lifestyle/food/3.htm'
            ],
            'Мой дом/Семья': [
                'https://7days.ru/lifestyle/family/',
                'https://7days.ru/lifestyle/family/2.htm',
                'https://7days.ru/lifestyle/family/3.htm'
            ],
            'Мой дом/Создаём уют': [
                'https://7days.ru/lifestyle/home/',
                'https://7days.ru/lifestyle/home/2.htm',
                'https://7days.ru/lifestyle/home/3.htm'
            ],
            'Мой дом/На отдых': [
                'https://7days.ru/lifestyle/travel/',
                'https://7days.ru/lifestyle/travel/2.htm',
                'https://7days.ru/lifestyle/travel/3.htm'
            ],
            'Кино/Новости': [
                'https://7days.ru/kino/movie-news/',
                'https://7days.ru/kino/movie-news/2.htm',
                'https://7days.ru/kino/movie-news/3.htm'
            ],
            'Кино/Рецензии': [
                'https://7days.ru/kino/review/',
                'https://7days.ru/kino/review/2.htm'
            ],
            'Кино/Подборки': [
                'https://7days.ru/kino/top/',
                'https://7days.ru/kino/top/2.htm'
            ],
            'Кино/Обзоры': [
                'https://7days.ru/kino/articles/',
                'https://7days.ru/kino/articles/2.htm'
            ],
            'Гороскопы/Гороскоп': [
                'https://7days.ru/astro/horoscope/',
                'https://7days.ru/astro/horoscope/2.htm'
            ],
            'Гороскопы/Сонник': [
                'https://7days.ru/astro/sonnik/',
                'https://7days.ru/astro/sonnik/2.htm'
            ],
            'Гороскопы/До звезды': [
                'https://7days.ru/astro/do-zvezdi/',
                'https://7days.ru/astro/do-zvezdi/2.htm'
            ],
            'Досуг/Афиша': [
                'https://7days.ru/dosug/afisha/',
                'https://7days.ru/dosug/afisha/2.htm'
            ],
            'Досуг/Книги': [
                'https://7days.ru/dosug/books/',
                'https://7days.ru/dosug/books/2.htm'
            ],
            'Досуг/Подкасты': [
                'https://7days.ru/dosug/podcasts/',
                'https://7days.ru/dosug/podcasts/2.htm'
            ],
            'Досуг/Тесты': [
                'https://7days.ru/dosug/tests/',
                'https://7days.ru/dosug/tests/2.htm'
            ],
            'Здоровье/Здоровье человека': [
                'https://health.7days.ru/vashe-zdorovie/',
                'https://health.7days.ru/vashe-zdorovie/2.htm',
                'https://health.7days.ru/vashe-zdorovie/3.htm'
            ],
            'Здоровье/Новости медицины': [
                'https://health.7days.ru/med-news/',
                'https://health.7days.ru/med-news/2.htm',
                'https://health.7days.ru/med-news/3.htm'
            ],
            'Здоровье/Похудеть': [
                'https://health.7days.ru/lose-weight/',
                'https://health.7days.ru/lose-weight/2.htm'
            ],
            'Здоровье/Питание': [
                'https://health.7days.ru/diets/',
                'https://health.7days.ru/diets/2.htm'
            ],
        }

        # Для отслеживания дубликатов
        self.processed_urls = set()
        
        # Загружаем уже обработанные URL если файл существует
        try:
            with open("processed_urls.txt", "r", encoding="utf-8") as f:
                self.processed_urls = set(line.strip() for line in f)
            print(f"Загружено {len(self.processed_urls)} обработанных URL")
        except FileNotFoundError:
            pass

    def save_processed_urls(self):
        """Сохраняет обработанные URL в файл"""
        with open("processed_urls.txt", "w", encoding="utf-8") as f:
            for url in self.processed_urls:
                f.write(url + "\n")

    def is_article_url(self, url):
        """Проверяет, является ли URL статьей"""
        if url in self.processed_urls:
            return False
            
        exclude_patterns = [
            r'/stars/\d+\.htm$', r'/news/\d+\.htm$', r'/page\d+',
            r'/search/', r'/tags/', r'/authors/', r'/copyright',
            r'/about', r'/contacts', r'/advertising', r'/gde-kupit-jurnal',
            r'/promokodi', r'/caravan', r'/rss'
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, url):
                return False
        
        include_patterns = [
            r'/.+/.+\.htm$', r'/.+-\d+\.htm$', r'/news/.+\.htm$',
            r'/stars/.+\.htm$', r'/style/.+\.htm$', r'/lifestyle/.+\.htm$',
            r'/kino/.+\.htm$', r'/astro/.+\.htm$', r'/dosug/.+\.htm$',
            r'health\.7days\.ru/.+\.htm$'
        ]
        
        for pattern in include_patterns:
            if re.search(pattern, url):
                return True
        
        return False

    def parse_article(self, url, category):
        """Парсит отдельную статью"""
        if url in self.processed_urls:
            return None
            
        try:
            print(f"Парсим статью: {url}")
            self.driver.get(url)
            time.sleep(2)
            
            if not self.is_article_page():
                return None
            
            # Заголовок
            try:
                title_element = self.driver.find_element(By.CSS_SELECTOR, "h1")
                title = title_element.text.strip()
                if not title or title in ['НОВОСТИ', 'ЗВЕЗДЫ', 'СТИЛЬ', 'КИНО', 'МОЙ ДОМ', 'ГОРОСКОПЫ', 'ДОСУГ', 'ЗДОРОВЬЕ']:
                    return None
            except:
                return None
            
            # Дата
            date = self.extract_date()
            
            # Текст
            text = self.extract_article_text()
            
            if not text or len(text) < 100:
                return None
            
            main_category = category.split('/')[0].strip()
            
            article_data = {
                "title": title,
                "text": text[:15000],
                "date": date,
                "url": url,
                "category": main_category
            }
            
            self.processed_urls.add(url)
            print(f"Успешно: {title[:50]}...")
            return article_data
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return None

    def is_article_page(self):
        """Проверяет, является ли страница статьей"""
        try:
            indicators = ["h1", "div.material-7days__content", "div.material-7days__autor-material"]
            for indicator in indicators:
                if self.driver.find_elements(By.CSS_SELECTOR, indicator):
                    return True
            return False
        except:
            return False

    def extract_date(self):
        """Извлекает дату"""
        try:
            selectors = [
                "div.material-7days__autor-material div",
                "div.material-7days__autor-material",
                "time",
                ".article__date"
            ]
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        text = element.text.strip()
                        if any(month in text for month in ['Января', 'Февраля', 'Марта', 'Апреля', 'Мая', 'Июня', 
                                                         'Июля', 'Августа', 'Сентября', 'Октября', 'Ноября', 'Декабря']):
                            date_match = re.search(r'(\d{1,2}\s+(?:Января|Февраля|Марта|Апреля|Мая|Июня|Июля|Августа|Сентября|Октября|Ноября|Декабря)\s+\d{4})', text)
                            if date_match:
                                return date_match.group(1)
                            return text.split('|')[-1].strip() if '|' in text else text
                except:
                    continue
        except:
            pass
        return "Не указана"

    def extract_article_text(self):
        """Извлекает текст статьи"""
        try:
            text_selectors = [
                "div.non-paywall div.material-7days__content p",
                "div.material-7days__content p",
                "div.material-7days__paragraf-content p",
                "div.material-7days__box p"
            ]
            
            all_text = []
            
            for selector in text_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        for element in elements:
                            text = element.text.strip()
                            if text and len(text) > 10:
                                all_text.append(text)
                        break
                except:
                    continue
            
            if not all_text:
                try:
                    content_div = self.driver.find_element(By.CSS_SELECTOR, "div.material-7days__box")
                    paragraphs = content_div.find_elements(By.TAG_NAME, "p")
                    all_text = [p.text.strip() for p in paragraphs if p.text.strip() and len(p.text.strip()) > 10]
                except:
                    pass
            
            return " ".join(all_text) if all_text else ""
            
        except:
            return ""

    def get_article_links_from_page(self, page_url):
        """Получает ссылки на статьи со страницы"""
        links = []
        try:
            self.driver.get(page_url)
            time.sleep(3)
            
            # Прокручиваем страницу для загрузки всего контента
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Ищем ссылки в различных блоках
            link_selectors = [
                "a[href*='/news/'][href$='.htm']",
                "a[href*='/stars/'][href$='.htm']",
                "a[href*='/style/'][href$='.htm']", 
                "a[href*='/lifestyle/'][href$='.htm']",
                "a[href*='/kino/'][href$='.htm']",
                "a[href*='/astro/'][href$='.htm']",
                "a[href*='/dosug/'][href$='.htm']",
                "a[href*='health.7days.ru'][href$='.htm']",
                ".base-material-7days__newsbox_item a",
                ".material-7days__slider_item a",
                "article a"
            ]
            
            found_links = set()
            
            for selector in link_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        try:
                            href = element.get_attribute("href")
                            if href and self.is_article_url(href):
                                found_links.add(href)
                        except:
                            continue
                except:
                    continue
            
            links = list(found_links)
            print(f"Найдено ссылок на {page_url}: {len(links)}")
            
        except Exception as e:
            print(f"Ошибка при загрузке {page_url}: {str(e)}")
        
        return links

    def parse_category_pages(self, category_name, category_urls):
        """Парсит все страницы категории"""
        all_articles = []
        
        for page_url in category_urls:
            print(f"Обрабатываем страницу: {page_url}")
            
            try:
                article_links = self.get_article_links_from_page(page_url)
                
                for link in article_links:
                    if link not in self.processed_urls:
                        article_data = self.parse_article(link, category_name)
                        if article_data:
                            all_articles.append(article_data)
                        
                        time.sleep(random.uniform(1, 2))
                        
                        # Сохраняем прогресс каждые 10 статей
                        if len(all_articles) % 10 == 0:
                            self.save_to_jsonl(all_articles, "progress_7days_articles.jsonl")
                            all_articles = []  # Очищаем после сохранения
                
            except Exception as e:
                print(f"Ошибка при обработке {page_url}: {str(e)}")
                continue
        
        return all_articles

    def save_to_jsonl(self, data, filename):
        """Сохраняет данные в JSONL"""
        with open(filename, 'a', encoding='utf-8') as f:
            for article in data:
                json_line = json.dumps(article, ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"Сохранено {len(data)} статей в {filename}")

    def run(self, target_articles=1000):
        """Запускает расширенный парсинг"""
        output_file = f"7days_articles_{target_articles}.jsonl"
        
        # Очищаем файл
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')
        
        total_articles = 0
        category_count = 0
        
        for category_name, category_urls in self.categories.items():
            if total_articles >= target_articles:
                break
                
            category_count += 1
            print(f"\n{'='*60}")
            print(f"Категория {category_count}/{len(self.categories)}: {category_name}")
            print(f"URLs для обработки: {len(category_urls)}")
            print(f"Уже собрано: {total_articles}/{target_articles}")
            print(f"{'='*60}")
            
            try:
                articles = self.parse_category_pages(category_name, category_urls)
                
                if articles:
                    self.save_to_jsonl(articles, output_file)
                    total_articles += len(articles)
                    print(f"Добавлено из {category_name}: {len(articles)} статей")
                
                # Сохраняем обработанные URL
                self.save_processed_urls()
                
            except Exception as e:
                print(f"Ошибка в категории {category_name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
        print(f"Собрано статей: {total_articles}")
        print(f"Уникальных URL: {len(self.processed_urls)}")
        print(f"Файл: {output_file}")
        print(f"{'='*60}")

    def close(self):
        """Закрывает браузер"""
        self.save_processed_urls()
        self.driver.quit()

# Дополнительный парсер для поиска по тегам
def run_extended_parsing():
    parser = AdvancedSevenDaysParser()
    try:
        parser.run(target_articles=1000)
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        parser.close()

if __name__ == "__main__":
    run_extended_parsing()