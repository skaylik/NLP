import re
import html
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import json

# Загрузка ресурсов NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextCleaner:
    def __init__(self, to_lowercase=True, remove_stopwords=True):
        self.to_lowercase = to_lowercase
        self.remove_stopwords = remove_stopwords

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('russian'))

        # УБИРАЕМ паттерны для URL и email - они будут обрабатываться во втором файле
        # Оставляем только для служебных символов
        self.special_chars_pattern = re.compile(r'[^\w\s@.:/-]', re.UNICODE)  # Изменили - оставляем @ . : / для URL и email
        # Паттерн для множественных пробелов
        self.multispace_pattern = re.compile(r'\s+')

    def remove_html_and_ads(self, text):
        """Удаление HTML-разметки и рекламных блоков"""
        if not text:
            return ""
        
        # Декодируем HTML-сущности
        text = html.unescape(text)
        
        # Создаем BeautifulSoup объект для парсинга HTML
        soup = BeautifulSoup(text, "html.parser")
        
        # Удаляем скрипты, стили и потенциальные рекламные блоки
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'ad', 'advertisement']):
            element.decompose()
        
        # Получаем чистый текст
        clean_text = soup.get_text(separator=" ")
        return clean_text

    def remove_service_symbols(self, text):
        """Удаление только служебных символов, НЕ трогая URL и email"""
        if not text:
            return ""
        
        # Удаляем только специальные символы, но сохраняем символы нужные для URL и email
        # @ . : / - остаются для последующей обработки во втором файле
        text = self.special_chars_pattern.sub(' ', text)
        
        return text

    def normalize_whitespace(self, text):
        """Стандартизация пробельных символов"""
        if not text:
            return ""
        
        # Заменяем все пробельные символы (табы, переносы) на обычные пробелы
        text = self.multispace_pattern.sub(' ', text)
        # Убираем пробелы в начале и конце
        return text.strip()

    def to_lowercase_func(self, text):
        """Приведение текста к нижнему регистру"""
        if not text or not self.to_lowercase:
            return text
        return text.lower()

    def remove_stopwords_func(self, text):
        """Фильтрация стоп-слов"""
        if not text or not self.remove_stopwords:
            return text

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def clean_text(self, text):
        """Основной метод очистки текста"""
        if not text:
            return ""

        cleaned_text = text

        # Порядок обработки важен!
        # 1. Удаляем HTML и рекламу
        cleaned_text = self.remove_html_and_ads(cleaned_text)
        
        # 2. Удаляем только служебные символы, НЕ трогая URL и email
        cleaned_text = self.remove_service_symbols(cleaned_text)
        
        # 3. Стандартизируем пробелы
        cleaned_text = self.normalize_whitespace(cleaned_text)
        
        # 4. Приводим к нижнему регистру (если нужно)
        if self.to_lowercase:
            cleaned_text = self.to_lowercase_func(cleaned_text)
        
        # 5. Удаляем стоп-слова (если нужно)
        if self.remove_stopwords:
            cleaned_text = self.remove_stopwords_func(cleaned_text)

        return cleaned_text

def clean_corpus_file(input_file, output_file, to_lowercase=True, remove_stopwords=True):
    """
    Очищает корпус статей из JSONL файла
    
    Args:
        input_file (str): Входной JSONL файл
        output_file (str): Выходной JSONL файл
        to_lowercase (bool): Приводить к нижнему регистру
        remove_stopwords (bool): Удалять стоп-слова
    """
    cleaner = TextCleaner(
        to_lowercase=to_lowercase,
        remove_stopwords=remove_stopwords
    )

    cleaned_count = 0
    total_articles = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    total_articles += 1
                    article = json.loads(line)

                    # Очищаем текст
                    original_text = article['text']
                    cleaned_text = cleaner.clean_text(original_text)
                    article['text'] = cleaned_text

                    # Обновляем счетчик слов (опционально)
                    if 'word_count' in article:
                        article['word_count'] = len(cleaned_text.split())

                    # Записываем очищенную статью
                    f_out.write(json.dumps(article, ensure_ascii=False) + '\n')
                    cleaned_count += 1

                    # Прогресс (каждые 100 статей)
                    if cleaned_count % 100 == 0:
                        print(f"Обработано статей: {cleaned_count}")

    print(f"\nОчистка завершена!")
    print(f"Всего статей: {total_articles}")
    print(f"Очищено статей: {cleaned_count}")
    print(f"Файл сохранен: {output_file}")

# Использование
if __name__ == "__main__":
    clean_corpus_file(
        input_file='indicator_ru_corpus.jsonl',
        output_file='indicator_ru_corpus_cleaned.jsonl',
        to_lowercase=True,      # Можно изменить на False
        remove_stopwords=True   # Можно изменить на False
    )