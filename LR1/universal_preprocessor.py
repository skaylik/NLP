import re
import json

class AdvancedTextCleaner:
    def __init__(self):
        # Паттерны для замены на токены
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)?\b')

        # Упрощенный словарь сокращений - только самые частые
        self.abbreviations = {
            # Общеязыковые сокращения
            'т.е.': 'то есть', 
            'т.д.': 'так далее', 
            'т.п.': 'тому подобное',
            'т.к.': 'так как', 
            'т.о.': 'таким образом',
            'т.н.': 'так называемый',
            'др.': 'другие', 
            'пр.': 'прочие',
            
            # Временные сокращения
            'г.': 'год', 
            'гг.': 'годы', 
            'в.': 'век', 
            'вв.': 'века',
            'н.э.': 'нашей эры',
            
            # Научные
            'см.': 'смотри',
            'рис.': 'рисунок', 
            'табл.': 'таблица',
        }

    def replace_with_tokens(self, text):
        """Замена числительных, URL и email на унифицированные токены"""
        if not text:
            return ""

        cleaned_text = text

        # Замена URL
        cleaned_text = self.url_pattern.sub('<URL>', cleaned_text)
        # Замена email
        cleaned_text = self.email_pattern.sub('<EMAIL>', cleaned_text)
        # Замена чисел (целых и дробных)
        cleaned_text = self.number_pattern.sub('<NUM>', cleaned_text)

        return cleaned_text

    def expand_abbreviations(self, text):
        """Обработка общеязыковых сокращений"""
        if not text:
            return ""

        # Создаем регулярное выражение для поиска сокращений
        sorted_abbr = sorted(self.abbreviations.keys(), key=len, reverse=True)
        pattern = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in sorted_abbr) + r')\b')

        def replace_match(match):
            return self.abbreviations[match.group(0)]

        # Заменяем все найденные сокращения
        cleaned_text = pattern.sub(replace_match, text)
        
        return cleaned_text

    def clean_text(self, text):
        """Основная функция очистки текста - ТОЛЬКО специфичные задачи"""
        if not text:
            return ""

        cleaned_text = text

        # Порядок обработки важен!
        # 1. Сначала заменяем URL, email и числа на токены
        cleaned_text = self.replace_with_tokens(cleaned_text)

        # 2. Затем раскрываем сокращения
        cleaned_text = self.expand_abbreviations(cleaned_text)

        return cleaned_text

def process_corpus(input_file, output_file):
    """
    Обрабатывает весь корпус статей с расширенной очисткой
    """
    cleaner = AdvancedTextCleaner()

    processed_count = 0
    total_articles = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    total_articles += 1
                    article = json.loads(line)

                    # Применяем расширенную очистку к тексту
                    original_text = article['text']
                    cleaned_text = cleaner.clean_text(original_text)
                    article['text'] = cleaned_text

                    # Записываем обратно
                    f_out.write(json.dumps(article, ensure_ascii=False) + '\n')
                    processed_count += 1

                    # Прогресс
                    if processed_count % 100 == 0:
                        print(f"Обработано статей: {processed_count}")

    print(f"\nОбработка завершена!")
    print(f"Всего статей: {total_articles}")
    print(f"Обработано: {processed_count}")
    print(f"Результат сохранен в: {output_file}")

# Основной блок выполнения
if __name__ == "__main__":
    process_corpus(
        'indicator_ru_corpus_cleaned.jsonl',
        'indicator_ru_corpus_advanced_cleaned.jsonl'
    )