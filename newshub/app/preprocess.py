import re
import logging
import pymorphy3

from razdel             import sentenize, tokenize
from typing             import List, Optional
from nltk.corpus        import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import spacy
except ImportError:
    spacy = None

class RussianTextPreprocessor:
    def __init__(self, use_nltk_stopwords: bool = True, 
                 additional_stopwords: List[str] = None):
        # Инициализация морфологического анализатора
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Загрузка модели spaCy для русского языка (опционально)
        if spacy is None:
            logger.warning("spaCy is not installed, skipping spaCy features")
            self.nlp = None
        else:
            try:
                self.nlp = spacy.load("ru_core_news_lg")
            except OSError:
                logger.warning(
                    "spaCy model 'ru_core_news_lg' not found. Install with: python -m spacy download ru_core_news_lg"
                )
                self.nlp = None
        
        # Загрузка стоп-слов
        self.stop_words = self._load_stop_words(use_nltk_stopwords, additional_stopwords)
    
    def _load_stop_words(self, use_nltk: bool, additional_stopwords: List[str]) -> set:
        """Загрузка стоп-слов из различных источников"""
        stop_words_set = set()
        
        # 1. NLTK стоп-слова (рекомендуется)
        if use_nltk:
            try:
                nltk_stopwords = set(stopwords.words('russian'))
                stop_words_set.update(nltk_stopwords)
                logger.info(f"Loaded {len(nltk_stopwords)} stopwords from NLTK")
            except LookupError:
                logger.warning("NLTK Russian stopwords not found. Run: nltk.download('stopwords')")
        
        # 2. Стоп-слова из spaCy (если доступно)
        if self.nlp:
            try:
                spacy_stopwords = set(self.nlp.Defaults.stop_words)
                stop_words_set.update(spacy_stopwords)
                logger.info(f"Added {len(spacy_stopwords)} stopwords from spaCy")
            except Exception as e:
                logger.warning(f"Could not load spaCy stopwords: {e}")
        
        # 3. Дополнительные пользовательские стоп-слова
        if additional_stopwords:
            stop_words_set.update(additional_stopwords)
            logger.info(f"Added {len(additional_stopwords)} custom stopwords")
        
        # 4. Дополнительные стоп-слова для новостных текстов
        news_specific_stopwords = {
            'com', 'www', 'http', 'https', 'html', 'php', 'руб', 'usd', 'eur',
            'км', 'кг', 'см', 'мм', 'млн', 'млрд', 'год', 'года', 'лет',
            'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль',
            'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь',
            'москва', 'россия', 'русский', 'said', 'would', 'could'
        }
        stop_words_set.update(news_specific_stopwords)
        
        logger.info(f"Total stopwords: {len(stop_words_set)}")
        return stop_words_set
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и форматирования"""
        if not text or not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление упоминаний и хештегов
        text = re.sub(r'[@#]\w+', '', text)
        
        # Удаление HTML тегов
        text = re.sub(r'<.*?>', '', text)
        
        # Удаление специальных символов, оставляем только буквы, цифры и основные знаки препинания
        text = re.sub(r'[^а-яёa-z0-9\s\.\,\!\?\-\:\(\)]', '', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Нормализация токенов (лемматизация)"""
        normalized = []
        for token in tokens:
            # Пропускаем стоп-слова
            if token in self.stop_words or len(token) <= 2:
                continue
            
            # Лемматизация с помощью pymorphy3
            try:
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                
                # Пропускаем короткие леммы и числа
                if len(lemma) > 2 and not lemma.isdigit():
                    normalized.append(lemma)
            except Exception as e:
                logger.debug(f"Error lemmatizing token '{token}': {e}")
                # В случае ошибки используем оригинальный токен
                if len(token) > 2 and not token.isdigit():
                    normalized.append(token)
        
        return normalized
    
    def tokenize_text(self, text: str) -> List[str]:
        """Токенизация текста с использованием razdel"""
        try:
            tokens = [token.text for token in tokenize(text)]
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Fallback на простую токенизацию
            return text.split()
    
    def process_post(self, text: str, 
                    min_length: int = 10,
                    max_length: int = 1000) -> Optional[str]:
        """
        Полная обработка поста
        """
        try:
            # Очистка текста
            cleaned_text = self.clean_text(text)
            
            if len(cleaned_text) < min_length:
                return None
            
            # Обрезка слишком длинных текстов
            if len(cleaned_text) > max_length:
                # Стараемся обрезать по границе предложения
                sentences = [sentence.text for sentence in sentenize(cleaned_text)]
                truncated_text = ""
                for sentence in sentences:
                    if len(truncated_text + sentence) <= max_length:
                        truncated_text += sentence + " "
                    else:
                        break
                cleaned_text = truncated_text.strip()
            
            # Токенизация
            tokens = self.tokenize_text(cleaned_text)
            
            # Нормализация
            normalized_tokens = self.normalize_tokens(tokens)
            
            # Возвращаем обработанный текст
            processed_text = " ".join(normalized_tokens)
            
            return processed_text if len(processed_text) >= min_length else None
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None
