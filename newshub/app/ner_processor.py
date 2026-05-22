#легаси NER модуль - отключено

import structlog
from typing import Dict, List, Optional, Tuple

try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Doc,
    )
    _NATASHA_AVAILABLE = True
except ImportError:
    _NATASHA_AVAILABLE = False

log = structlog.get_logger()


class NERProcessor:
    """
    Класс для извлечения именованных сущностей из текста
    с использованием библиотеки Natasha
    """

    def __init__(self):
        """Инициализация моделей Natasha"""
        if not _NATASHA_AVAILABLE:
            log.warning("Natasha is not installed, NER disabled")
            self.segmenter = None
            self.morph_vocab = None
            self.emb = None
            self.morph_tagger = None
            self.ner_tagger = None
            return

        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            self.emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(self.emb)
            self.ner_tagger = NewsNERTagger(self.emb)
            log.info("NERProcessor initialized successfully")
        except Exception as e:
            log.error(f"Error initializing NERProcessor: {e}", exc_info=True)
            self.segmenter = None
            self.morph_vocab = None
            self.emb = None
            self.morph_tagger = None
            self.ner_tagger = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Извлечение именованных сущностей из текста
        
        Args:
            text: Текст для обработки
            
        Returns:
            Словарь с сущностями по типам: {'PER': [...], 'LOC': [...], 'ORG': [...]}
        """
        if not text or not isinstance(text, str):
            return {'PER': [], 'LOC': [], 'ORG': []}
        if not self.ner_tagger:
            return {'PER': [], 'LOC': [], 'ORG': []}

        try:
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)
            
            entities = {
                'PER': [],  # Персоны
                'LOC': [],  # Локации
                'ORG': []   # Организации
            }
            
            for span in doc.spans:
                if span.type in entities:
                    entities[span.type].append(span.text)
            
            return entities
        except Exception as e:
            log.error(f"Error extracting entities: {e}", exc_info=True)
            return {'PER': [], 'LOC': [], 'ORG': []}

    def extract_entities_with_offsets(self, text: str) -> Dict[str, List[Dict]]:
        """
        Извлечение сущностей с сохранением позиций в тексте
        
        Args:
            text: Текст для обработки
            
        Returns:
            Словарь с сущностями и их позициями:
            {
                'PER': [{'text': 'Иван Иванов', 'start': 0, 'end': 11, 'confidence': 0.95}],
                ...
            }
        """
        if not text or not isinstance(text, str):
            return {'PER': [], 'LOC': [], 'ORG': []}
        if not self.ner_tagger:
            return {'PER': [], 'LOC': [], 'ORG': []}

        try:
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)
            
            entities = {
                'PER': [],
                'LOC': [],
                'ORG': []
            }
            
            for span in doc.spans:
                if span.type in entities:
                    entities[span.type].append({
                        'text': span.text,
                        'start': span.start,
                        'end': span.stop,
                        'confidence': getattr(span, 'confidence', None)
                    })
            
            return entities
        except Exception as e:
            log.error(f"Error extracting entities with offsets: {e}", exc_info=True)
            return {'PER': [], 'LOC': [], 'ORG': []}

    @staticmethod
    def entity_overlap_score(entities1: Dict[str, List[str]], 
                            entities2: Dict[str, List[str]]) -> float:
        """
        Вычисление коэффициента пересечения сущностей (Jaccard similarity)
        
        Args:
            entities1: Первый набор сущностей
            entities2: Второй набор сущностей
            
        Returns:
            Число от 0 до 1, где 1 - полное совпадение, 0 - нет пересечений
        """
        score = 0
        valid_types = 0
        
        for entity_type in ['PER', 'LOC', 'ORG']:
            set1 = set(entities1.get(entity_type, []))
            set2 = set(entities2.get(entity_type, []))
            
            if set1 or set2:
                union = len(set1 | set2)
                if union > 0:
                    jaccard = len(set1 & set2) / union
                    score += jaccard
                    valid_types += 1
        
        return score / valid_types if valid_types > 0 else 0

    @staticmethod
    def entity_overlap_score_weighted(entities1: Dict[str, List[str]],
                                      entities2: Dict[str, List[str]],
                                      weights: Optional[Dict[str, float]] = None) -> float:
        """
        Взвешенное вычисление коэффициента пересечения сущностей
        
        Args:
            entities1: Первый набор сущностей
            entities2: Второй набор сущностей
            weights: Словарь весов для типов {'PER': 0.4, 'LOC': 0.3, 'ORG': 0.3}
            
        Returns:
            Взвешенное число от 0 до 1
        """
        if weights is None:
            weights = {'PER': 0.4, 'LOC': 0.3, 'ORG': 0.3}
        
        score = 0
        
        for entity_type in ['PER', 'LOC', 'ORG']:
            set1 = set(entities1.get(entity_type, []))
            set2 = set(entities2.get(entity_type, []))
            
            if set1 or set2:
                union = len(set1 | set2)
                if union > 0:
                    jaccard = len(set1 & set2) / union
                    score += weights.get(entity_type, 0) * jaccard
        
        return score

    @staticmethod
    def merge_entities(*entity_dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Объединение нескольких наборов сущностей
        
        Args:
            *entity_dicts: Переменное количество словарей сущностей
            
        Returns:
            Объединённый словарь без дубликатов
        """
        merged = {'PER': [], 'LOC': [], 'ORG': []}
        
        for entity_dict in entity_dicts:
            for entity_type in ['PER', 'LOC', 'ORG']:
                merged[entity_type].extend(entity_dict.get(entity_type, []))
        
        # Удаляем дубликаты, сохраняя порядок
        for entity_type in ['PER', 'LOC', 'ORG']:
            merged[entity_type] = list(dict.fromkeys(merged[entity_type]))
        
        return merged

    @staticmethod
    def get_entity_statistics(entities: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Получение статистики по сущностям
        
        Args:
            entities: Словарь сущностей
            
        Returns:
            Словарь со статистикой: {'PER': 5, 'LOC': 3, 'ORG': 2, 'total': 10}
        """
        stats = {
            'PER': len(entities.get('PER', [])),
            'LOC': len(entities.get('LOC', [])),
            'ORG': len(entities.get('ORG', []))
        }
        stats['total'] = stats['PER'] + stats['LOC'] + stats['ORG']
        return stats
