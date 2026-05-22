import os
import re
import structlog
from pathlib import Path
import json

try:
    from catboost import CatBoostClassifier
except ImportError:  # optional embeddings dependency; rule-only tests should still import.
    CatBoostClassifier = None


logger = structlog.get_logger()

_MODEL = None
_FEATURE_META = None
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$")
TOKEN_RE = re.compile(r"[a-zа-я0-9\-]+", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d+")
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
UPPER_RE = re.compile(r"[A-ZА-ЯЁ]")
LOWER_RE = re.compile(r"[a-zа-яё]")

GENERIC_BY_TYPE = {
    "person": {
        "человек", "мужчина", "женщина", "девушка", "ребенок", "подросток",
        "пациентка", "депутат", "автор", "критик", "участник", "житель",
        "врач", "военный", "аналитик", "пользователь", "мама", "папа",
        "адвокат", "собственник", "водитель", "нарушитель", "гражданин",
        "губернатор", "президент", "погибший", "раненый", "ученый",
        "агент", "мошенник", "россиянин", "россиянка", "неизвестный",
        "волонтер", "синоптик", "банкир", "топ-менеджер", "глава",
        "иммигрант", "дипломат", "военнослужащий", "гинеколог",
        "специалист", "курильщик", "посланник", "сотрудник", "доктор",
        "главврач", "инженер", "эксперт", "отец", "предприниматель",
        "потерпевший", "парень", "актер", "актёр",
    },
    "organization": {
        "комиссия", "правительство", "прокуратура", "служба", "ведомство",
        "министерство", "компания", "организация", "сервис", "центр",
        "армия", "завод", "власть", "бюджет", "банк", "парламент",
        "фонд", "суд", "студия", "разведка", "комитет", "командование",
        "генконсульство", "посольство", "лига", "издание", "газета",
        "портал", "канал", "источник", "общество", "компания", "вуз", "дилер",
    },
    "location": {
        "город", "центр", "регион", "страна", "дом", "многоквартирный дом",
        "рубеж", "столица", "аэродром", "аэропорт", "область", "край",
        "коровник", "граница", "берег", "море", "база", "остров", "двор",
        "небо", "акватория", "побережье", "улица", "ресторан", "дом", "центр", 
        "офис",
    },
    "event": {
        "встреча", "депортация", "катастрофа", "призыв", "война",
        "демонстрация", "переговоры", "конфликт", "удар", "перебой",
        "санкция", "совещание", "революция", "выборы", "пожар",
        "дружба", "гонка", "операция", "апокалипсис",
    },
    "law": {
        "право", "закон", "законопроект", "законодательство", "правило",
        "иск", "налог",
    },
    "product": {
        "сервис", "иномарка", "облигация", "бонд", "бензин", "топливо",
        "продукт", "товар", "талисман", "пиво", "цена", "платье",
        "нефть",
    },
}

DOMAIN_LABEL_TO_BASE = {
    "political party": "organization",
    "company": "organization",
    "startup": "organization",
    "institution": "organization",
    "military unit": "organization",
    "athlete": "person",
    "coach": "person",
    "diplomat": "person",
    "territory": "location",
    "election": "event",
    "deal": "event",
    "bankruptcy": "event",
    "operation": "event",
    "ceasefire": "event",
    "tournament": "event",
    "transfer": "event",
    "data breach": "event",
    "discovery": "event",
    "experiment": "event",
    "natural disaster": "event",
    "casualty": "event",
    "rescue operation": "event",
    "treaty": "law",
    "sanction": "law",
    "weapon": "product",
    "stock": "product",
    "currency": "product",
    "algorithm": "product",
    "publication": "product",
    "patent": "law",
}

EVENT_SUBTYPE_PROMOTE = {
    "election",
    "deal",
    "bankruptcy",
    "operation",
    "ceasefire",
    "tournament",
    "transfer",
    "data breach",
    "discovery",
    "experiment",
    "natural disaster",
    "casualty",
    "rescue operation",
}

LAW_SUBTYPE_PROMOTE = {
    "treaty",
    "sanction",
    "patent",
}

PRODUCT_SUBTYPE_PROMOTE = {
    "weapon",
    "stock",
    "currency",
    "algorithm",
    "publication",
}

ORG_SUBTYPE_PROMOTE = {
    "company",
    "startup",
    "institution",
    "political party",
    "military unit",
}

DESCRIPTOR_PREFIXES = {
    "российский", "российская", "российское", "российские",
    "украинский", "украинская", "украинское", "украинские",
    "американский", "американская", "американское", "американские",
    "израильский", "израильская",
    "иранский", "иранская",
    "китайский", "китайская",
    "международный", "международная",
    "федеральный", "федеральная",
    "областной", "областная",
    "местный", "местная", "местное", "местные",
    "главный", "главная",
    "центральный", "центральная",
    "опытный", "опытная",
    "специальный", "специальная",
    "военный", "военная",
    "дипломатический", "дипломатическая",
    "антидиффамационный", "антидиффамационная",
    "нижегородский", "нижегородская",
    "краснодарский", "краснодарская",
    "ростовский", "ростовская",
    "перинатальный", "перинатальная",
    "стриминговый", "стриминговая",
}

ORG_GENERIC_HEADS = {
    "компания", "служба", "организация", "ведомство", "центр",
    "завод", "фонд", "парламент", "суд", "студия", "банк", "армия",
    "разведка", "комитет", "командование", "генконсульство",
    "посольство", "лига", "издание", "газета", "портал", "канал",
    "сервис", "источник",
}

VALID_SHORT_ORGS = {
    "оон", "фас", "вшэ", "дпс", "нпз", "росстат", "тасс", "емисс",
    "amazon", "adnoc", "минэнерго", "куйбышевазот",
    "cnn", "reuters", "рбк", "нтв-плюс", "триколор",
    "telegram", "телеграм", "pvo", "пво", "apple", "bmw", "spotify", "netflix",
    "hbo", "paramount", "автоваз", "минфин", "госдума",
    "минпромторг", "росстандарт", "мвд", "минтранс",
    "еаэс", "polymarket", "google", "газпром", "минобороны", "nvidia", "wildberries", "сбер",
    "втб", "газпромбанк", "ржд", "мтс", "мфц", "мэа", "фсб",
    "евросоюз", "amazon", "microsoft", "oracle", "spacex", "hsbc",
    "barclays", "santander", "inditex", "heineken", "spotify",
    "netflix", "telegram", "apple", "bmw", "hbo", "paramount",
    "polymarket", "альфа-банк", "аэрофлот", "балтика",
    "динамо", "зенит", "спартак", "цска", "челси", "реал",
    "бенфика", "рубин", "факел",
    "мид", "github", "openai", "sony", "maersk",
    "роснефть", "хизбалл", "хезболла",
}

VALID_SHORT_PERSONS = {
    "путин", "трамп", "лавров", "собянин", "галицкий", "зеленский",
    "цукерберг", "пушилин", "дуров", "маск", "собчак",
    "захарова", "бастрыкин", "богомаз", "нетаньяху", "хегсет",
    "макрон", "сурков", "кадыров", "кац", "месси", "орбан", "милнер", "утяшева",
    "костылев", "зобнина", "мелкадзе", "большунов", "евсеев",
    "ющенко", "кейн", "шойгу", "сталин", "цаликов",
}

VALID_SHORT_PRODUCTS = {
    "iphone", "macbook", "bmw", "bmw x5 m", "lada",
    "биткоин", "биткоина", "электровелосипед", "ethereum", 
    "iphone", "ботокс",
}

KNOWN_LOCATIONS = {
    "сша", "иран", "израиль", "украина", "россия", "китай", "индия",
    "бахрейн", "манама", "тегеран", "вашингтон", "париж", "брянск",
    "славянск", "якутия", "москва", "европа", "флорида",
    "дубай", "италия", "австралия", "турция", "германия", "франция",
    "испания", "англия", "великобритания", "казахстан", "япония",
    "бразилия", "канада",
    "красноярск", "бейрут", "катар", "сочи", "осака", "венесуэла",
    "сан-франциско", "волгоград", "эр-рияда", "тель-авив",
    "балашиха", "люксембург", "долгопрудный",
}

NO_ML_PROMOTE_OVERRIDE_REASONS = {
    "source_mention",
    "generic_person_phrase",
    "single_token_person",
    "generic_org",
    "single_token_org",
    "generic_location",
    "generic_event",
    "generic_product",
    "generic_law",
}

ML_PROMOTE_OVERRIDE_PROBA_THRESHOLD = 0.75
SINGLE_TOKEN_LOCATION_KEEP_PROBA_THRESHOLD = 0.85

ROLE_NOUNS = {
    "губернатор", "президент", "гражданин", "водитель", "нарушитель",
    "погибший", "раненый", "ученый", "агент", "житель", "военный",
    "теннисистка", "ребенок", "адвокат", "студентка", "банкир",
    "инженер", "синоптик", "волонтер", "топ-менеджер", "глава",
    "иммигрант", "дипломат", "военнослужащий", "гинеколог",
    "специалист", "курильщик", "посланник", "сотрудник", "доктор",
    "главврач", "эксперт", "министр", "врач", "конструктор",
    "человек", "отец", "предприниматель", "потерпевший", "парень",
    "директор", "аналитик", "кандидат", "разработчик", "чиновник",
    "учредитель", "продавец", "фотограф", "жених", "невеста",
    "клиент", "студент", "выпускник",
}

NUMERIC_PERSON_WORDS = {
    "один", "одна", "два", "две", "три", "четыре", "пять", "шесть",
    "семь", "восемь", "девять", "десять",
}

SOURCE_ORGS = {
    "the bell", "reuters", "cnn", "рбк", "тасс", "интерфакс",
    "meduza", "медуза", "axios", "forbes", "bbc", "ap",
    "associated press", "the washington post", "cbs news",
    "the hill", "коммерсантъ", "abc news", "ynet",
    "the wall street journal", "wsj", "nbc news", "fox news",
}

SOURCE_CUES = {
    "по данным", "сообщает", "сообщил", "сообщила", "пишет", "передает",
    "по информации", "как пишет", "как сообщает", "издание", "газета",
    "телеграм-канал", "канал", "по сообщению", "рассказал изданию",
    "со ссылкой на", "по версии", "по оценке",
}

WEAK_ORG_PHRASES = {
    "международный терроризм",
    "израильский американский источник",
    "стриминговый сервис",
    "армия мир",
    "режим аятолл",
    "американский разведка",
    "центральный командование",
    "российский генконсульство",
    "американский антидиффамационный лига",
    "военный сша",
    "иранский режим",
    "иранский народ",
    "российский интернет",
    "армия израиль",
    "строительный вуз"
}

GENERIC_PERSON_PHRASES = {
    "молодой человек",
    "главный конструктор",
    "главный внештатный специалист",
    "опытный гинеколог",
    "американский военнослужащий",
    "толстый курильщик",
    "иранский дипломат",
    "господин орбан",
    "красивый американец",
    "хаменея школьник",
    "генеральный директор",
    "исполнительный директор",
    "опытный специалист",
}

PERSON_LIKE_TOKENS = {
    "хаменея", "орбан", "нетаньяху", "макрон", "трамп", "путин",
    "зеленский", "хегсет", "сурков", "дуров", "маск",
}

LOCATION_BLOCKLIST_SINGLE = {
    "граница", "берег", "море", "база", "остров", "двор", "небо",
}

ORG_GENERIC_MENTION_ONLY = {
    "прокуратура",
}

ORG_WEAK_PROMOTE_EXCEPTIONS = {
    "армия израиль",
}

PRODUCT_REJECT_SINGLE = {
    "вклад",
}

PRODUCT_MENTION_EXCEPTIONS = {
    "нефть",
}

ROMAN_NUMERAL_TOKENS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
}

STORYLINE_PRODUCT_CONTEXT_CUES = {
    "миссия", "экспедиция", "программа", "экипаж", "запуск",
    "полет", "полёт", "корабль", "ракета", "орбита", "модуль",
    "капсула", "версия", "серия", "модель", "платформа",
}

STORYLINE_PRODUCT_ANCHOR_HEADS = {
    "модуль", "программа", "миссия", "корабль", "ракета", "капсула",
    "платформа", "система", "станция", "комплекс", "аппарат",
}

ORG_SUFFIX_BLOCKLIST = {
    "режим", "народ", "интернет",
}

ROLE_NOUNS = {
    "директор", "специалист", "аналитик", "кандидат", "разработчик",
    "чиновник", "учредитель", "продавец", "фотограф", "жених",
    "невеста", "клиент", "студент", "выпускник",
}

ORG_GENERIC_HEADS = {
    "компания", "служба", "организация", "ведомство", "центр",
    "завод", "фонд", "парламент", "суд", "студия", "банк", "армия",
    "разведка", "комитет", "командование", "генконсульство",
    "посольство", "лига", "издание", "газета", "портал", "канал",
    "сервис", "источник", "общество", "вуз", "агентство", "министерство",
}

def get_decision_model():
    global _MODEL, _FEATURE_META
    if _MODEL is not None:
        return _MODEL, _FEATURE_META

    art_dir = Path(os.getenv("ENTITY_DECISION_ARTIFACTS", "/app/embeddings_service"))
    model_path = art_dir / "entity_decision_catboost.cbm"
    meta_path = art_dir / "entity_decision_feature_cols.json"

    if CatBoostClassifier is None:
        raise RuntimeError("catboost is required to load entity decision model")

    model = CatBoostClassifier()
    model.load_model(model_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _MODEL = model
    _FEATURE_META = meta
    logger.info(
        "entity_decision_model_loaded",
        model_path=str(model_path),
        feature_meta=meta,
    )
    return _MODEL, _FEATURE_META

def _build_model_features_from_dict(feat: dict, meta: dict) -> list:
    cols = meta["feature_cols"]
    row = []
    for c in cols:
        v = feat.get(c)
        row.append(v)
    return row

def has_role_noun(words: list[str]) -> bool:
    return any(w in ROLE_NOUNS for w in words)

def has_generic_org_head(words: list[str]) -> bool:
    return bool(words) and words[-1] in ORG_GENERIC_HEADS

def looks_like_role_phrase(words: list[str]) -> bool:
    if not words:
        return False
    if len(words) <= 3 and has_role_noun(words):
        if not any(w in VALID_SHORT_PERSONS for w in words):
            return True
    return False

def looks_like_descriptor_generic_org(words: list[str]) -> bool:
    if len(words) < 2:
        return False
    if words[0] in DESCRIPTOR_PREFIXES and words[-1] in ORG_GENERIC_HEADS:
        return True
    return False

def normalize_entity_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("ё", "е")
    t = PUNCT_EDGE_RE.sub("", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")

def is_all_caps_like(text: str) -> bool:
    s = (text or "").strip()
    has_upper = bool(UPPER_RE.search(s))
    has_lower = bool(LOWER_RE.search(s))
    return has_upper and not has_lower

def canonicalize_person_name(norm: str) -> str:
    aliases = {
        "путин": "владимир путин",
        "трамп": "дональд трамп",
        "зеленский": "владимир зеленский",
        "дуров": "павел дуров",
        "маск": "илон маск",
        "собчак": "ксения собчак",
        "захарова": "мария захарова",
        "бастрыкин": "александр бастрыкин",
        "богомаз": "александр богомаз",
        "нетаньяху": "биньямин нетаньяху",
        "макрон": "эммануэль макрон",
        "сурков": "владислав сурков",
        "хегсет": "пит хегсет",
    }
    return aliases.get(norm, norm)

def canonical_entity_text(text: str, label: str) -> str:
    norm = normalize_entity_text(text)
    if label == "person":
        return canonicalize_person_name(norm)
    return norm

def canonical_entity_key(text: str, label: str) -> str:
    canon = canonical_entity_text(text, label)
    return f"{label}:{canon.replace(' ', '_')}"


def contextual_canonical_entity_text(
    text: str,
    label: str,
    *,
    full_text: str = "",
    subtype: str = "",
) -> str:
    norm = canonical_entity_text(text, label)
    if label != "product":
        return norm
    words = tokenize(norm)
    if subtype in PRODUCT_SUBTYPE_PROMOTE:
        return norm
    base = _storyline_product_single_token_base_form(text, norm, words, full_text)
    return base or norm


def contextual_canonical_entity_key(
    text: str,
    label: str,
    *,
    full_text: str = "",
    subtype: str = "",
) -> str:
    canon = contextual_canonical_entity_text(
        text,
        label,
        full_text=full_text,
        subtype=subtype,
    )
    return f"{label}:{canon.replace(' ', '_')}"


def canonicalize_entity_label(label: str) -> tuple[str, str]:
    raw = normalize_entity_text(label)
    if not raw:
        return "", ""
    base = DOMAIN_LABEL_TO_BASE.get(raw, raw)
    subtype = raw if raw != base else ""
    return base, subtype

def looks_like_descriptor_only_org(words: list[str]) -> bool:
    if not words:
        return False
    if len(words) == 1 and words[0] in GENERIC_BY_TYPE["organization"]:
        return True
    if len(words) == 2 and words[0] in DESCRIPTOR_PREFIXES and words[1] in ORG_GENERIC_HEADS:
        return True
    if len(words) == 2 and words[1] in ORG_SUFFIX_BLOCKLIST and words[0] in DESCRIPTOR_PREFIXES:
        return True
    if len(words) == 2 and words[1] in ORG_GENERIC_HEADS and words[0].endswith(("ский", "ская", "ское", "ые", "ий", "ая")):
        return True
    if len(words) >= 3 and words[-1] in ORG_GENERIC_HEADS and any(w in DESCRIPTOR_PREFIXES for w in words[:-1]):
        return True
    return False

def looks_like_numeric_person(words: list[str]) -> bool:
    if not words:
        return False
    if words[0] in NUMERIC_PERSON_WORDS:
        return True
    if DIGIT_RE.fullmatch(words[0] or ""):
        return True
    return False

def looks_like_suspicious_person(norm: str, words: list[str]) -> bool:
    if not words:
        return False
    if norm in GENERIC_PERSON_PHRASES:
        return True
    if len(words) == 2 and words[0] == "the":
        return True
    if looks_like_numeric_person(words):
        return True
    if words[-1] in ROLE_NOUNS:
        return True
    if len(words) >= 2 and words[0] in DESCRIPTOR_PREFIXES and words[-1] in ROLE_NOUNS:
        return True
    if len(words) >= 2 and any(w in ROLE_NOUNS for w in words):
        return True
    return False

def looks_like_bad_location(norm: str, words: list[str]) -> bool:
    if norm in GENERIC_BY_TYPE["location"]:
        return True
    if len(words) == 2 and words[1] in {"область", "край"} and words[0].endswith(("ский", "ий")):
        return True
    if words and words[-1] in {"пространстве", "пространство", "орбите", "орбита", "космосе", "космос", "борту"}:
        return True
    return False


def _storyline_product_single_token_base_form(
    raw_text: str,
    norm: str,
    words: list[str],
    full_text: str = "",
) -> str:
    if len(words) != 1:
        return ""
    if not norm or len(norm) < 5:
        return ""
    if LATIN_RE.search(raw_text or ""):
        return ""
    if not CYRILLIC_RE.search(raw_text or ""):
        return ""
    if norm in GENERIC_BY_TYPE["product"] or norm in PRODUCT_REJECT_SINGLE:
        return ""

    full_norm = normalize_entity_text(full_text)
    if not any(cue in full_norm for cue in STORYLINE_PRODUCT_CONTEXT_CUES):
        return ""

    raw = (raw_text or "").strip()
    looks_named = bool(raw[:1].isupper()) or bool(DIGIT_RE.search(raw)) or "-" in raw
    if not looks_named:
        return ""

    candidates: list[str] = []
    for suffix_len in (3, 2, 1):
        if len(norm) - suffix_len < 5:
            continue
        suffix = norm[-suffix_len:]
        if suffix_len == 3 and suffix in {"ами", "ями", "ого", "ему", "ому", "иях"}:
            candidates.append(norm[:-suffix_len])
        elif suffix_len == 2 and suffix in {"ом", "ем", "ой", "ою", "ев", "ов", "ам", "ям", "ах", "ях"}:
            candidates.append(norm[:-suffix_len])
        elif suffix_len == 1 and suffix in {"а", "я", "у", "ю", "е", "и", "ы"}:
            candidates.append(norm[:-suffix_len])

    for candidate in candidates:
        candidate = candidate.strip("- ")
        if len(candidate) < 5:
            continue
        if candidate in GENERIC_BY_TYPE["product"] or candidate in PRODUCT_REJECT_SINGLE:
            continue
        return candidate
    return ""


def looks_like_storyline_anchor_product(
    raw_text: str,
    norm: str,
    words: list[str],
    full_text: str = "",
) -> bool:
    if _storyline_product_single_token_base_form(raw_text, norm, words, full_text):
        return True
    if len(words) < 2:
        return False
    has_latin = bool(LATIN_RE.search(raw_text or ""))
    has_numeric_marker = bool(DIGIT_RE.search(raw_text or "")) or any(w in ROMAN_NUMERAL_TOKENS for w in words)
    full_norm = normalize_entity_text(full_text)
    has_storyline_context = any(cue in full_norm for cue in STORYLINE_PRODUCT_CONTEXT_CUES)
    if has_numeric_marker and (has_latin or has_storyline_context):
        return True

    head = words[0]
    tail_tokens = words[1:]
    if (
        has_storyline_context
        and head in STORYLINE_PRODUCT_ANCHOR_HEADS
        and tail_tokens
        and all(token not in GENERIC_BY_TYPE["product"] for token in tail_tokens)
        and any(len(token) >= 4 for token in tail_tokens)
    ):
        return True
    return False


def looks_like_storyline_anchor_law(norm: str, words: list[str], full_text: str = "", subtype: str = "") -> bool:
    if not norm or norm in GENERIC_BY_TYPE["law"]:
        return False
    if subtype in LAW_SUBTYPE_PROMOTE and len(words) >= 1:
        return True
    if len(words) >= 2:
        return True
    text = normalize_entity_text(full_text)
    if not text:
        return False
    return any(
        cue in text
        for cue in {
            "закон о",
            "законопроект о",
            "поправк",
            "кодекс",
            "бюджетн",
            "налогов",
            "санкцион",
        }
    )


def looks_like_storyline_anchor_event(norm: str, words: list[str], full_text: str = "", subtype: str = "") -> bool:
    if not norm or norm in GENERIC_BY_TYPE["event"]:
        return False
    if subtype in EVENT_SUBTYPE_PROMOTE and (len(words) >= 1 or " " in subtype):
        return True
    if len(words) >= 2:
        return True
    text = normalize_entity_text(full_text)
    if not text:
        return False
    return any(
        cue in text
        for cue in {
            "операция",
            "миссия",
            "кампания",
            "саммит",
            "перемирие",
            "референдум",
            "выборы",
            "форум",
            "турнир",
            "экспедиция",
        }
    )


def looks_like_valid_short_org(raw_text: str, norm: str) -> bool:
    if norm in VALID_SHORT_ORGS:
        return True
    if not norm or len(tokenize(norm)) != 1:
        return False
    if not LATIN_RE.search(raw_text or ""):
        return False
    if not is_all_caps_like(raw_text or ""):
        return False
    return 3 <= len(norm) <= 6


def is_storyline_anchor_candidate(
    *,
    raw_text: str,
    label: str,
    subtype: str = "",
    decision: str,
    reason: str,
    features: dict | None = None,
    full_text: str = "",
) -> bool:
    features = features or {}
    norm = normalize_entity_text(raw_text)
    words = tokenize(norm)

    if not raw_text or not label:
        return False
    if features.get("source_cue_hit") and label not in {"organization", "law", "event"}:
        return False

    if decision == "promote":
        if label == "location" and looks_like_bad_location(norm, words):
            return False
        return True

    if decision != "mention":
        return False

    if label == "organization":
        if subtype in ORG_SUBTYPE_PROMOTE and len(words) >= 1:
            return True
        if looks_like_valid_short_org(raw_text, norm):
            return True
        if features.get("source_cue_hit"):
            return False
        if len(words) >= 2 and not looks_like_descriptor_generic_org(words) and not looks_like_descriptor_only_org(words):
            if norm not in GENERIC_BY_TYPE["organization"] and norm not in WEAK_ORG_PHRASES and norm not in ORG_GENERIC_MENTION_ONLY:
                return True
        return False

    if label == "product":
        if subtype in PRODUCT_SUBTYPE_PROMOTE and len(words) >= 1:
            return True
        if looks_like_storyline_anchor_product(raw_text, norm, words, full_text):
            return True
        if len(words) == 1 and norm in VALID_SHORT_PRODUCTS:
            return True
        return False

    if label == "person":
        return reason in {"valid_short_person", "named_person"}

    if label == "location":
        return reason in {"known_location"}

    if label == "law":
        return looks_like_storyline_anchor_law(norm, words, full_text, subtype)

    if label == "event":
        return looks_like_storyline_anchor_event(norm, words, full_text, subtype)

    return False


def is_rag_evidence_candidate(
    *,
    raw_text: str,
    label: str,
    subtype: str = "",
    decision: str,
    reason: str,
    features: dict | None = None,
    full_text: str = "",
) -> bool:
    features = features or {}
    norm = normalize_entity_text(raw_text)
    words = tokenize(norm)

    if not raw_text or not label:
        return False
    if decision == "reject":
        return False

    if label == "person":
        return reason in {"valid_short_person", "named_person"} or (decision == "promote" and len(words) >= 2)

    if label == "organization":
        if norm in GENERIC_BY_TYPE["organization"] or norm in WEAK_ORG_PHRASES:
            return False
        if looks_like_descriptor_generic_org(words) or looks_like_descriptor_only_org(words):
            return False
        if looks_like_valid_short_org(raw_text, norm):
            return True
        if subtype in ORG_SUBTYPE_PROMOTE:
            return True
        if norm in ORG_GENERIC_MENTION_ONLY:
            return False
        if len(words) >= 2:
            return True
        return bool(features.get("source_cue_hit") and len(words) == 1 and looks_like_valid_short_org(raw_text, norm))

    if label == "location":
        if looks_like_bad_location(norm, words):
            return False
        return reason in {"known_location", "named_location"} or decision == "promote"

    if label == "product":
        if len(words) == 1 and norm in PRODUCT_REJECT_SINGLE:
            return False
        if norm in GENERIC_BY_TYPE["product"]:
            return False
        if subtype in PRODUCT_SUBTYPE_PROMOTE:
            return True
        if looks_like_storyline_anchor_product(raw_text, norm, words, full_text):
            return True
        if len(words) >= 2:
            return True
        return len(words) == 1 and norm in VALID_SHORT_PRODUCTS

    if label == "law":
        if norm in GENERIC_BY_TYPE["law"]:
            return False
        return (
            subtype in LAW_SUBTYPE_PROMOTE
            or looks_like_storyline_anchor_law(norm, words, full_text, subtype)
            or len(words) >= 2
        )

    if label == "event":
        if norm in GENERIC_BY_TYPE["event"]:
            return False
        return (
            subtype in EVENT_SUBTYPE_PROMOTE
            or looks_like_storyline_anchor_event(norm, words, full_text, subtype)
            or len(words) >= 2
        )

    return False

def is_source_mention(norm: str, label: str, full_text: str) -> bool:
    if label != "organization":
        return False
    if norm not in SOURCE_ORGS:
        return False

    text = normalize_entity_text(full_text)
    if not text:
        return False

    idx = text.find(norm)
    if idx >= 0:
        left = text[max(0, idx - 80):idx]
        right = text[idx:idx + 80]
        context = f"{left} {right}"
        for cue in SOURCE_CUES:
            if cue in context:
                return True

    return False

# --- patch classify_candidate() ---

def classify_candidate(text: str, label: str, full_text: str = "", subtype: str = "") -> tuple[str, str]:
    norm = normalize_entity_text(text)
    words = tokenize(norm)

    if not norm or not label:
        return "reject", "empty"
    if not words:
        return "reject", "no_tokens"
    if len(words) > 8:
        return "reject", "too_long"
    if norm.isdigit():
        return "reject", "digit_only"

    if label == "person":
        if DIGIT_RE.search(text or "") and any(w in GENERIC_BY_TYPE["person"] for w in words):
            return "reject", "generic_person_phrase"
        if norm in GENERIC_BY_TYPE["person"] or norm in GENERIC_PERSON_PHRASES:
            return "reject", "generic_person_phrase"
        if looks_like_role_phrase(words):
            return "mention", "role_phrase"
        if len(words) == 1:
            if norm in VALID_SHORT_PERSONS:
                return "promote", "valid_short_person"
            return "mention", "single_token_person"
        if looks_like_suspicious_person(norm, words):
            return "mention", "suspicious_person_phrase"
        return "promote", "named_person"

    if label == "organization":
        if norm in VALID_SHORT_PERSONS:
            return "mention", "person_like_org"
        if is_source_mention(norm, label, full_text):
            return "mention", "source_mention"
        if subtype in ORG_SUBTYPE_PROMOTE and len(words) >= 1 and norm not in WEAK_ORG_PHRASES:
            if len(words) == 1 and not looks_like_valid_short_org(text, norm):
                return "mention", "single_token_org"
            return "promote", "typed_org"
        if norm in ORG_WEAK_PROMOTE_EXCEPTIONS:
            return "promote", "weak_org_exception_promote"
        if norm in WEAK_ORG_PHRASES:
            return "reject", "weak_org_phrase"
        if norm in ORG_GENERIC_MENTION_ONLY:
            return "mention", "generic_org_mention_only"
        if norm in GENERIC_BY_TYPE["organization"]:
            return "reject", "generic_org"
        if len(words) == 1:
            if looks_like_valid_short_org(text, norm):
                return "promote", "valid_short_org"
            return "mention", "single_token_org"
        if looks_like_descriptor_generic_org(words) or looks_like_descriptor_only_org(words):
            return "mention", "descriptor_only_org"
        return "promote", "named_org"

    if label == "location":
        if norm in VALID_SHORT_PERSONS or norm in PERSON_LIKE_TOKENS:
            return "mention", "person_like_location"
        if norm in KNOWN_LOCATIONS:
            return "promote", "known_location"
        if norm in GENERIC_BY_TYPE["location"]:
            return "reject", "generic_location"
        if len(words) == 1 and norm in LOCATION_BLOCKLIST_SINGLE:
            return "reject", "generic_location"
        if len(words) == 1:
            return "promote", "single_token_location"
        if looks_like_bad_location(norm, words):
            return "mention", "suspicious_location_form"
        return "promote", "named_location"

    if label == "product":
        if len(words) == 1 and norm in PRODUCT_REJECT_SINGLE:
            return "reject", "generic_product"
        if norm in PRODUCT_MENTION_EXCEPTIONS:
            return "mention", "product_mention_only"
        if norm in GENERIC_BY_TYPE["product"]:
            return "reject", "generic_product"
        if subtype in PRODUCT_SUBTYPE_PROMOTE and (len(words) >= 2 or bool(DIGIT_RE.search(text or ""))):
            return "promote", "typed_product"
        if _storyline_product_single_token_base_form(text, norm, words, full_text):
            return "promote", "named_product_topic"
        if looks_like_storyline_anchor_product(text, norm, words, full_text):
            return "mention", "named_product_topic"
        if len(words) == 1 and norm in VALID_SHORT_PRODUCTS:
            return "promote", "valid_short_product"
        if len(words) == 1:
            return "mention", "single_token_product"
        return "mention", "product_mention_only"

    if label == "law":
        if norm in GENERIC_BY_TYPE["law"]:
            return "reject", "generic_law"
        if subtype in LAW_SUBTYPE_PROMOTE and looks_like_storyline_anchor_law(norm, words, full_text, subtype):
            return "promote", "typed_law"
        if looks_like_storyline_anchor_law(norm, words, full_text, subtype):
            return "mention", "law_mention_only"
        return "mention", "law_mention_only"

    if label == "event":
        if norm in GENERIC_BY_TYPE["event"]:
            return "mention", "generic_event"
        if subtype in EVENT_SUBTYPE_PROMOTE and looks_like_storyline_anchor_event(norm, words, full_text, subtype):
            return "promote", "typed_event"
        if looks_like_storyline_anchor_event(norm, words, full_text, subtype):
            return "mention", "event_mention_only"
        if len(words) == 1:
            return "mention", "single_token_event"
        return "mention", "event_mention_only"

    return "mention", "fallback"

def build_entity_features(entity: dict, norm: str, label: str, full_text: str = "", subtype: str = "") -> dict:
    raw_text = entity.get("text", "")
    words = tokenize(norm)

    start = entity.get("start")
    end = entity.get("end")
    score = entity.get("score")

    text_len = len(full_text or "")
    span_char_len = None
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        span_char_len = end - start

    return {
        "raw_text": raw_text,
        "normalized_text": norm,
        "label": label,
        "entity_subtype": subtype,

        "gliner_score": round(float(score), 6) if score is not None else None,
        "span_start": start if isinstance(start, int) else None,
        "span_end": end if isinstance(end, int) else None,
        "span_char_len": span_char_len,

        "raw_char_len": len(raw_text or ""),
        "text_char_len": text_len,
        "word_count_raw": len(tokenize((raw_text or "").lower())),
        "word_count_norm": len(words),

        "token_count": len(words),
        "is_single_token": len(words) == 1,

        "contains_digit": bool(DIGIT_RE.search(raw_text or "")),
        "contains_latin": bool(LATIN_RE.search(raw_text or "")),
        "contains_cyrillic": bool(CYRILLIC_RE.search(raw_text or "")),
        "contains_hyphen": "-" in (raw_text or ""),
        "is_all_caps_like": is_all_caps_like(raw_text or ""),

        "in_first_200_chars": isinstance(start, int) and start < 200,

        "has_role_noun": has_role_noun(words),
        "has_generic_head": has_generic_org_head(words),
        "has_descriptor_prefix": bool(words and words[0] in DESCRIPTOR_PREFIXES),
        "alias_hit_person": norm in VALID_SHORT_PERSONS,
        "alias_hit_org": norm in VALID_SHORT_ORGS,
        "known_location_hit": norm in KNOWN_LOCATIONS,
        "mapped_domain_label": 1 if subtype else 0,
        "source_cue_hit": is_source_mention(norm, label, full_text),
        "looks_like_role_phrase": looks_like_role_phrase(words),
        "looks_like_descriptor_org": looks_like_descriptor_generic_org(words),
        "in_weak_org_phrases": norm in WEAK_ORG_PHRASES,
        "in_generic_person_phrases": norm in GENERIC_PERSON_PHRASES,
        "in_generic_bucket": norm in GENERIC_BY_TYPE.get(label, set()),
    }

def partition_entities(
    entities: list[dict],
    full_text: str = "",
    post_id: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    promoted = []
    mentions = []
    rejected = []
    seen_promoted = set()
    seen_mentions = set()

    # Load once per post/task. If unavailable, fall back to rules without noisy per-entity errors.
    model = None
    meta = None
    try:
        model, meta = get_decision_model()
    except Exception as exc:
        logger.error("entity_decision_model_load_failed", post_id=post_id, exc=str(exc))

    for e in entities:
        raw_text = e.get("text", "")
        extracted_label = (e.get("original_label") or e.get("label") or "").strip().lower()
        label, subtype = canonicalize_entity_label(extracted_label)

        norm = normalize_entity_text(raw_text)
        label_rescued = False
        if label == "organization" and norm in KNOWN_LOCATIONS:
            label = "location"
            label_rescued = True

        rule_verdict, rule_reason = classify_candidate(raw_text, label, full_text=full_text, subtype=subtype)
        features = build_entity_features(e, norm, label, full_text, subtype)
        final_verdict = rule_verdict
        final_reason = rule_reason

        item = dict(e)
        item["raw_text"] = raw_text
        item["normalized_text"] = norm
        item["text"] = contextual_canonical_entity_text(
            raw_text,
            label,
            full_text=full_text,
            subtype=subtype,
        )
        item["label"] = label
        item["original_label"] = extracted_label
        item["entity_subtype"] = subtype
        item["label_rescued"] = label_rescued
        item["canonical_key"] = contextual_canonical_entity_key(
            raw_text,
            label,
            full_text=full_text,
            subtype=subtype,
        )
        item["rule_decision"] = rule_verdict
        item["rule_reason"] = rule_reason
        item["features"] = features

        override_applied = False
        if model and meta:
            feat_row = _build_model_features_from_dict(features, meta)
            # CatBoost в production-режиме лучше дергать через predict_proba с 2D входом
            proba = float(model.predict_proba([feat_row])[0][1])
            pred = int(proba >= 0.5)
            item["decision_model_proba"] = proba
            item["decision_model_label"] = pred

            # простая логика совмещения: модель может *повышать* кандидата до promote
            if (
                pred == 1
                and proba >= ML_PROMOTE_OVERRIDE_PROBA_THRESHOLD
                and final_verdict != "promote"
                and rule_reason not in NO_ML_PROMOTE_OVERRIDE_REASONS
                and not features.get("source_cue_hit")
            ):
                final_verdict = "promote"
                final_reason = "ml_promote_override"
                override_applied = True
            elif pred == 0 and final_verdict == "promote":
                # опционально: модель может понижать
                # verdict = "mention"
                # reason = "ml_demote_override"
                pass

        # Extra precision gate: single-token locations are noisy in practice.
        # Keep only strong-confidence candidates unless they are known locations.
        if (
            final_verdict == "promote"
            and label == "location"
            and features.get("token_count") == 1
            and not features.get("known_location_hit")
        ):
            model_proba = item.get("decision_model_proba")
            gliner_score = features.get("gliner_score") or 0.0
            # Keep single-token locations on strong model signal.
            # Hard blocklist remains rejected by rules and won't arrive here.
            if model_proba is not None:
                if model_proba < SINGLE_TOKEN_LOCATION_KEEP_PROBA_THRESHOLD and gliner_score < 0.85:
                    final_verdict = "mention"
                    final_reason = "single_token_location_low_confidence"
            elif gliner_score < 0.85:
                final_verdict = "mention"
                final_reason = "single_token_location_low_confidence"

        item["decision"] = final_verdict
        item["reason"] = final_reason
        item["override_applied"] = override_applied
        item["storyline_anchor_eligible"] = is_storyline_anchor_candidate(
            raw_text=raw_text,
            label=label,
            subtype=subtype,
            decision=final_verdict,
            reason=final_reason,
            features=features,
            full_text=full_text,
        )
        item["rag_anchor_eligible"] = is_rag_evidence_candidate(
            raw_text=raw_text,
            label=label,
            subtype=subtype,
            decision=final_verdict,
            reason=final_reason,
            features=features,
            full_text=full_text,
        )

        logger.info(
            "entity_candidate",
            post_id=post_id,
            raw_text=raw_text,
            normalized_text=norm,
            label=label,
            original_label=extracted_label,
            entity_subtype=subtype,
            label_rescued=label_rescued,
            rule_decision=rule_verdict,
            rule_reason=rule_reason,
            decision=final_verdict,
            reason=final_reason,
            final_decision=final_verdict,
            final_reason=final_reason,
            override_applied=override_applied,
            storyline_anchor_eligible=item.get("storyline_anchor_eligible"),
            rag_anchor_eligible=item.get("rag_anchor_eligible"),
            decision_model_label=item.get("decision_model_label"),
            decision_model_proba=item.get("decision_model_proba"),
            features=features,
        )

        dedup_key = (item["label"], item["text"])

        if final_verdict == "reject":
            if dedup_key not in seen_mentions:
                seen_mentions.add(dedup_key)
                rejected.append(item)
            continue

        if final_verdict == "mention":
            if dedup_key not in seen_mentions:
                seen_mentions.add(dedup_key)
                mentions.append(item)
            continue

        if final_verdict == "promote":
            if dedup_key not in seen_promoted:
                seen_promoted.add(dedup_key)
                promoted.append(item)

    return promoted, mentions, rejected

def dedup_entities(entities: list[dict]) -> list[dict]:
    promoted, _, _ = partition_entities(entities)
    return promoted
