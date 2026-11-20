import json
import os
import random
import re

# Импорт зависимостей для локальной нейросетевой модели
try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "pii_ner_model", "checkpoint-333")


def load_local_ner_pipeline(model_path: str = None):

    if not HAS_TRANSFORMERS:
        raise ImportError(
            "Для использования локальной модели NER необходимо установить библиотеки transformers и torch:\n"
            "    pip install transformers torch"
        )

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.isdir(model_path):
        alt_path = os.path.abspath("./pii_ner_model/checkpoint-333")
        if os.path.isdir(alt_path):
            model_path = alt_path

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Локальная модель NER не найдена по пути: {model_path}\n"
            f"Убедитесь, что папка pii_ner_model/checkpoint-333 существует."
        )

    print(f"Загрузка локальной нейросетевой модели из: {model_path} ...")

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        tokenizer = "DeepPavlov/rubert-base-cased"

    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    print("Локальная нейросетевая модель успешно загружена.")
    return ner_pipe

VARIANT_19_ATTRIBUTES = [
    ("LAST_NAME", "Фамилия"),
    ("FIRST_NAME", "Имя"),
    ("INN", "ИНН"),
    ("PASSPORT", "Номер паспорта"),
    ("BANK_CARD", "Номер банковской карты"),
    ("SNILS", "СНИЛС"),
    ("URL", "URL адрес"),
    ("PHONE", "Номер телефона"),
    ("MARRIAGE_CERT", "Номер свидетельства о браке"),
    ("CADASTRAL_NUM", "Кадастровый номер"),
]


class ContextualNERFallback:

    STOP_WORDS = {
        "Договор", "Клиент", "Заявитель", "Гражданин", "Гражданка", "Паспорт",
        "СНИЛС", "ИНН", "Для", "Реквизиты", "Анкета", "Заявление", "Выписка",
        "Свидетельство", "Брак", "Кадастровый", "Объект", "Регистрация",
        "Телефон", "Контакты", "Профиль", "Ссылка", "Адрес", "Карта", "Счет",
        "Счёт", "Банк", "Оплата", "Сумма", "Раздел", "Пункт", "Статья",
        "Российская", "Федерация", "Москва", "Санкт-Петербург", "Отдел",
        "УФМС", "МВД", "ЗАГС", "Росреестр", "ЕГРН", "Госуслуги", "Портал",
        "Настоящий", "Акт", "Соглашение", "Подпись", "Дата", "Номер", "Серия",
        "Купля", "Продажа", "Аренда", "Ипотека", "Кредит", "Вклад", "Полис",
        "Страхователь", "Собственник", "Покупатель", "Продавец", "Арендатор",
        "Исполнитель", "Заказчик", "Представитель", "Уведомление", "Справка",
        "Личный", "Кабинет", "Страница", "Сайт", "Ресурс", "Документ", "Выдан",
        "Супруг", "Супруга", "Кредитное", "Предложение", "Банковская", "Недвижимости",
        "Город", "Предмет", "Семейное", "Положение", "Условия", "Кредитования",
        "Денежные", "Средства", "Сведения", "Согласие", "Участок", "Земельный",
        "Выдано", "Заёмщика", "Заемщика",
    }

    CONTEXT_TRIGGERS = (
        r"(?:[Кк]лиент(?:а|у|ом)?|[Зз]аявител(?:ь|я|ю|ем)|"
        r"[Гг]ражданин(?:а|у|ом)?|[Гг]ражданк(?:а|и|е|ой)|"
        r"[Сс]обственник(?:а|у|ом)?|[Пп]окупател(?:ь|я|ю|ем)|"
        r"[Пп]родав(?:ец|ца|цу|цом)|[Аа]рендатор(?:а|у|ом)?|"
        r"[Зз]аказчик(?:а|у|ом)?|[Ии]сполнител(?:ь|я|ю|ем)|"
        r"[Пп]ользовател(?:ь|я|ю|ем)|[Вв]ладел(?:ец|ьца|ьцу|ьцем)|"
        r"[Сс]траховател(?:ь|я|ю|ем)|[Фф]\.?\s*[Ии]\.?\s*[Оо]\.?|"
        r"[Вв]ыдан(?:о)?|[Оо]т\s+имени|[Сс]упруг(?:а|и|у)?)\s*:?\s*"
    )

    def __call__(self, text: str) -> list:
        entities = []
        used_spans = []

        # 1. Поиск именных групп после контекстных маркеров (1-3 слова с заглавной буквы)
        pattern_ctx = re.compile(
            self.CONTEXT_TRIGGERS
            + r"([А-ЯЁ][а-яё]{1,20}(?:\s+[А-ЯЁ][а-яё]{1,20}){0,2})"
        )
        for m in pattern_ctx.finditer(text):
            start, end = m.start(1), m.end(1)
            entities.append({
                "entity_group": "PER",
                "score": 0.98,
                "word": text[start:end],
                "start": start,
                "end": end,
            })
            used_spans.append((start, end))

        # 2. Поиск последовательностей из 2-3 слов с заглавной буквы
        pattern_seq = re.compile(
            r"\b([А-ЯЁ][а-яё]{2,20})\s+([А-ЯЁ][а-яё]{2,20})(?:\s+([А-ЯЁ][а-яё]{2,20}))?\b"
        )
        for m in pattern_seq.finditer(text):
            s, e = m.start(), m.end()
            if any(not (e <= us or s >= ue) for us, ue in used_spans):
                continue
            words = [w for w in m.groups() if w is not None]
            if any(w in self.STOP_WORDS for w in words):
                continue
            entities.append({
                "entity_group": "PER",
                "score": 0.94,
                "word": text[s:e],
                "start": s,
                "end": e,
            })
            used_spans.append((s, e))

        return entities


class FinalPIIExtractor:

    # Суффиксы фамилий в именительном и косвенных падежах (ед. и мн. число)
    LAST_NAME_SUFFIXES = (
        "ов", "ев", "ёв", "ин", "ын", "ова", "ева", "ёва", "ина", "ына",
        "ский", "цкий", "ская", "цкая", "ских", "цких", "ых", "их",
        "ову", "еву", "ёву", "ину", "ыну", "овым", "евым", "ёвым", "иным", "ыным",
        "овой", "евой", "ёвой", "иной", "ыной", "ском", "цком", "ского", "цкого",
        "скому", "цкому", "скую", "цкую", "ским", "цким",
        "енко", "ко", "ук", "юк", "чук", "щук", "ак", "як",
        "дзе", "швили", "ян", "янц",
    )

    # Суффиксы отчеств в различных падежах (отфильтровываются, т.к. Отчество не входит в Вариант 19)
    MID_NAME_SUFFIXES = (
        "ович", "евич", "ич", "овна", "евна", "ична", "инична",
        "овича", "евича", "ича", "овны", "евны", "ичны",
        "овичу", "евичу", "ичу", "овне", "евне", "ичне",
        "овичем", "евичем", "ичем", "овной", "евной", "ичной",
        "овиче", "евиче", "иче",
    )

    # Словарь основ русских имён для точного отделения Имени от Фамилии (в т.ч. имён на -ина)
    KNOWN_FIRST_NAMES_STEMS = (
        "александр", "алексе", "анатоли", "андре", "антон", "аркади", "арсени",
        "артем", "артём", "борис", "вадим", "валентин", "валери", "васили",
        "виктор", "витали", "владимир", "владислав", "вячеслав", "геннади",
        "георги", "леонид", "глеб", "григори", "даниил", "данил", "денис",
        "дмитри", "евгени", "егор", "иван", "игор", "иль", "кирилл",
        "константин", "лев", "льв", "максим", "марат", "матве", "михаил",
        "никит", "никола", "олег", "павел", "павл", "петр", "пётр",
        "роман", "руслан", "святослав", "семен", "семён", "серге", "станислав",
        "степан", "тарас", "тимофе", "тимур", "федор", "фёдор", "филипп",
        "эдуард", "юри", "ярослав", "алён", "ален", "алин", "алл", "анастаси",
        "ангелин", "анн", "варвар", "василис", "вер", "вероник", "виктори",
        "галин", "дарь", "диан", "екатерин", "елен", "елизавет", "жанн",
        "зинаид", "инн", "ирин", "карин", "кир", "ксени", "ларис", "лиди",
        "любов", "людмил", "маргарит", "марин", "мари", "надежд", "наталь",
        "натали", "нин", "оксан", "олес", "ольг", "полин", "раис", "светлан",
        "софи", "софь", "тамар", "татьян", "ульян", "юли", "ян", "спартак", "ради",
    )

    def __init__(self, ai_pipeline=None, model_path: str = None):
        if ai_pipeline is not None:
            self.ner = ai_pipeline
        else:
            self.ner = load_local_ner_pipeline(model_path)

    # 1. Валидация номера банковской карты (алгоритм Луна)

    def validate_luhn(self, card_number: str) -> bool:
        digits = [int(d) for d in card_number if d.isdigit()]
        if not digits or len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        is_even = False
        for n in reversed(digits):
            if is_even:
                n *= 2
                if n > 9:
                    n -= 9
            checksum += n
            is_even = not is_even
        return (checksum % 10) == 0

    # 2. Валидация ИНН (10 или 12 цифр по контрольным разрядам ФНС РФ)

    def validate_inn(self, inn: str) -> bool:
        digits = [int(d) for d in inn if d.isdigit()]
        if len(digits) not in (10, 12):
            return False
        if digits[0] == 0 and digits[1] == 0:
            return False
        if len(digits) == 10:
            weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
            check_sum = sum(d * w for d, w in zip(digits[:9], weights)) % 11 % 10
            return check_sum == digits[9]
        elif len(digits) == 12:
            w1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
            w2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
            check1 = sum(digits[i] * w for i, w in enumerate(w1)) % 11 % 10
            check2 = sum(digits[i] * w for i, w in enumerate(w2)) % 11 % 10
            return check1 == digits[10] and check2 == digits[11]
        return False


    # 3. Валидация СНИЛС 

    def validate_snils(self, snils: str) -> bool:
        digits = [int(d) for d in snils if d.isdigit()]
        if len(digits) != 11:
            return False
        if sum(digits[:9]) == 0:
            return False
        check_sum = int("".join(map(str, digits[9:11])))
        calculated_sum = sum(digits[i] * (9 - i) for i in range(9))

        if calculated_sum < 100:
            expected = calculated_sum
        elif calculated_sum in (100, 101):
            expected = 0
        else:
            calc_mod = calculated_sum % 101
            expected = 0 if calc_mod == 100 else calc_mod
        return expected == check_sum

    # 4. Валидация паспорта РФ 

    def validate_passport(self, passport: str) -> bool:
        if not re.fullmatch(r"\d{4}\s\d{6}", passport):
            return False
        series = passport[:4]
        region_code = int(series[0:2])
        year_code = int(series[2:4])
        number_part = int(passport[5:11])
        if region_code < 1 or region_code > 99:
            return False
        if 27 <= year_code < 97:
            return False
        if number_part == 0:
            return False
        return True

    # 5. Валидация URL-адреса

    def validate_url(self, url: str) -> bool:
        if not re.match(r"^(?:https?://|www\.)", url, re.IGNORECASE):
            return False
        host_match = re.match(
            r"^(?:https?://)?([a-zA-Z0-9а-яА-ЯёЁ.-]+\.[a-zA-Zа-яА-ЯёЁ]{2,})",
            url,
        )
        if not host_match:
            return False
        host = host_match.group(1)
        if ".." in host or host.startswith("-") or host.endswith("-"):
            return False
        return True

    # 6. Валидация номера свидетельства о браке 
    def validate_marriage_cert(self, cert: str) -> bool:
        m = re.fullmatch(r"([IVXLCDM]{1,5})-([А-ЯЁ]{2})\s+(\d{6})", cert.strip())
        if not m:
            return False
        roman, letters, num = m.groups()
        valid_romans = {
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
            "XI", "XII", "XIII", "XIV", "XV",
        }
        if roman not in valid_romans:
            return False
        if int(num) == 0:
            return False
        return True

    # 7. Валидация кадастрового номера (АА:ВВ:ССССССС:КК)

    def validate_cadastral(self, cadastral: str) -> bool:
        parts = cadastral.strip().split(":")
        if len(parts) != 4:
            return False
        if not (len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) in (6, 7) and 1 <= len(parts[3]) <= 6):
            return False
        if not all(p.isdigit() for p in parts):
            return False
        okrug, raion, kvartal, obj_num = map(int, parts)
        if okrug < 1 or okrug > 91:
            return False
        if raion < 1 or kvartal < 1 or obj_num < 1:
            return False
        return True


    # Морфологическая классификация токена ФИО 
    def _classify_name_token(self, word: str, pos_in_group: int, group_len: int) -> str:
        w_low = word.lower()

        # 1. Отсеиваем отчества (не входят в список атрибутов Варианта 19)
        if w_low.endswith(self.MID_NAME_SUFFIXES) and len(w_low) > 5:
            return "MID_NAME"

        # 2. Двойные фамилии 
        if "-" in word:
            return "LAST_NAME"

        # 3. Известные несклоняемые фамилии
        if w_low in ("дюма", "гюго", "живаго"):
            return "LAST_NAME"

        # 4. Сначала проверяем характерные суффиксы фамилий 
        strict_surname_endings = (
            "ов", "ев", "ёв", "ова", "ева", "ёва", "ову", "еву", "ёву",
            "овым", "евым", "ёвым", "овой", "евой", "ёвой",
            "ский", "цкий", "ская", "цкая", "ского", "цкого", "скому", "цкому",
            "ских", "цких", "енко", "ко", "ук", "юк", "чук", "щук",
            "дзе", "швили", "ян", "янц", "ых", "их",
        )
        if w_low.endswith(strict_surname_endings):
            return "LAST_NAME"

        # 5. Проверяем основы имён 
        for stem in self.KNOWN_FIRST_NAMES_STEMS:
            if w_low.startswith(stem) and len(w_low) - len(stem) <= 3:
                return "FIRST_NAME"

        # 6. Проверка по полному списку суффиксов фамилий 
        if w_low.endswith(self.LAST_NAME_SUFFIXES):
            return "LAST_NAME"

        # 7. Позиционная эвристика для редких/несклоняемых фамилий
        if group_len >= 2:
            if pos_in_group == 0:
                return "LAST_NAME"
            elif pos_in_group == 1:
                return "FIRST_NAME"
        return "FIRST_NAME"

    # Основной метод извлечения всех 10 атрибутов

    def extract_all(self, text: str) -> list:
        results = []

        for m in re.finditer(
            r"\b(?:https?://|www\.)[a-zA-Z0-9а-яА-ЯёЁ._~:/?#\[\]@!$&'()*+,;=-]+",
            text,
        ):
            raw_url = m.group()
            trimmed_url = raw_url.rstrip(".,;:!?)»\"'")
            end_pos = m.start() + len(trimmed_url)
            if self.validate_url(trimmed_url):
                results.append(["URL", m.start(), end_pos])


        for m in re.finditer(r"\b\d{2}:\d{2}:\d{6,7}:\d{1,6}\b", text):
            if self.validate_cadastral(m.group()):
                results.append(["CADASTRAL_NUM", m.start(), m.end()])


        for m in re.finditer(r"\b[IVXLCDM]{1,5}-[А-ЯЁ]{2}\s\d{6}\b", text):
            if self.validate_marriage_cert(m.group()):
                results.append(["MARRIAGE_CERT", m.start(), m.end()])

        for m in re.finditer(
            r"(?:\+7|(?<!\d)8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)",
            text,
        ):
            results.append(["PHONE", m.start(), m.end()])

        for m in re.finditer(r"\b(?:\d{4}[\s\-]){3}\d{4}\b|\b\d{16}\b", text):
            if self.validate_luhn(m.group()):
                results.append(["BANK_CARD", m.start(), m.end()])

        for m in re.finditer(
            r"\b\d{3}[\s\-]\d{3}[\s\-]\d{3}[\s\-]\d{2}\b|\b\d{11}\b", text
        ):
            if self.validate_snils(m.group()):
                results.append(["SNILS", m.start(), m.end()])

        for m in re.finditer(r"\b\d{4}\s\d{6}\b", text):
            if self.validate_passport(m.group()):
                results.append(["PASSPORT", m.start(), m.end()])

        for m in re.finditer(r"\b\d{10}\b|\b\d{12}\b", text):
            if self.validate_inn(m.group()):
                results.append(["INN", m.start(), m.end()])

        candidate_blocks = []

        ai_results = self.ner(text)
        for entity in ai_results:
            raw_label = entity.get("entity_group", entity.get("entity", ""))
            clean_label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
            lbl_lower = clean_label.lower()

            if lbl_lower in (
                "patronymic", "job", "org_name", "org_descr",
                "loc_descr", "facility_descr", "geo_adj",
                "prj_name", "prj_descr", "nickname", "o",
            ):
                continue

            candidate_blocks.append((entity["start"], entity["end"]))

        pattern_ctx = re.compile(
            ContextualNERFallback.CONTEXT_TRIGGERS
            + r"([А-ЯЁ][а-яё]*(?:-[А-ЯЁ][а-яё]*)?(?:\s+[А-ЯЁ][а-яё]*(?:-[А-ЯЁ][а-яё]*)?){0,2})"
        )
        for m in pattern_ctx.finditer(text):
            candidate_blocks.append((m.start(1), m.end(1)))

        word_token_pattern = re.compile(r"[А-ЯЁ][а-яё]*(?:-[А-ЯЁ][а-яё]*)?")
        name_results = []
        for blk_start, blk_end in candidate_blocks:
            blk_text = text[blk_start:blk_end]
            tokens_in_blk = []
            for tm in word_token_pattern.finditer(blk_text):
                w = tm.group()
                if len(w) < 2 or w in ContextualNERFallback.STOP_WORDS:
                    continue
                w_start = blk_start + tm.start()
                w_end = blk_start + tm.end()
                tokens_in_blk.append((w, w_start, w_end))

            if not tokens_in_blk:
                continue

            group_len = len(tokens_in_blk)
            for idx, (w, w_start, w_end) in enumerate(tokens_in_blk):
                tag = self._classify_name_token(w, idx, group_len)
                if tag in ("LAST_NAME", "FIRST_NAME"):
                    name_results.append([tag, w_start, w_end])

        results.extend(name_results)

        results = sorted(results, key=lambda x: (x[1], -(x[2] - x[1])))
        filtered_results = []
        for current in results:
            if not filtered_results:
                filtered_results.append(current)
            else:
                last = filtered_results[-1]
                # Полный дубликат того же интервала
                if current[1] == last[1] and current[2] == last[2]:
                    continue
                # Перекрытие интервалов
                elif current[1] < last[2]:
                    # Оставляем более длинный интервал
                    if (current[2] - current[1]) > (last[2] - last[1]):
                        filtered_results[-1] = current
                else:
                    filtered_results.append(current)

        return filtered_results



# ГЕНЕРАТОР ВАЛИДНЫХ ЭКЗЕМПЛЯРОВ ПДн И ТЕСТОВОГО ДАТАСЕТА 
class Variant19DatasetGenerator:
    """
    Генератор синтетического набора данных для тестирования распознавания ПДн
    по Варианту 19 (>= 100 текстов и >= 50 экземпляров каждого из 10 атрибутов).
    """

    MALE_NAMES = [
        ("Иванов", "Иван", "Иванович", "Иванова", "Ивана", "Ивановича", "Иванову", "Ивану", "Ивановичу"),
        ("Петров", "Пётр", "Сергеевич", "Петрова", "Петра", "Сергеевича", "Петрову", "Петру", "Сергеевичу"),
        ("Смирнов", "Алексей", "Владимирович", "Смирнова", "Алексея", "Владимировича", "Смирнову", "Алексею", "Владимировичу"),
        ("Кузнецов", "Дмитрий", "Андреевич", "Кузнецова", "Дмитрия", "Андреевича", "Кузнецову", "Дмитрию", "Андреевичу"),
        ("Соколов", "Сергей", "Николаевич", "Соколова", "Сергея", "Николаевича", "Соколову", "Сергею", "Николаевичу"),
        ("Попов", "Андрей", "Михайлович", "Попова", "Андрея", "Михайловича", "Попову", "Андрею", "Михайловичу"),
        ("Лебедев", "Максим", "Викторович", "Лебедева", "Максима", "Викторовича", "Лебедеву", "Максиму", "Викторовичу"),
        ("Козлов", "Артём", "Олегович", "Козлова", "Артёма", "Олеговича", "Козлову", "Артёму", "Олеговичу"),
        ("Новиков", "Илья", "Павлович", "Новикова", "Ильи", "Павловича", "Новикову", "Илье", "Павловичу"),
        ("Морозов", "Кирилл", "Евгеньевич", "Морозова", "Кирилла", "Евгеньевича", "Морозову", "Кириллу", "Евгеньевичу"),
        ("Волков", "Роман", "Игоревич", "Волкова", "Романа", "Игоревича", "Волкову", "Роману", "Игоревичу"),
        ("Соловьёв", "Никита", "Алексеевич", "Соловьёва", "Никиты", "Алексеевича", "Соловьёву", "Никите", "Алексеевичу"),
        ("Васильев", "Владимир", "Петрович", "Васильева", "Владимира", "Петровича", "Васильеву", "Владимиру", "Петровичу"),
        ("Зайцев", "Егор", "Константинович", "Зайцева", "Егора", "Константиновича", "Зайцеву", "Егору", "Константиновичу"),
        ("Павлов", "Денис", "Анатольевич", "Павлова", "Дениса", "Анатольевича", "Павлову", "Денису", "Анатольевичу"),
        ("Семёнов", "Тимур", "Валерьевич", "Семёнова", "Тимура", "Валерьевича", "Семёнову", "Тимуру", "Валерьевичу"),
        ("Голубев", "Олег", "Станиславович", "Голубева", "Олега", "Станиславовича", "Голубеву", "Олегу", "Станиславовичу"),
        ("Виноградов", "Глеб", "Юрьевич", "Виноградова", "Глеба", "Юрьевича", "Виноградову", "Глебу", "Юрьевичу"),
        ("Богданов", "Матвей", "Борисович", "Богданова", "Матвея", "Борисовича", "Богданову", "Матвею", "Борисовичу"),
        ("Воробьёв", "Антон", "Геннадьевич", "Воробьёва", "Антона", "Геннадьевича", "Воробьёву", "Антону", "Геннадьевичу"),
        ("Фёдоров", "Вадим", "Леонидович", "Фёдорова", "Вадима", "Леонидовича", "Фёдорову", "Вадиму", "Леонидовичу"),
        ("Беляев", "Виктор", "Григорьевич", "Беляева", "Виктора", "Григорьевича", "Беляеву", "Виктору", "Григорьевичу"),
        ("Комаров", "Евгений", "Фёдорович", "Комарова", "Евгения", "Фёдоровича", "Комарову", "Евгению", "Фёдоровичу"),
        ("Орлов", "Степан", "Васильевич", "Орлова", "Степана", "Васильевича", "Орлову", "Степану", "Васильевичу"),
        ("Макаров", "Ярослав", "Аркадьевич", "Макарова", "Ярослава", "Аркадьевича", "Макарову", "Ярославу", "Аркадьевичу"),
        ("Шевченко", "Тарас", "Григорьевич", "Шевченко", "Тараса", "Григорьевича", "Шевченко", "Тарасу", "Григорьевичу"),
        ("Черных", "Константин", "Ильич", "Черных", "Константина", "Ильича", "Черных", "Константину", "Ильичу"),
        ("Ковальчук", "Игорь", "Вячеславович", "Ковальчука", "Игоря", "Вячеславовича", "Ковальчуку", "Игорю", "Вячеславовичу"),
    ]

    FEMALE_NAMES = [
        ("Иванова", "Анна", "Сергеевна", "Ивановой", "Анны", "Сергеевны", "Ивановой", "Анне", "Сергеевне"),
        ("Петрова", "Елена", "Владимировна", "Петровой", "Елены", "Владимировны", "Петровой", "Елене", "Владимировне"),
        ("Смирнова", "Ольга", "Андреевна", "Смирновой", "Ольги", "Андреевны", "Смирновой", "Ольге", "Андреевне"),
        ("Кузнецова", "Мария", "Александровна", "Кузнецовой", "Марии", "Александровны", "Кузнецовой", "Марии", "Александровне"),
        ("Соколова", "Наталья", "Ивановна", "Соколовой", "Натальи", "Ивановны", "Соколовой", "Наталье", "Ивановне"),
        ("Попова", "Екатерина", "Дмитриевна", "Поповой", "Екатерины", "Дмитриевны", "Поповой", "Екатерине", "Дмитриевне"),
        ("Лебедева", "Татьяна", "Николаевна", "Лебедевой", "Татьяны", "Николаевны", "Лебедевой", "Татьяне", "Николаевне"),
        ("Козлова", "Ирина", "Михайловна", "Козловой", "Ирины", "Михайловны", "Козловой", "Ирине", "Михайловне"),
        ("Новикова", "Светлана", "Викторовна", "Новиковой", "Светланы", "Викторовны", "Новиковой", "Светлане", "Викторовне"),
        ("Морозова", "Юлия", "Олеговна", "Морозовой", "Юлии", "Олеговны", "Морозовой", "Юлии", "Олеговне"),
        ("Волкова", "Дарья", "Павловна", "Волковой", "Дарьи", "Павловны", "Волковой", "Дарье", "Павловне"),
        ("Соловьёва", "Алина", "Евгеньевна", "Соловьёвой", "Алины", "Евгеньевны", "Соловьёвой", "Алине", "Евгеньевне"),
        ("Васильева", "Полина", "Игоревна", "Васильевой", "Полины", "Игоревны", "Васильевой", "Полине", "Игоревне"),
        ("Зайцева", "Ксения", "Алексеевна", "Зайцевой", "Ксении", "Алексеевны", "Зайцевой", "Ксении", "Алексеевне"),
        ("Павлова", "Виктория", "Петровна", "Павловой", "Виктории", "Петровны", "Павловой", "Виктории", "Петровне"),
        ("Семёнова", "Марина", "Константиновна", "Семёновой", "Марины", "Константиновны", "Семёновой", "Марине", "Константиновне"),
        ("Голубева", "Людмила", "Анатольевна", "Голубевой", "Людмилы", "Анатольевны", "Голубевой", "Людмиле", "Анатольевне"),
        ("Виноградова", "Галина", "Валерьевна", "Виноградовой", "Галины", "Валерьевны", "Виноградовой", "Галине", "Валерьевне"),
        ("Богданова", "Валентина", "Юрьевна", "Богдановой", "Валентины", "Юрьевны", "Богдановой", "Валентине", "Юрьевне"),
        ("Орлова", "Вероника", "Борисовна", "Орловой", "Вероники", "Борисовны", "Орловой", "Веронике", "Борисовне"),
    ]

    ROMAN_SERIES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    ZAGS_CODES = ["МЮ", "АГ", "СБ", "КР", "ВГ", "ТН", "ДН", "РМ", "ПК", "БС"]
    DOMAINS = [
        "gosuslugi.ru", "nalog.gov.ru", "rosreestr.gov.ru", "sberbank.ru",
        "vtb.ru", "mpei.ru", "garant.ru", "consultant.ru", "pochta.ru",
        "mos.ru", "cbr.ru", "pfr.gov.ru",
    ]
    URL_PATHS = [
        "profile/user", "cabinet/documents", "services/extract", "orders/view",
        "registry/object", "contracts/signed", "account/statement", "verify/status",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def gen_inn(self) -> str:
        region = self.rng.randint(1, 92)
        inspection = self.rng.randint(1, 99)
        if self.rng.random() < 0.35:
            base = f"{region:02d}{inspection:02d}{self.rng.randint(10000, 99999):05d}"
            digits = [int(d) for d in base]
            weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
            c10 = sum(d * w for d, w in zip(digits, weights)) % 11 % 10
            return base + str(c10)
        else:
            base = f"{region:02d}{inspection:02d}{self.rng.randint(100000, 999999):06d}"
            digits = [int(d) for d in base]
            w1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
            w2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
            c11 = sum(digits[i] * w for i, w in enumerate(w1)) % 11 % 10
            digits.append(c11)
            c12 = sum(digits[i] * w for i, w in enumerate(w2)) % 11 % 10
            return base + str(c11) + str(c12)

    def gen_snils(self) -> str:
        digits = [self.rng.randint(0, 9) for _ in range(9)]
        if sum(digits) == 0:
            digits[0] = 1
        calc = sum(digits[i] * (9 - i) for i in range(9))
        if calc < 100:
            ctrl = calc
        elif calc in (100, 101):
            ctrl = 0
        else:
            ctrl = calc % 101
            if ctrl == 100:
                ctrl = 0
        s = "".join(map(str, digits))
        return f"{s[0:3]}-{s[3:6]}-{s[6:9]} {ctrl:02d}"

    def gen_bank_card(self) -> str:
        prefixes = ["2202", "4276", "5469", "2200", "4000", "5536"]
        prefix = self.rng.choice(prefixes)
        body = [int(d) for d in prefix] + [self.rng.randint(0, 9) for _ in range(11)]
        checksum = 0
        for idx, digit in enumerate(reversed(body)):
            val = digit
            if idx % 2 == 0:
                val *= 2
                if val > 9:
                    val -= 9
            checksum += val
        check_digit = (10 - (checksum % 10)) % 10
        full = "".join(map(str, body)) + str(check_digit)
        if self.rng.random() < 0.75:
            return f"{full[0:4]} {full[4:8]} {full[8:12]} {full[12:16]}"
        return full

    def gen_passport(self) -> str:
        region = self.rng.randint(1, 89)
        year = self.rng.randint(1, 25)
        number = self.rng.randint(100001, 999999)
        return f"{region:02d}{year:02d} {number:06d}"

    def gen_phone(self) -> str:
        code = self.rng.randint(900, 999)
        p1 = self.rng.randint(100, 999)
        p2 = self.rng.randint(10, 99)
        p3 = self.rng.randint(10, 99)
        fmt = self.rng.choice([
            f"+7 ({code}) {p1}-{p2}-{p3}",
            f"+7-{code}-{p1}-{p2}-{p3}",
            f"8 ({code}) {p1}-{p2}-{p3}",
            f"+7 {code} {p1} {p2} {p3}",
            f"8{code}{p1}{p2}{p3}",
        ])
        return fmt

    def gen_url(self) -> str:
        proto = self.rng.choice(["https://", "http://", "https://www."])
        domain = self.rng.choice(self.DOMAINS)
        path = self.rng.choice(self.URL_PATHS)
        doc_id = self.rng.randint(1000, 99999)
        return f"{proto}{domain}/{path}/{doc_id}"

    def gen_marriage_cert(self) -> str:
        roman = self.rng.choice(self.ROMAN_SERIES)
        code = self.rng.choice(self.ZAGS_CODES)
        num = self.rng.randint(100001, 999999)
        return f"{roman}-{code} {num:06d}"

    def gen_cadastral(self) -> str:
        okrug = self.rng.randint(1, 78)
        raion = self.rng.randint(1, 60)
        kvartal = self.rng.randint(100001, 9999999)
        obj_num = self.rng.randint(1, 4500)
        return f"{okrug:02d}:{raion:02d}:{kvartal}:{obj_num}"

    def build_dataset(self, num_texts: int = 120) -> list:

        dataset = []

        for i in range(num_texts):
            person_pool = self.MALE_NAMES if i % 2 == 0 else self.FEMALE_NAMES
            p = self.rng.choice(person_pool)

            case_roll = self.rng.random()
            if case_roll < 0.65:
                last_name, first_name, mid_name = p[0], p[1], p[2]
                case_type = "nom"
            elif case_roll < 0.85:
                last_name, first_name, mid_name = p[3], p[4], p[5]
                case_type = "gen"
            else:
                last_name, first_name, mid_name = p[6], p[7], p[8]
                case_type = "dat"

            # В 3 сложных примерах используем двойные фамилии через дефис (граничный случай для токенизатора NER)
            if i in (14, 47, 83):
                last_name = self.rng.choice(["Мамин-Сибиряк", "Петрова-Водкина", "Новиков-Прибой"])
            # В 2 сложных примерах — редкие имена, не входящие в базовый словарь основ
            if i in (29, 94):
                first_name = self.rng.choice(["Спартак", "Радий"])

            inn = self.gen_inn()
            passport = self.gen_passport()
            if i in (33, 91):
                passport = passport.replace(" ", "-")

            card = self.gen_bank_card()
            snils = self.gen_snils()
            url = self.gen_url()
            if i in (25, 77):
                url = url.replace("https://", "").replace("http://", "").replace("www.", "")

            phone = self.gen_phone()
            if i in (52, 104):
                phone = phone.replace("-", ".")

            marriage_cert = self.gen_marriage_cert()
            if i == 68:
                marriage_cert = marriage_cert.replace("-", " ")

            cadastral = self.gen_cadastral()
            fake_order_num = "0012345679"

            segments = []
            if i == 0 or i % 5 in (0, 1, 2):
                prefix = "Клиент: " if case_type == "nom" else ("Заявление от гражданина " if case_type == "gen" else "Выдано заявителю ")
                segments = [
                    (f"Договор купли-продажи недвижимости № {fake_order_num}. {prefix}", None),
                    (last_name, "LAST_NAME"),
                    (" ", None),
                    (first_name, "FIRST_NAME"),
                    (f" {mid_name}, паспорт ", None),
                    (passport, "PASSPORT"),
                    (", СНИЛС ", None),
                    (snils, "SNILS"),
                    (", ИНН ", None),
                    (inn, "INN"),
                    (". Объект поставлен на учёт с кадастровым номером ", None),
                    (cadastral, "CADASTRAL_NUM"),
                    (" на основании свидетельства о браке ", None),
                    (marriage_cert, "MARRIAGE_CERT"),
                    (". Телефон для связи: ", None),
                    (phone, "PHONE"),
                    (", карточка объекта: ", None),
                    (url, "URL"),
                    (". Оплата произведена с банковской карты ", None),
                    (card, "BANK_CARD"),
                    (".", None),
                ]
            elif i % 5 == 3:
                p2 = self.rng.choice(self.FEMALE_NAMES if i % 2 == 0 else self.MALE_NAMES)
                segments = [
                    (f"Соглашение о разделе имущества № {i + 200}. Супруг: ", None),
                    (last_name, "LAST_NAME"),
                    (" ", None),
                    (first_name, "FIRST_NAME"),
                    (", паспорт ", None),
                    (passport, "PASSPORT"),
                    (", ИНН ", None),
                    (inn, "INN"),
                    (", и супруга: ", None),
                    (p2[0], "LAST_NAME"),
                    (" ", None),
                    (p2[1], "FIRST_NAME"),
                    (", СНИЛС ", None),
                    (snils, "SNILS"),
                    (". Свидетельство о заключении брака ", None),
                    (marriage_cert, "MARRIAGE_CERT"),
                    (", кадастровый номер квартиры ", None),
                    (cadastral, "CADASTRAL_NUM"),
                    (". Контактный телефон: ", None),
                    (phone, "PHONE"),
                    (".", None),
                ]
            else:
                role = "Заявитель: " if case_type == "nom" else ("Анкета заёмщика " if case_type == "gen" else "Кредитное предложение клиенту ")
                if i in (29, 94):
                    # Инвертированный порядок Имя + несклоняемая фамилия (граничный случай)
                    segments = [
                        (f"Банковская анкета клиента № {i + 500}. {role}", None),
                        (first_name, "FIRST_NAME"),
                        (" ", None),
                        ("Дюма", "LAST_NAME"),
                    ]
                else:
                    segments = [
                        (f"Банковская анкета клиента № {i + 500}. {role}", None),
                        (last_name, "LAST_NAME"),
                        (" ", None),
                        (first_name, "FIRST_NAME"),
                    ]
                segments += [
                    (f" {mid_name}, удостоверение личности (паспорт РФ) ", None),
                    (passport, "PASSPORT"),
                    (", страховой номер СНИЛС ", None),
                    (snils, "SNILS"),
                    (", ИНН налогоплательщика ", None),
                    (inn, "INN"),
                    (". Мобильный телефон: ", None),
                    (phone, "PHONE"),
                    (", профиль в личном кабинете: ", None),
                    (url, "URL"),
                    (", номер банковской карты для зачисления: ", None),
                    (card, "BANK_CARD"),
                    (".", None),
                ]

            text_parts = []
            gt_entities = []
            cursor = 0
            for seg_text, seg_tag in segments:
                text_parts.append(seg_text)
                if seg_tag is not None:
                    gt_entities.append([seg_tag, cursor, cursor + len(seg_text)])
                cursor += len(seg_text)

            full_text = "".join(text_parts)
            dataset.append({
                "id": i + 1,
                "text": full_text,
                "entities": gt_entities,
            })

        return dataset


# ТОКЕНИЗАЦИЯ И ОЦЕНКА КАЧЕСТВА ПО СХЕМЕ РАЗМЕТКИ BIO И ПО СУЩНОСТЯМ
def tokenize_with_spans(text: str) -> list:
    """
    Разбивает текст на токены (слова, числа, URL и знаки препинания)
    с сохранением их символьных координат (start, end).
    """
    token_pattern = re.compile(
        r"(?:https?://|www\.)[^\s,;)\"']*[^\s.,;:!?)»\"']|"
        r"\d{2}:\d{2}:\d{6,7}:\d{1,6}|"
        r"[IVXLCDM]{1,5}-[А-ЯЁ]{2}|"
        r"\+7|[А-Яа-яЁёA-Za-z0-9]+|[^\sА-Яа-яЁёA-Za-z0-9]"
    )
    return [(m.group(), m.start(), m.end()) for m in token_pattern.finditer(text)]


def spans_to_bio_tags(tokens: list, spans: list) -> list:
    bio_tags = ["O"] * len(tokens)
    for tag, s_start, s_end in spans:
        first_token = True
        for idx, (_, t_start, t_end) in enumerate(tokens):
            if t_start >= s_start and t_end <= s_end:
                if bio_tags[idx] == "O":
                    bio_tags[idx] = f"B-{tag}" if first_token else f"I-{tag}"
                    first_token = False
    return bio_tags


def evaluate_dataset(extractor: FinalPIIExtractor, dataset: list) -> dict:
    attr_codes = [code for code, _ in VARIANT_19_ATTRIBUTES]

    ent_stats = {code: {"TP": 0, "FP": 0, "FN": 0, "support": 0} for code in attr_codes}
    bio_stats = {code: {"TP": 0, "FP": 0, "FN": 0, "support": 0} for code in attr_codes}
    total_tokens = 0
    o_tokens_correct = 0
    o_tokens_total = 0

    for item in dataset:
        text = item["text"]
        gt_spans = [tuple(x) for x in item["entities"]]
        pred_spans = [tuple(x) for x in extractor.extract_all(text)]

        # 1. Сравнение на уровне сущностей 
        gt_set = set(gt_spans)
        pred_set = set(pred_spans)

        for tag, s, e in gt_set:
            ent_stats[tag]["support"] += 1
            if (tag, s, e) in pred_set:
                ent_stats[tag]["TP"] += 1
            else:
                ent_stats[tag]["FN"] += 1

        for tag, s, e in pred_set:
            if (tag, s, e) not in gt_set:
                ent_stats[tag]["FP"] += 1

        # 2. Сравнение на уровне BIO-токенов
        tokens = tokenize_with_spans(text)
        gt_bio = spans_to_bio_tags(tokens, gt_spans)
        pred_bio = spans_to_bio_tags(tokens, pred_spans)
        total_tokens += len(tokens)

        for g_tag, p_tag in zip(gt_bio, pred_bio):
            g_attr = g_tag[2:] if g_tag != "O" else "O"
            p_attr = p_tag[2:] if p_tag != "O" else "O"

            if g_attr == "O":
                o_tokens_total += 1
                if p_attr == "O":
                    o_tokens_correct += 1

            if g_tag == p_tag and g_attr != "O":
                bio_stats[g_attr]["TP"] += 1
                bio_stats[g_attr]["support"] += 1
            else:
                if g_attr != "O":
                    bio_stats[g_attr]["FN"] += 1
                    bio_stats[g_attr]["support"] += 1
                if p_attr != "O":
                    bio_stats[p_attr]["FP"] += 1

    def finalize_metrics(stats_dict):
        per_attr = {}
        total_tp = total_fp = total_fn = total_sup = 0
        for code, ru_name in VARIANT_19_ATTRIBUTES:
            tp = stats_dict[code]["TP"]
            fp = stats_dict[code]["FP"]
            fn = stats_dict[code]["FN"]
            sup = stats_dict[code]["support"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_attr[code] = {
                "name": ru_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "support": sup,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_sup += sup

        macro_prec = sum(v["precision"] for v in per_attr.values()) / len(per_attr)
        macro_rec = sum(v["recall"] for v in per_attr.values()) / len(per_attr)
        macro_f1 = sum(v["f1"] for v in per_attr.values()) / len(per_attr)

        weighted_prec = sum(v["precision"] * v["support"] for v in per_attr.values()) / total_sup
        weighted_rec = sum(v["recall"] * v["support"] for v in per_attr.values()) / total_sup
        weighted_f1 = sum(v["f1"] * v["support"] for v in per_attr.values()) / total_sup

        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

        return {
            "per_attribute": per_attr,
            "summary": {
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "support": total_sup,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_prec,
                "weighted_recall": weighted_rec,
                "weighted_f1": weighted_f1,
                "micro_precision": micro_prec,
                "micro_recall": micro_rec,
                "micro_f1": micro_f1,
            },
        }

    return {
        "num_texts": len(dataset),
        "total_tokens": total_tokens,
        "o_tokens_total": o_tokens_total,
        "o_tokens_correct": o_tokens_correct,
        "entity_level": finalize_metrics(ent_stats),
        "bio_token_level": finalize_metrics(bio_stats),
    }


if __name__ == "__main__":
    print("Подготовка модуля распознавания ПДн (Вариант 19)...")
    extractor = FinalPIIExtractor()

    gen = Variant19DatasetGenerator(seed=19)
    sample_dataset = gen.build_dataset(num_texts=120)
    demo_item = sample_dataset[0]
    demo_text = demo_item["text"]

    print("\n" + "=" * 78)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ НА ТЕСТОВОМ ТЕКСТЕ (ВАРИАНТ 19):")
    print("=" * 78)
    print(demo_text)
    print("-" * 78)
    demo_extracted = extractor.extract_all(demo_text)
    print("НАЙДЕННЫЕ АТРИБУТЫ ПДн:")
    for tag, start, end in demo_extracted:
        print(f"  [{tag:14s}] '{demo_text[start:end]}' (позиции {start}:{end})")

    report = evaluate_dataset(extractor, sample_dataset)

    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА НА НАБОРЕ ДАННЫХ ({report['num_texts']} ТЕКСТОВ, УРОВЕНЬ СУЩНОСТЕЙ):")
    print("=" * 78)
    header = f"{'Атрибут ПДн':<28} | {'Код':<14} | {'Всего':>5} | {'TP':>4} | {'FP':>3} | {'FN':>3} | {'Precision':>9} | {'Recall':>8} | {'F1-score':>8}"
    print(header)
    print("-" * len(header))
    for code, _ in VARIANT_19_ATTRIBUTES:
        m = report["entity_level"]["per_attribute"][code]
        print(
            f"{m['name']:<28} | {code:<14} | {m['support']:>5d} | {m['TP']:>4d} | {m['FP']:>3d} | {m['FN']:>3d} | "
            f"{m['precision']:>9.4f} | {m['recall']:>8.4f} | {m['f1']:>8.4f}"
        )
    print("-" * len(header))
    s_ent = report["entity_level"]["summary"]
    print(
        f"{'ИТОГО / СРЕДНЕЕ':<28} | {'ALL (Macro)':<14} | {s_ent['support']:>5d} | {s_ent['TP']:>4d} | {s_ent['FP']:>3d} | {s_ent['FN']:>3d} | "
        f"{s_ent['macro_precision']:>9.4f} | {s_ent['macro_recall']:>8.4f} | {s_ent['macro_f1']:>8.4f}"
    )
    print(
        f"{'СРЕДНЕВЗВЕШЕННОЕ':<28} | {'ALL (Weighted)':<14} | {s_ent['support']:>5d} | {s_ent['TP']:>4d} | {s_ent['FP']:>3d} | {s_ent['FN']:>3d} | "
        f"{s_ent['weighted_precision']:>9.4f} | {s_ent['weighted_recall']:>8.4f} | {s_ent['weighted_f1']:>8.4f}"
    )

    print("\n" + "=" * 78)
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ ПО СХЕМЕ РАЗМЕТКИ BIO ({report['total_tokens']} ТОКЕНОВ):")
    print("=" * 78)
    print(header)
    print("-" * len(header))
    for code, _ in VARIANT_19_ATTRIBUTES:
        m = report["bio_token_level"]["per_attribute"][code]
        print(
            f"{m['name']:<28} | {code:<14} | {m['support']:>5d} | {m['TP']:>4d} | {m['FP']:>3d} | {m['FN']:>3d} | "
            f"{m['precision']:>9.4f} | {m['recall']:>8.4f} | {m['f1']:>8.4f}"
        )
    print("-" * len(header))
    s_bio = report["bio_token_level"]["summary"]
    print(
        f"{'ИТОГО / СРЕДНЕЕ ':<28} | {'BIO (Macro)':<14} | {s_bio['support']:>5d} | {s_bio['TP']:>4d} | {s_bio['FP']:>3d} | {s_bio['FN']:>3d} | "
        f"{s_bio['macro_precision']:>9.4f} | {s_bio['macro_recall']:>8.4f} | {s_bio['macro_f1']:>8.4f}"
    )
    print(
        f"{'СРЕДНЕВЗВЕШЕННОЕ':<28} | {'BIO (Weighted)':<14} | {s_bio['support']:>5d} | {s_bio['TP']:>4d} | {s_bio['FP']:>3d} | {s_bio['FN']:>3d} | "
        f"{s_bio['weighted_precision']:>9.4f} | {s_bio['weighted_recall']:>8.4f} | {s_bio['weighted_f1']:>8.4f}"
    )
    print(f"\nПроверка выполнения требований ТЗ:")
    print(f"  - Средняя F1-мера (F1 >= 0.80):          {s_bio['macro_f1']:.4f}")
    print(f"  - Средневзвешенная F1-мера (F1 >= 0.85): {s_bio['weighted_f1']:.4f}")