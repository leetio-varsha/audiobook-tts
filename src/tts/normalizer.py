"""Russian text normalizer for TTS — converts numbers, dates, abbreviations,
special symbols, and other non-speakable text into spoken form.

Runs BEFORE ruaccent (stress marks) and Silero TTS.
"""

from __future__ import annotations

import re
import logging

from num2words import num2words

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-е (твёрдое е) in loanwords — consonant stays hard, е sounds as э.
# Key: word with е → value: word with э replacing the hard-е position.
# Only unambiguous, common cases included.
# ---------------------------------------------------------------------------
# Stem-based: key is the invariant stem (longest prefix shared across all
# inflected forms), value is the same stem with е→э.  The regex matches
# stem + any Russian-letter suffix, so "тенденци" covers тенденция/тенденции/тенденций.
# Sorted longest-first at lookup time to avoid partial clobber.
HARD_E_STEMS: dict[str, str] = {
    # Технологии / IT
    "интернет": "интэрнэт",
    "компьютер": "компьютэр",
    "принтер": "принтэр",
    "сервер": "сэрвер",
    "сервис": "сэрвис",
    "модем": "модэм",
    "лазер": "лазэр",
    "интерфейс": "интэрфейс",
    # Бизнес / экономика
    "бизнесмен": "бизнэсмэн",
    "бизнес": "бизнэс",
    "бренд": "брэнд",
    "тренд": "трэнд",
    "менеджмент": "мэнэджмент",
    "менеджер": "мэнэджер",
    "маркетинг": "маркэтинг",
    "продюсер": "продюсэр",
    # Наука / техника
    "тестирован": "тэстирован",
    "тест": "тэст",
    "проект": "проэкт",
    "эффект": "эффэкт",
    "дефект": "дэфэкт",
    "объект": "объэкт",
    "субъект": "субъэкт",
    "термин": "тэрмин",
    "энерги": "энэрги",
    "экстерн": "экстэрн",
    "патент": "патэнт",
    # Еда / напитки (indeclinable — match full word)
    "кафе": "кафэ",
    "пюре": "пюрэ",
    "резюме": "рэзюмэ",
    "карате": "каратэ",
    "канапе": "канапэ",
    # Одежда / мода
    "модел": "модэл",
    "кашне": "кашнэ",
    "декольте": "дэкольтэ",
    "свитер": "свитэр",
    # Культура / искусство
    "шедевр": "шэдэвр",
    "темп": "тэмп",
    "тембр": "тэмбр",
    "теннис": "тэннис",
    "стенд": "стэнд",
    "гротеск": "гротэск",
    "бестселлер": "бэстсэллер",
    # Общие слова
    "депресси": "дэпрэсси",
    "агресси": "агрэсси",
    "сенсаци": "сэнсаци",
    "тенденци": "тэндэнци",
    "стресс": "стрэсс",
    "экспресс": "экспрэсс",
    "прогресс": "прогрэсс",
    "конгресс": "конгрэсс",
    "компресс": "компрэсс",
    "протест": "протэст",
    "контекст": "контэкст",
    "текст": "тэкст",
    "секс": "сэкс",
    "индекс": "индэкс",
    "кодекс": "кодэкс",
    "комплекс": "комплэкс",
    "рейтинг": "рэйтинг",
    "дизайнер": "дизайнэр",
    "детектив": "дэтэктив",
    "интеллект": "интэллэкт",
    "реле": "рэлэ",
    "антенн": "антэнн",
}

# ---------------------------------------------------------------------------
# Abbreviation dictionary
# ---------------------------------------------------------------------------
ABBREVIATIONS: dict[str, str] = {
    # Common Russian abbreviations
    "т.е.": "то есть",
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "т.к.": "так как",
    "т.н.": "так называемый",
    "т.о.": "таким образом",
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "и пр.": "и прочее",
    "и др.": "и другие",
    "напр.": "например",
    "ок.": "около",
    "прим.": "примечание",
    "гг.": "годов",
    "г.": "года",
    "вв.": "веков",
    "в.": "века",
    "н.э.": "нашей эры",
    "до н.э.": "до нашей эры",
    "руб.": "рублей",
    "коп.": "копеек",
    "тыс.": "тысяч",
    "млн.": "миллионов",
    "млн": "миллионов",
    "млрд.": "миллиардов",
    "млрд": "миллиардов",
    "им.": "имени",
    "ул.": "улица",
    "пр.": "проспект",
    "д.": "дом",
    "кв.": "квартира",
    "стр.": "страница",
    "гл.": "глава",
    "рис.": "рисунок",
    "табл.": "таблица",
    "см.": "смотри",
    "ср.": "сравни",
}

# Letter-by-letter abbreviations (uppercase acronyms)
ACRONYMS: dict[str, str] = {
    "СССР": "эс-эс-эс-эр",
    "США": "сэ-шэ-а",
    "ООН": "о-о-эн",
    "ЕС": "е-эс",
    "РФ": "эр-эф",
    "РСФСР": "эр-эс-эф-эс-эр",
    "КГБ": "ка-гэ-бэ",
    "ФСБ": "эф-эс-бэ",
    "МВД": "эм-вэ-дэ",
    "ВМФ": "вэ-эм-эф",
    "ДНК": "дэ-эн-ка",
    "НИИ": "эн-и-и",
    "ЭВМ": "э-вэ-эм",
    "КПСС": "ка-пэ-эс-эс",
    "ВКП": "вэ-ка-пэ",
    "НКВД": "эн-ка-вэ-дэ",
    "ГУМ": "гум",
    "ЦУМ": "цум",
    "ТАСС": "тасс",
    "МИД": "мид",
    "ВУЗ": "вуз",
    "ЗАГС": "загс",
}

# Special symbols
SYMBOLS: dict[str, str] = {
    "%": " процентов",
    "‰": " промилле",
    "№": "номер ",
    "$": " долларов",
    "€": " евро",
    "£": " фунтов",
    "¥": " йен",
    "₽": " рублей",
    "°C": " градусов Цельсия",
    "°F": " градусов Фаренгейта",
    "°": " градусов",
    "²": " в квадрате",
    "³": " в кубе",
    "+": " плюс ",
    "=": " равно ",
    "~": " примерно ",
}

# Units
UNITS: dict[str, str] = {
    "км/ч": "километров в час",
    "м/с": "метров в секунду",
    "кг": "килограмм",
    "г": "грамм",
    "мг": "миллиграмм",
    "км": "километров",
    "м": "метров",
    "см": "сантиметров",
    "мм": "миллиметров",
    "л": "литров",
    "мл": "миллилитров",
    "га": "гектаров",
}

# Roman numeral mapping
_ROMAN_VALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _roman_to_int(s: str) -> int:
    """Convert Roman numeral string to integer."""
    result = 0
    i = 0
    for val, numeral in _ROMAN_VALS:
        while i < len(s) and s[i:].startswith(numeral):
            result += val
            i += len(numeral)
    return result


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class RussianTextNormalizer:
    """Normalize Russian text for TTS consumption."""

    def normalize(self, text: str) -> str:
        """Apply all normalization rules in order."""
        text = self._remove_footnotes(text)
        text = self._expand_acronyms(text)
        text = self._expand_initials(text)
        text = self._expand_dates(text)
        text = self._expand_roman_numerals(text)
        text = self._expand_numbers_with_units(text)  # Before abbreviations (prevents кг → кгода)
        text = self._expand_abbreviations(text)       # After units
        text = self._expand_numbers(text)
        text = self._expand_symbols(text)
        text = self._normalize_punctuation(text)
        text = self._clean_quotes_brackets(text)
        text = self._fix_hard_e(text)
        text = self._collapse_whitespace(text)
        return text.strip()

    # ------------------------------------------------------------------
    # Footnotes / references
    # ------------------------------------------------------------------

    def _remove_footnotes(self, text: str) -> str:
        """Remove [1], [прим. ред.], etc."""
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[прим\.?\s*ред\.?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[сноска\]", "", text, flags=re.IGNORECASE)
        return text

    # ------------------------------------------------------------------
    # Abbreviations
    # ------------------------------------------------------------------

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        # Sort by length (longest first) to avoid partial matches
        for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
            # Single-letter abbreviations like "г." and "в." need a word boundary
            # or whitespace before them to avoid matching inside words
            # (e.g. "деревьев." should NOT match "в." → "века")
            escaped = re.escape(abbr)
            if len(abbr) <= 2:
                pattern = r"(?<!\S)" + escaped + r"(?!\S)"
            else:
                pattern = escaped
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    def _expand_acronyms(self, text: str) -> str:
        """Expand known uppercase acronyms."""
        for acronym, spoken in ACRONYMS.items():
            text = re.sub(r"\b" + re.escape(acronym) + r"\b", spoken, text)
        return text

    def _expand_initials(self, text: str) -> str:
        """А.С. Пушкин → А. С. Пушкин (add spaces for natural pauses)."""
        # Match patterns like А.С. or А. С.
        text = re.sub(
            r"([А-ЯЁ])\.([А-ЯЁ])\.",
            r"\1. \2.",
            text,
        )
        return text

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def _expand_dates(self, text: str) -> str:
        """12.06.1941 → двенадцатого июня тысяча девятьсот сорок первого года."""
        months_genitive = {
            1: "января", 2: "февраля", 3: "марта", 4: "апреля",
            5: "мая", 6: "июня", 7: "июля", 8: "августа",
            9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
        }

        def _date_replace(m: re.Match) -> str:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (1 <= month <= 12 and 1 <= day <= 31):
                return m.group(0)
            try:
                day_word = num2words(day, to="ordinal", lang="ru")
                # Convert to genitive case (approximate: replace last vowel pattern)
                # num2words gives nominative, we need genitive for dates
                month_word = months_genitive.get(month, str(month))
                year_word = num2words(year, to="ordinal", lang="ru")
                return f"{day_word} {month_word} {year_word} года"
            except Exception:
                return m.group(0)

        # DD.MM.YYYY (optionally followed by "года"/"г." which we consume)
        text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\s*(?:года?\.?)?\b", _date_replace, text)

        # Standalone years: 1917 г. → тысяча девятьсот семнадцатый год
        def _year_replace(m: re.Match) -> str:
            year = int(m.group(1))
            suffix = m.group(2)  # "год", "года", "году" etc.
            if 1000 <= year <= 2100:
                try:
                    return num2words(year, to="ordinal", lang="ru") + " " + suffix
                except Exception:
                    pass
            return m.group(0)

        text = re.sub(r"\b(\d{4})\s*(год[ауе]?)\b", _year_replace, text)

        # Year ranges: 1799–1837 → с ... по ...
        def _year_range_replace(m: re.Match) -> str:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 1000 <= y1 <= 2100 and 1000 <= y2 <= 2100:
                try:
                    w1 = num2words(y1, to="ordinal", lang="ru")
                    w2 = num2words(y2, to="ordinal", lang="ru")
                    return f"с {w1} по {w2} год"
                except Exception:
                    pass
            return m.group(0)

        text = re.sub(r"\((\d{4})\s*[–—-]\s*(\d{4})\)", _year_range_replace, text)

        return text

    # ------------------------------------------------------------------
    # Roman numerals
    # ------------------------------------------------------------------

    def _expand_roman_numerals(self, text: str) -> str:
        """XIX век → девятнадцатый век, Глава XII → глава двенадцатая."""
        def _roman_replace(m: re.Match) -> str:
            roman = m.group(1)
            suffix = m.group(2).strip().lower() if m.group(2) else ""
            try:
                n = _roman_to_int(roman)
                if n == 0 or n > 3999:
                    return m.group(0)
                word = num2words(n, to="ordinal", lang="ru")
                return f"{word} {suffix}".strip()
            except Exception:
                return m.group(0)

        # Pattern 1: "XIX век" — numeral before context word
        text = re.sub(
            r"\b([IVXLCDM]{2,})\s*(век[аеу]?|глав[аеуы]?|том[аеу]?|част[ьию]|книг[аеу]?|акт[аеу]?)\b",
            _roman_replace,
            text,
            flags=re.IGNORECASE,
        )

        # Pattern 2: "Глава XII" — context word before numeral
        def _roman_after_word(m: re.Match) -> str:
            word = m.group(1)
            roman = m.group(2)
            try:
                n = _roman_to_int(roman)
                if n == 0 or n > 3999:
                    return m.group(0)
                num_word = num2words(n, to="ordinal", lang="ru")
                return f"{word} {num_word}"
            except Exception:
                return m.group(0)

        text = re.sub(
            r"\b(глав[аеуы]?|том|част[ьию]|книг[аеу]?|акт|раздел)\s+([IVXLCDM]{2,})\b",
            _roman_after_word,
            text,
            flags=re.IGNORECASE,
        )

        return text

    # ------------------------------------------------------------------
    # Numbers with units
    # ------------------------------------------------------------------

    def _expand_numbers_with_units(self, text: str) -> str:
        """42 км → сорок два километров."""
        # Sort units by length (longest first) to match "км/ч" before "км" before "м"
        for unit, spoken in sorted(UNITS.items(), key=lambda x: -len(x[0])):
            # Require the unit to NOT be followed by a letter (prevent "кг" matching in "кгода")
            pattern = r"(\d+(?:[.,]\d+)?)\s*" + re.escape(unit) + r"(?![а-яёА-ЯЁa-zA-Z])"
            def _repl(m: re.Match, spoken=spoken) -> str:
                try:
                    n = m.group(1).replace(",", ".")
                    word = num2words(float(n) if "." in n else int(n), lang="ru")
                    return f"{word} {spoken}"
                except Exception:
                    return m.group(0)
            text = re.sub(pattern, _repl, text)
        return text

    # ------------------------------------------------------------------
    # Numbers
    # ------------------------------------------------------------------

    def _expand_numbers(self, text: str) -> str:
        """Convert remaining numbers to words."""
        def _num_replace(m: re.Match) -> str:
            raw = m.group(0)
            try:
                # Handle decimal numbers
                if "," in raw or "." in raw:
                    n = float(raw.replace(",", "."))
                    return num2words(n, lang="ru")
                else:
                    n = int(raw)
                    # Don't expand very large numbers that are likely IDs/codes
                    if abs(n) > 999999999:
                        return raw
                    return num2words(n, lang="ru")
            except (ValueError, OverflowError):
                return raw

        # Match numbers (with optional decimal), but not part of already-processed text
        text = re.sub(r"\b\d+(?:[.,]\d+)?\b", _num_replace, text)
        return text

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    def _expand_symbols(self, text: str) -> str:
        """Replace special symbols with spoken equivalents."""
        # Handle "число%" pattern (no space before %)
        text = re.sub(r"(\S)%", r"\1 процентов", text)

        for sym, spoken in SYMBOLS.items():
            if sym != "%":  # Already handled above
                text = text.replace(sym, spoken)
        return text

    # ------------------------------------------------------------------
    # Punctuation normalization
    # ------------------------------------------------------------------

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for better TTS prosody."""
        # Ellipsis → period + pause hint
        text = re.sub(r"\.{3,}", "...", text)
        # Multiple exclamation/question marks → single
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        # Em dash in dialogue → comma (natural pause)
        text = re.sub(r"\s*[—–]\s*", ", ", text)
        return text

    # ------------------------------------------------------------------
    # Quotes and brackets
    # ------------------------------------------------------------------

    def _clean_quotes_brackets(self, text: str) -> str:
        """Remove decorative quotes and brackets."""
        # Remove «», "", ''
        text = re.sub(r"[«»„""\"\']", "", text)
        # Remove parenthetical content that's just references
        text = re.sub(r"\(\s*\d+\s*\)", "", text)
        return text

    # ------------------------------------------------------------------
    # Hard-е (твёрдое е) in loanwords
    # ------------------------------------------------------------------

    def _fix_hard_e(self, text: str) -> str:
        """Replace е→э in loanwords where consonant stays hard.

        Uses stem-based matching to handle all inflected forms
        (компьютер → компьютера/компьютеров, тенденция → тенденции/тенденций).
        """
        for stem_e, stem_э in sorted(HARD_E_STEMS.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(
                r"\b" + re.escape(stem_e) + r"([а-яё]*)\b",
                re.IGNORECASE,
            )
            def _repl(m: re.Match, dst=stem_э) -> str:
                suffix = m.group(1)
                if m.group(0)[0].isupper():
                    return dst[0].upper() + dst[1:] + suffix
                return dst + suffix
            text = pattern.sub(_repl, text)
        return text

    # ------------------------------------------------------------------
    # Whitespace
    # ------------------------------------------------------------------

    def _collapse_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/newlines and normalize special whitespace."""
        text = text.replace("\xa0", " ")  # non-breaking space → regular space
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
