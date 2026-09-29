"""Label text check for the «probable» zone of the cascade.

A vision-language model reads the label crop (producer, wine name, grapes, colour, sweetness).
The reading is compared with the structured catalog card of the nearest image candidates:

* conflict — the label clearly says something the candidate is not (another grape, another
  sweetness class, a distinctive wine name the candidate does not have);
* support — the label confirms the candidate (its distinctive name words or its grapes).

Decision (only for the «probable» zone, «found» answers are never touched):
* top-1 has no conflict                        -> keep it;
* top-1 conflicts, a lower candidate is supported and has no conflict -> switch to it;
* every candidate conflicts                    -> «нет в каталоге».

The functions here are pure (no I/O) so they are unit-tested and reused by offline experiments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

CATALOG_JSON = Path(__file__).resolve().parents[3] / "sommelier" / "data" / "catalog.json"

KEEP = "keep"
SWITCH = "switch"
REJECT = "reject"

_LAT2CYR = [
    ("sch", "ш"),
    ("shch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ya", "я"),
    ("yu", "ю"),
    ("yo", "е"),
    ("ye", "е"),
    ("ou", "у"),
    ("oo", "у"),
    ("ee", "и"),
    ("ph", "ф"),
    ("th", "т"),
    ("ck", "к"),
    ("qu", "кв"),
    ("a", "а"),
    ("b", "б"),
    ("c", "к"),
    ("d", "д"),
    ("e", "е"),
    ("f", "ф"),
    ("g", "г"),
    ("h", "х"),
    ("i", "и"),
    ("j", "ж"),
    ("k", "к"),
    ("l", "л"),
    ("m", "м"),
    ("n", "н"),
    ("o", "о"),
    ("p", "п"),
    ("q", "к"),
    ("r", "р"),
    ("s", "с"),
    ("t", "т"),
    ("u", "у"),
    ("v", "в"),
    ("w", "в"),
    ("x", "кс"),
    ("y", "и"),
    ("z", "з"),
]
# Latin wine words as they are spelt in the Russian catalog (letter-by-letter transliteration fails on them)
_LATIN_WORDS = {
    "cabernet": "каберне",
    "sauvignon": "совиньон",
    "pinot": "пино",
    "noir": "нуар",
    "gris": "гри",
    "grigio": "гриджио",
    "blanc": "блан",
    "franc": "фран",
    "chardonnay": "шардоне",
    "riesling": "рислинг",
    "merlot": "мерло",
    "syrah": "сира",
    "shiraz": "шираз",
    "muscat": "мускат",
    "moscato": "мускат",
    "aligote": "алиготе",
    "viognier": "вионье",
    "rkatsiteli": "ркацители",
    "saperavi": "саперави",
    "malbec": "мальбек",
    "grenache": "гренаш",
    "chenin": "шенен",
    "semillon": "семильон",
    "gewurztraminer": "гевюрцтраминер",
    "traminer": "траминер",
    "verdot": "вердо",
    "petit": "пти",
    "tempranillo": "темпранильо",
    "sangiovese": "санджовезе",
    "nebbiolo": "неббиоло",
    "primitivo": "примитиво",
    "zinfandel": "зинфандель",
    "marsanne": "марсан",
    "roussanne": "руссан",
    "kokur": "кокур",
    "krasnostop": "красностоп",
    "rose": "розе",
    "brut": "брют",
    "extra": "экстра",
    "rouge": "руж",
    "sec": "сек",
    "demi": "деми",
    "reserve": "резерв",
    "pet": "пет",
    "nat": "нат",
    "orange": "оранж",
    "white": "белое",
    "red": "красное",
}
_SWEETNESS = {  # label / catalog wording -> class
    "сухое": "dry",
    "брют": "dry",
    "экстра брют": "dry",
    "extra brut": "dry",
    "brut": "dry",
    "dry": "dry",
    "sec": "dry",
    "полусухое": "semidry",
    "demi sec": "semisweet",
    "полусладкое": "semisweet",
    "сладкое": "sweet",
    "десертное": "sweet",
}
_SWEET_ORDER = {"dry": 0, "semidry": 1, "semisweet": 2, "sweet": 3}
_COLOR = {"красное": "red", "белое": "white", "розовое": "rose", "оранжевое": "orange", "розе": "rose", "rose": "rose"}
# Words that never distinguish one wine from another
_GENERIC = {
    "вино",
    "вина",
    "винодельня",
    "винодельни",
    "красное",
    "белое",
    "розовое",
    "оранжевое",
    "сухое",
    "полусухое",
    "полусладкое",
    "сладкое",
    "брют",
    "экстра",
    "игристое",
    "шампанское",
    "российское",
    "россии",
    "крым",
    "крыма",
    "кубань",
    "резерв",
    "reserve",
    "выдержанное",
    "ограниченная",
    "партия",
    "урожая",
    "год",
    "коллекционное",
    "авторское",
    "select",
    "selection",
    "series",
    "estate",
    "winery",
    "wine",
    "wines",
    "vineyards",
    "the",
    "and",
    "de",
    "la",
    "le",
    "by",
    "за",
    "от",
    "из",
    "для",
    "на",
    "и",
    "усадьба",
    "поместье",
    "шато",
    "chateau",
    "домен",
    "domaine",
    "cuvee",
    "кюве",
    "blanc",
    "blend",
    "бленд",
    "купаж",
    "розе",
    "rose",
    "руж",
    "rouge",
    "блан",
    "бьянко",
    "россо",
    "нуво",
}


def _generic(word: str) -> bool:
    return word in _GENERIC_NORM


def norm(text: str | None) -> str:
    """Lower case, ё -> е, Latin transliterated to Cyrillic, punctuation -> spaces."""
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    out = []
    for word in text.split():
        if word in _LATIN_WORDS:
            word = _LATIN_WORDS[word]
        elif re.search(r"[a-z]", word):
            for lat, cyr in _LAT2CYR:
                word = word.replace(lat, cyr)
        out.append(word)
    return " ".join(out)


_GENERIC_NORM: set[str] = set()


def tokens(text: str | None) -> set[str]:
    return {word for word in norm(text).split() if len(word) >= 3 and not word.isdigit()}


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) >= 5 and (a.startswith(b[:5]) and b.startswith(a[:5])):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def _found(word: str, words: set[str]) -> bool:
    return any(_similar(word, other) for other in words)


def sweetness_class(text: str | None) -> str | None:
    """Most specific sweetness class mentioned in the text (полусухое before сухое)."""
    value = norm(text)
    for key in ("экстра брют", "полусухое", "полусладкое", "сладкое", "сухое", "брют", "demi sec", "extra brut", "brut", "dry"):
        if re.search(rf"(^| ){norm(key)}( |$)", value):
            return _SWEETNESS[key]
    return None


@dataclass(frozen=True)
class LabelReading:
    producer: str = ""
    name: str = ""
    grapes: tuple[str, ...] = ()
    color: str = ""
    sweetness: str = ""
    text: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> LabelReading:
        grapes = data.get("grapes") or []
        if isinstance(grapes, str):
            grapes = [grapes]
        return cls(
            producer=str(data.get("producer") or ""),
            name=str(data.get("name") or ""),
            grapes=tuple(str(g) for g in grapes if g),
            color=str(data.get("color") or ""),
            sweetness=str(data.get("sweetness") or ""),
            text=str(data.get("text") or ""),
        )

    @property
    def all_words(self) -> set[str]:
        return tokens(" ".join([self.producer, self.name, " ".join(self.grapes), self.text]))


@dataclass(frozen=True)
class CatalogCard:
    slug: str
    name: str
    winery: str
    grapes: tuple[str, ...]
    category: str  # Красное / Белое / Розовое / Оранжевое
    type: str  # e.g. «розовое игристое полусухое»

    @classmethod
    def from_catalog(cls, item: dict) -> CatalogCard:
        return cls(
            slug=item["id"],
            name=item.get("name") or "",
            winery=item.get("winery") or "",
            grapes=tuple(item.get("grapes") or []),
            category=item.get("category") or "",
            type=item.get("type") or "",
        )


@dataclass
class CandidateCheck:
    slug: str
    conflicts: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    unique: set[str] = field(default_factory=set)
    unique_hit: set[str] = field(default_factory=set)


@dataclass
class LabelDecision:
    action: str  # keep | switch | reject
    slug: str | None
    reason: str
    checks: list[CandidateCheck]


def name_words(card: CatalogCard, label: LabelReading | None = None) -> set[str]:
    """Words of the catalog name that can tell wines apart: not generic and not the producer."""
    producer = tokens(card.winery) | (tokens(label.producer) if label else set())
    return {w for w in tokens(card.name) if not _generic(w) and not _found(w, producer)}


def check_candidate(label: LabelReading, card: CatalogCard, others: list[CatalogCard] = ()) -> CandidateCheck:
    """conflicts — hard contradictions (grape, sweetness); support — words of this card found on the label.

    `unique` are the name words of this card that the other candidates do not have (e.g. «рубин» vs
    «пино нуар» inside the «Красная стрелка» series); only they can decide between close candidates.
    """
    result = CandidateCheck(slug=card.slug)
    label_words = label.all_words
    grape_words = {w for g in card.grapes for w in tokens(g)}

    words = name_words(card, label)
    other_words = {w for other in others for w in name_words(other, label)}
    result.unique = {w for w in words if not _found(w, other_words)}
    result.unique_hit = {w for w in result.unique if _found(w, label_words)}
    matched = {w for w in words if _found(w, label_words)}
    if matched:
        result.support.append(f"название: {', '.join(sorted(matched))}")

    # Grapes written on the label must overlap with the card
    label_grapes = {w for g in label.grapes for w in tokens(g) if not _generic(w)}
    if label_grapes and grape_words:
        if any(_found(w, grape_words) for w in label_grapes):
            result.support.append("сорт совпадает")
        elif not any(_found(w, label_words) for w in grape_words):  # the grape field can be misread, the text too
            result.conflicts.append(f"сорт на этикетке ({', '.join(label.grapes)}) ≠ {', '.join(card.grapes)}")

    # Sweetness: only a clear contradiction counts (сухое vs полусладкое). Neighbouring classes are
    # not a conflict: the model often drops «полу-» and reads «полусухое» as «сухое».
    label_sweet = sweetness_class(label.sweetness)
    card_sweet = sweetness_class(f"{card.type} {card.name}")
    if label_sweet and card_sweet and abs(_SWEET_ORDER[label_sweet] - _SWEET_ORDER[card_sweet]) >= 2:
        result.conflicts.append(f"сладость: этикетка «{label.sweetness}», в каталоге «{card.type}»")
    return result


def decide(label: LabelReading, cards: list[CatalogCard], top_k: int = 5) -> LabelDecision:
    """cards are ordered by the image ranking; cards[0] is the current «probable» answer.

    1. top-1 contradicts the label (grape / sweetness)            -> switch to a confirmed candidate or reject;
    2. the label has the unique name words of a lower candidate
       and not those of top-1                                      -> switch to it;
    3. the model read a wine name on the label, it is not the name of
       any candidate and top-1 is not confirmed by anything        -> reject («нет в каталоге»);
    4. otherwise                                                   -> keep top-1.
    """
    cards = cards[:top_k]
    checks = [check_candidate(label, card, [other for other in cards if other is not card]) for card in cards]
    if not checks:
        return LabelDecision(KEEP, None, "нет кандидатов", checks)
    top = checks[0]
    label_producer = tokens(label.producer) | label.all_words

    def same_producer(card: CatalogCard) -> bool:
        # switching to another winery needs its name on the label; within the top-1 winery it is implied
        return card.winery == cards[0].winery or any(_found(w, label_producer) for w in tokens(card.winery) if not _generic(w))

    confirmed = [c for c, card in zip(checks[1:], cards[1:], strict=False) if not c.conflicts and c.unique_hit and same_producer(card)]

    if top.conflicts:
        if confirmed:
            return LabelDecision(SWITCH, confirmed[0].slug, f"{top.conflicts[0]}; на этикетке «{', '.join(sorted(confirmed[0].unique_hit))}» — {confirmed[0].slug}", checks)
        return LabelDecision(REJECT, None, top.conflicts[0], checks)
    if top.unique and not top.unique_hit and confirmed:
        return LabelDecision(SWITCH, confirmed[0].slug, f"на этикетке «{', '.join(sorted(confirmed[0].unique_hit))}», а не «{', '.join(sorted(top.unique))}»", checks)
    label_name = {w for w in tokens(label.name) if not _generic(w) and not _found(w, tokens(label.producer))}
    candidate_words = {w for card in cards for w in name_words(card, label) | {x for g in card.grapes for x in tokens(g)}}
    if label_name and not top.support and top.unique and not any(_found(w, candidate_words) for w in label_name):
        return LabelDecision(REJECT, None, f"на этикетке «{label.name}» — такого названия нет у {len(checks)} ближайших вин (у лучшего — «{cards[0].name}»)", checks)
    why = "; ".join(top.support) or "противоречий нет"
    return LabelDecision(KEEP, top.slug, f"этикетка не противоречит: {why}", checks)


_GENERIC_NORM.update(norm(word) for word in _GENERIC)


def load_catalog_cards(path: Path = CATALOG_JSON) -> dict[str, CatalogCard]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: CatalogCard.from_catalog(item) for item in items if item.get("id")}
