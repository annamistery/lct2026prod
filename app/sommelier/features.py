"""Извлечение структурированных признаков вина ТОЛЬКО из полей базы (название, slug, описание, категория).

Каждый признак либо найден в тексте (и тогда хранится доказательство-цитата), либо None («в базе не указано»).
Ничего не «додумывается» из общих знаний о сорте или регионе.
"""
import re


def norm(s):
    return (s or "").lower().replace("ё", "е")


# ---- сахар / тип -------------------------------------------------------------------------
# шкала сладости: 0 сухое/брют, 1 полусухое, 2 полусладкое, 3 сладкое
SWEET_LABEL = {0: "сухое", 1: "полусухое", 2: "полусладкое", 3: "сладкое"}

_SWEET_RULES = [  # (regex, значение) — порядок важен: сначала более специфичные
    (r"brut[\s-]*nature|брют[\s-]*натур|экстра[\s-]*брют|extra[\s-]*brut|zero", 0),
    (r"экстра[\s-]*драй|extra[\s-]*dry|extra[\s-]*sec", 1),
    (r"полусух|semi[\s-]*dry|demi[\s-]*dry|off[\s-]*dry", 1),
    (r"полусладк|semi[\s-]*sweet|demi[\s-]*sec|демисек", 2),
    (r"(?<!полу)(?<!не )сладк(ое|ий|ая)\b(?!\s*(послевкус|нот|тон|аромат|ягод|фрукт))|\bsweet\b|десертн|ликерн", 3),
    (r"брют|brut|\bсухое\b|\bсухой\b|\bdry\b|\bsec\b", 0),
]
_SWEET_SLUG = {"polusuhoe": 1, "polusladkoe": 2, "sladkoe": 3, "suhoe": 0, "brut": 0, "dry": 0, "sweet": 3, "semi": 2}


def sweetness(name, slug, desc):
    n = norm(name)
    for rx, v in _SWEET_RULES:
        m = re.search(rx, n)
        if m:
            return v, f"название: «{m.group(0)}»"
    toks = norm(slug).split("-")
    for t, v in _SWEET_SLUG.items():
        if t in toks:
            return v, f"slug: «{t}»"
    d = norm(desc)
    m = re.search(r"(полусухое|полусладкое|сухое|сладкое|брют\w*)\s*(белое|красное|розовое|оранжевое|игристое)?\s*(вино|игристое)", d) \
        or re.search(r"(вино|игристое)\s*[-–,]?\s*(полусухое|полусладкое|сухое|сладкое)", d)
    if m:
        w = m.group(0)
        for rx, v in _SWEET_RULES:
            if re.search(rx, w):
                return v, f"описание: «{w}»"
    return None, None


def is_sparkling(name, slug, desc):
    t = norm(name) + " " + norm(slug).replace("-", " ") + " " + norm(desc)
    m = re.search(r"игрист|брют|\bbrut\b|шампанск|просекко|креман|cremant|пет[\s-]?нат|pet[\s-]?nat|жемчуж|пузырьк|перляж|перлаж", t)
    return (True, m.group(0)) if m else (False, None)


# ---- шкалы тела / кислотности / танинов ----------------------------------------------------
def _window_levels(text, stem, low, mid, high, win=28):
    """Ищем прилагательные рядом с основой (танин/кислотн). Возвращает (уровень 1..3.5, цитата) или (None, None)."""
    hits = []
    for m in re.finditer(stem, text):
        seg = text[max(0, m.start() - win): m.end() + win]
        seg = re.sub(r"^\S*\s", "", seg) if m.start() - win > 0 else seg      # не режем слова по краям цитаты
        seg = re.sub(r"\s\S*$", "", seg) if m.end() + win < len(text) else seg
        lv = None
        if re.search(high, seg):
            lv = 3.5
        elif re.search(low, seg):
            lv = 1.0
        elif re.search(mid, seg):
            lv = 2.0
        hits.append((lv if lv is not None else 2.0, seg.strip(), lv is not None))
    if not hits:
        return None, None
    hits.sort(key=lambda h: (not h[2]))  # квалифицированные упоминания приоритетнее
    return hits[0][0], hits[0][1]


TANNIN_LOW = r"мягк|нежн|шелков|бархат|тонк|округл|легк|деликат|незаметн|слаб|шлифован"
TANNIN_MID = r"умерен|средн|сбалансир|гармонич|аккуратн"
TANNIN_HIGH = r"выражен|плотн|мощн|структур|терпк|крепк|сильн|высок|насыщ|ощутим|заметн|грубоват|вяжущ"
ACID_LOW = r"мягк|нежн|низк|неярк|приглушен|округл|деликат"
ACID_MID = r"умерен|средн|сбалансир|гармонич|достаточн|приятн"
ACID_HIGH = r"высок|ярк|выражен|свеж|живая|живой|острая|острый|хруст|сочн|освежающ|пронзительн|энергичн"


def tannin(d):
    return _window_levels(d, r"танин", TANNIN_LOW, TANNIN_MID, TANNIN_HIGH)


def acidity(d):
    lv, q = _window_levels(d, r"кислотн", ACID_LOW, ACID_MID, ACID_HIGH)
    if lv is None and re.search(r"кислинк", d):
        m = re.search(r".{0,20}кислинк.{0,20}", d)
        return 1.5, m.group(0).strip()
    return lv, q


_LIGHT_STRONG = r"легк\w*|питк\w*|невесом\w*|воздушн\w*"
_LIGHT_WEAK = r"изящн\w*|деликатн\w*|освежающ\w*|нежн\w*"
_FULL_STRONG = r"плотн\w*|мощн\w*|густ\w*|массивн\w*|концентрирован\w*|полнотел\w*|экстрактивн\w*|полн(?:ый|ое|ым|ая)\s+(?:вкус|тел)\w*"
_FULL_WEAK = r"насыщенн\w*|объемн\w*|богат\w*|структурн\w*|интенсивн\w*|терпк\w*|глубок\w*"
_MID = r"средн\w*\s*(?:тел|плотн|насыщ|структур)\w*|умеренн\w*\s*(?:тел|плотн|насыщ)\w*|среднетел\w*"


def body(d, tan=None, has_oak=False, months=None):
    """Тело вина (шкала 0.5–4) как сумма вкладов слов описания и косвенных признаков (танин, выдержка в дубе).
    Нет ни одного признака -> None. Вклады перечислены в доказательстве."""
    score, why = 2.5, []
    def hit(rx, delta, cap):
        nonlocal score
        ms = re.findall(rx, d)
        if ms:
            got = re.search(rx, d).group(0)
            score += max(-cap, min(cap, delta * len(ms))) if delta < 0 else min(cap, delta * len(ms))
            why.append(got)
    hit(_LIGHT_STRONG, -0.8, 1.6)
    hit(_LIGHT_WEAK, -0.3, 0.6)
    hit(_FULL_STRONG, 1.0, 1.6)
    hit(_FULL_WEAK, 0.4, 0.8)
    m = re.search(_MID, d)
    if m:
        score = (score + 2.5) / 2
        why.append(m.group(0))
    if tan is not None:
        score += 0.6 if tan >= 3 else (-0.2 if tan <= 1 else 0.1)
        why.append(f"танин={tan}")
    if has_oak and months and months >= 6:
        score += 0.4
        why.append(f"выдержка {months} мес.")
    if not why:
        return None, None
    return max(0.5, min(4.0, round(score, 2))), "; ".join(why)


# ---- дуб, выдержка ----------------------------------------------------------------------
def oak(d):
    m = re.search(r"дуб|бочк|баррик|ванил|тост|кокос|обжиг|кедр", d)
    return (True, m.group(0)) if m else (False, None)


def aging_months(d):
    m = re.search(r"(\d{1,2})\s*(?:[-–]\s*\d{1,2}\s*)?(месяц|мес\b)", d)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2})\s*(лет|года|год)\b", d)
    if m and re.search(r"выдерж", d):
        return int(m.group(1)) * 12
    return None


# ---- аромат / вкус -----------------------------------------------------------------------
AROMA = {
    "красные ягоды": r"вишн|малин|клубник|земляник|красн\w+ ягод|черешн|брусник|клюкв|красн\w+ смородин|гранат",
    "тёмные ягоды": r"черн\w+ смородин|ежевик|черник|слив[аыуо]|сливов|чернослив|тут\b|темн\w+ ягод|терн\b|боярышник",
    "цитрус": r"цитрус|лимон|лайм|грейпфрут|мандарин|апельсин|цедр|бергамот",
    "косточковые": r"персик|абрикос|нектарин|желтая слив|алыч",
    "тропические": r"тропич|ананас|манго|маракуй|банан|папай|личи|экзотич",
    "яблоко/груша": r"яблок|груш|айв|дюшес",
    "цветы": r"цвет(ы|ов|очн|ущ)|акаци|роз[аыуе]\b|фиалк|жасмин|липа|липов|ромашк|бузин|герань",
    "травы/зелень": r"трав|зелен|скошенн|мят|базилик|тимьян|розмарин|крапив|чабрец|полын|шалфей|фенхель|спарж",
    "пряности/перец": r"пряност|пряный|пряные|пряным|специ|перц|перец|гвоздик|корица|корицей|мускатн\w+ орех|анис|лакрич|можжевел",
    "дуб/ваниль": r"дуб|ванил|тост|кокос|кедр|бочк|карамел",
    "шоколад/кофе": r"шоколад|кофе|какао|мокко",
    "мёд": r"\bмед[а-я]*\b|медов|воск|нектар",
    "минеральность": r"минерал|кремень|кремн|камен|соленост|солоноват|морск|йод",
    "землистое/грибное": r"земл|гриб|трюфел|лесн\w+ подстил|подлесок|прель|мох\b|табак|кожа|кожан",
    "орехи": r"орех|миндал|фундук|грецк|арахис",
    "сухофрукты": r"сухофрукт|изюм|курага|инжир|финик|чернослив|джем|варень|конфитюр|цукат",
    "выпечка/дрожжи": r"бриош|дрожж|хлеб|выпечк|сливочн|печень|булочк|автолиз",
    "мускат": r"мускат",
    "дым/смола": r"дым|копчен|смол|гарь|деготь|жженый|горелый|графит",
}


def aroma_tags(d):
    return sorted(t for t, rx in AROMA.items() if re.search(rx, d))


FOOD = {
    "сыр": r"сыр", "красное мясо": r"говядин|стейк|баранин|ягнен|дичь|шашлык|мясо|мясн|колбас",
    "птица": r"птиц|курин|индейк|утк|курица", "свинина": r"свинин|ветчин|бекон",
    "рыба": r"рыб|лосос|форел|семг|тунец|осетр|судак|треск|дорад",
    "морепродукты": r"морепродукт|устриц|креветк|мидии|краб|кальмар|гребешк|лобстер",
    "десерт": r"десерт|торт|пирожн|шоколад|мороженое|фрукт",
    "закуски": r"закуск|канапе|тапас|орешк|оливк",
    "паста/пицца": r"паст[аыу]|макарон|пицц|ризотто|лазань",
    "овощи/салат": r"овощ|салат|гриб",
    "острое/азиатская": r"остр|азиатск|карри|тайск|суши|роллы|японск|восточн",
    "аперитив": r"аперитив",
}
_PAIR_MARK = r"подав|сочета|гастроном|подойд|подходит|идеал|хорош\w* к|составит|компани|\bк (мяс|рыб|сыр|птиц|десерт|закуск|блюд)|блюдам|сырам|мясу|рыбе"


def food_mentions(desc):
    """Блюда, которые описание само упоминает рядом с маркерами сочетаемости."""
    out, quotes = set(), []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", norm(desc)):
        if re.search(_PAIR_MARK, sent):
            f = [t for t, rx in FOOD.items() if re.search(rx, sent)]
            if f:
                out.update(f)
                quotes.append(sent.strip())
    return sorted(out), (" | ".join(quotes) or None)


def serve_temp(d):
    m = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*°", d)
    return (int(m.group(1)), int(m.group(2))) if m else None


def alcohol(d):
    """Только явное указание крепости («алкоголь 13%», «13% об.»); проценты сахаров при сборе и т.п. не считаем."""
    m = re.search(r"(?:алкогол\w*|крепост\w*)\W{0,12}(\d{1,2}(?:[.,]\d)?)\s*%", d)         or re.search(r"(\d{1,2}(?:[.,]\d)?)\s*%\s*(?:об|vol|алк)", d)
    return float(m.group(1).replace(",", ".")) if m else None


def extract(name, slug, category, desc):
    """Возвращает (features:dict, evidence:dict)."""
    d = norm(desc)
    ev, f = {}, {}
    f["sweetness"], ev["sweetness"] = sweetness(name, slug, desc)
    f["sparkling"], ev["sparkling"] = is_sparkling(name, slug, desc)
    f["acidity"], ev["acidity"] = acidity(d)
    f["tannin"], ev["tannin"] = tannin(d)
    f["oak"], ev["oak"] = oak(d)
    f["aging_months"] = aging_months(d)
    f["body"], ev["body"] = body(d, f["tannin"], f["oak"], f["aging_months"])
    f["aroma"] = aroma_tags(d)
    f["food_in_desc"], ev["food_in_desc"] = food_mentions(desc or "")
    f["serve_temp"] = serve_temp(d)
    f["alcohol"] = alcohol(d)
    return f, ev
