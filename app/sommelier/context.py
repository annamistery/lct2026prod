"""Слой 1: свободный текст -> структурированный Context (правила; LLM может подменить эту функцию, см. llm.py)."""
import re

from .dishes import parse_dish
from .engine import Context
from .features import norm

COLOR_RX = [
    ("Красное", r"красн\w+|\bкрасное\b"),
    ("Белое", r"бел(ое|ого|ым|ом|ые|ых)\b"),
    ("Розовое", r"розов\w+|розе\b|rose\b"),
    ("Оранжевое", r"оранжев\w+|orange\b|апельсиновое вино|с мацерацией|скин-контакт"),
]
SWEET_PREF = [
    ({0}, r"(?<!не )(?<!полу)\bсух(ое|ого|ое вино|им)\b|только сух|люблю сух|без сахара|\bбрют\b"),
    ({0, 1}, r"не сладк\w+|несладк\w+|не люблю сладк\w+|без сладост\w*|поменьше сахар"),
    ({2}, r"полусладк\w+"),
    ({3}, r"(?<!полу)\bсладк(ое|ого|им|ое вино)\b|десертн\w+ вин|люблю сладк"),
    ({1}, r"полусух\w+"),
]
OCCASION_RX = [
    ("celebration", r"годовщин|праздн|день рожден|юбилей|новый год|свадьб|отмечаем|отпраздн|повод|торжеств"),
    ("aperitif", r"аперитив|встреч\w+ гост|перед ужином|фуршет"),
    ("date", r"свидани|романтическ|вдвоем|вдвоём|с партнер|с партнёр|с девушк|с парнем|с женой|с мужем"),
    ("business", r"деловая|деловой|переговор|партнеры по бизнесу|партнёры по бизнесу|клиент|коллег|корпоратив"),
    ("gift", r"подарок|в подарок|подарить|дарим"),
    ("family", r"семейн|с семьей|с семьёй|с родител|с родственник"),
    ("casual", r"просто так|посидеть|пятниц|будни|после работы|расслаб"),
]
EXPERT_RX = r"терруар|танинн\w+ структур|винтаж|кюве|ассамбляж|мацерац|бариков|минеральн\w+ тон|послевкус|аппелясьон|урожа[йя]"
NOVICE_RX = r"ничего не понимаю|не разбираюсь|новичок|первый раз|впервые|проще|простыми словами|просто\b.*вино|что такое"


def _stem(tok):
    return tok[: max(4, len(tok) - 2)]


class Parser:
    def __init__(self, wines):
        self.grapes = sorted({g for w in wines for g in w.get("grapes", [])})
        self.regions = sorted({w["region"] for w in wines if w["region"]})
        self._gtok = {g: [_stem(t) for t in norm(g).replace("-", " ").split()] for g in self.grapes}

    def grapes_in(self, t):
        # группируем сорта по первому слову; если названо уточнение («Каберне Совиньон») — берём только его,
        # если нет («Мускат», «Рислинг») — все сорта группы; независимые группы («Мерло») не теряем
        groups = {}
        for g, toks in self._gtok.items():
            if toks and re.search(r"(?<!\w)" + re.escape(toks[0]), t):
                groups.setdefault(toks[0], []).append(g)
        chosen = set()
        for _first, gs in groups.items():
            full = [g for g in gs if len(self._gtok[g]) > 1 and all(re.search(r"(?<!\w)" + re.escape(s), t) for s in self._gtok[g])]
            chosen |= set(full or gs)
        return chosen

    def parse(self, text):
        t = norm(text)
        ctx = Context()
        # блюдо
        dish = parse_dish(text)
        if dish:
            ctx.dish, ctx.dish_text = dish, text
        # цвет
        neg_colors = set(m for m in re.findall(r"не (?:люблю|хочу|надо|нужно) (красн|бел|розов|оранжев)", t))
        for cat, rx in COLOR_RX:
            if re.search(rx, t) and cat[:4].lower() not in [c[:4] for c in neg_colors]:
                ctx.colors.add(cat)
        # игристое
        if re.search(r"игрист|шампанск|пузырьк|брют|просекко|cremant|креман", t):
            ctx.sparkling = False if re.search(r"не (?:люблю|хочу|надо) (игрист|пузырьк|шампанск)|без пузырьк", t) else True
        # сладость
        for levels, rx in SWEET_PREF:
            if re.search(rx, t):
                ctx.sweet_ok = (ctx.sweet_ok or set()) | levels
        # сорта
        avoid_zone = re.findall(r"не люблю ([^.,;!?]*)|без ([^.,;!?]*)|кроме ([^.,;!?]*)|не хочу ([^.,;!?]*)", t)
        avoid_txt = " ".join(x for grp in avoid_zone for x in grp)
        gs = self.grapes_in(t)
        avoid_g = self.grapes_in(avoid_txt) if avoid_txt else set()
        ctx.grapes_avoid = avoid_g
        ctx.grapes_like = gs - avoid_g
        # дуб
        if re.search(r"не люблю дуб|без дуба|без бочк|не люблю бочк|без ванил|не люблю ванил|не хочу дуб|без бариков|невыдержанн|не выдержанн", t):
            ctx.avoid_oak = True
        # танины / тело
        if re.search(r"не люблю танин|без танин|мало танин|мягк\w+ танин|не терпк|без терпкост|не вяжущ|не люблю вяжущ", t):
            ctx.tannin_pref = "low"
        elif re.search(r"люблю танинн|танинн\w+ (красн|вин)|терпк|вяжущ|побольше танин", t):
            ctx.tannin_pref = "high"
        if re.search(r"легк\w+|нетяжел|не тяжел|питк|полегче|воздушн", t) and not re.search(r"легк\w+ (закуск|салат|ужин|блюд)", t):
            ctx.body_pref = "light"
        elif re.search(r"плотн\w+ вин|мощн\w+ вин|насыщенн\w+ вин|полнотел|погуще", t):
            ctx.body_pref = "full"
        # регион
        for r in self.regions:
            if norm(r).split()[0][:5] in t:
                ctx.regions.add(r)
        if re.search(r"донск|долина дона", t):
            ctx.regions.add("Долина Дона")
        # повод
        for oc, rx in OCCASION_RX:
            if re.search(rx, t):
                ctx.occasion = oc
                break
        if ctx.dish and ctx.dish.get("course") == "аперитив" and not ctx.occasion:
            ctx.occasion = "aperitif"
        # бюджет (в базе цен нет — фиксируем, но не применяем)
        m = re.search(r"(до|около|не дороже|не более|в пределах)\s*(\d[\d\s]*)\s*(руб|₽|р\b|т\.?р)|недорог\w+|не сильно дорого|бюджетн\w+|подороже|премиум|дорог\w+", t)
        if m:
            ctx.budget = m.group(0)
            ctx.notes.append("budget_unsupported")
        # опыт
        if re.search(EXPERT_RX, t):
            ctx.experience = "expert"
        elif re.search(NOVICE_RX, t):
            ctx.experience = "novice"
        return ctx
