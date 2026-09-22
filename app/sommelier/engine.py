"""Детерминированный движок подбора: жёсткие фильтры -> общие оси -> скоринг правил -> диверсификация.

Все факты о вине берутся из catalog.json (features + evidence). Веса правил — «настройки вкуса» системы,
их предполагается калибровать по обратной связи (см. feedback.py).
Нет данных о признаке -> вклад правила = 0 и признак попадает в список «в базе не указано».
"""
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ---- веса правил (методика из ТЗ) ---------------------------------------------------------------
W = dict(
    weight=3.0,      # соответствие по весу/интенсивности
    fat_acid=2.0,    # кислотность режет жир
    sweet=3.5,       # сладость вина >= сладости блюда (+ контраст с острым/солёным)
    tannin=2.5,      # танин: связывается с белком и жиром, конфликтует с солью и острым
    region=1.0,      # «что растёт вместе, то сочетается»
    aroma=1.5,       # дополнение/контраст по аромату (только если аромат есть в описании)
    occasion=1.2,
    prefs=2.5,
)
MAX_POS = sum(W.values())


@dataclass
class Context:
    dish_text: str | None = None
    dish: dict | None = None                 # результат dishes.parse_dish
    colors: set = field(default_factory=set)    # {'Красное', ...}; пусто = любые
    sparkling: bool | None = None
    sweet_ok: set | None = None              # допустимые уровни сладости {0,1,2,3}
    grapes_like: set = field(default_factory=set)
    grapes_avoid: set = field(default_factory=set)
    avoid_oak: bool = False
    tannin_pref: str | None = None           # 'low' | 'high'
    body_pref: str | None = None             # 'light' | 'full'
    regions: set = field(default_factory=set)
    occasion: str | None = None              # celebration|aperitif|date|business|gift|family|casual
    experience: str = "novice"                  # novice|enthusiast|expert
    budget: str | None = None                # в базе цен нет — только фиксируем
    excluded_ids: set = field(default_factory=set)
    notes: list = field(default_factory=list)

    def merge(self, new):
        """Многоходовой диалог: новые уточнения перекрывают старые, остальное сохраняется."""
        for k, v in vars(new).items():
            if k in ("notes",):
                self.notes += [n for n in v if n not in self.notes]
            elif isinstance(v, set):
                setattr(self, k, getattr(self, k) | v)
            elif v not in (None, False, "") and not (k == "experience" and v == "novice"):
                setattr(self, k, v)
        return self


def _sweet_bucket(w):
    s = w["features"]["sweetness"]
    return None if s is None else s


# ---- вспомогательное ------------------------------------------------------------------------------
def body_words(b):
    if b is None:
        return None
    return "лёгкое" if b < 1.7 else "среднее" if b < 2.8 else "плотное"


def acid_words(a):
    if a is None:
        return None
    return "низкая" if a < 1.5 else "умеренная" if a < 2.8 else "высокая"


def tannin_words(t):
    if t is None:
        return None
    return "мягкие" if t < 1.5 else "умеренные" if t < 2.8 else "выраженные"


class Engine:
    def __init__(self, wines):
        self.wines = wines
        self.by_id = {w["id"]: w for w in wines}
        self.grape_count = Counter(g for w in wines for g in self.grapes_of(w))
        cnts = sorted(self.grape_count.values())
        self.classic_threshold = cnts[int(len(cnts) * 0.85)] if cnts else 0   # «классика» — верхние 15% по частоте
        self.rare_threshold = 8
        # если в описании красного вина танины не упомянуты, берём МЕДИАНУ по красным каталога, где они упомянуты
        # (оценка по самой базе, вклад правила вдвое слабее и помечается как оценка)
        reds = sorted(w["features"]["tannin"] for w in wines if w["category"] == "Красное" and w["features"]["tannin"] is not None)
        self.red_tannin_prior = reds[len(reds) // 2] if reds else None

    @staticmethod
    def grapes_of(w):
        return w.get("grapes") or []

    # ---- жёсткие фильтры ------------------------------------------------------------------------
    def hard_filter(self, ctx):
        out, dropped = [], Counter()
        for w in self.wines:
            f = w["features"]
            if w["id"] in ctx.excluded_ids:
                dropped["исключено вами ранее"] += 1
                continue
            if ctx.colors and w["category"] not in ctx.colors:
                dropped["другой цвет"] += 1
                continue
            if ctx.sparkling is True and not f["sparkling"]:
                dropped["не игристое"] += 1
                continue
            if ctx.sparkling is False and f["sparkling"]:
                dropped["игристое"] += 1
                continue
            if ctx.sweet_ok is not None and f["sweetness"] is not None and f["sweetness"] not in ctx.sweet_ok:
                dropped["другая сладость"] += 1
                continue
            if ctx.grapes_avoid and set(self.grapes_of(w)) & ctx.grapes_avoid:
                dropped["нежелательный сорт"] += 1
                continue
            if ctx.avoid_oak and f["oak"]:
                dropped["упомянут дуб/бочка"] += 1
                continue
            out.append(w)
        return out, dropped

    # ---- скоринг --------------------------------------------------------------------------------
    def score(self, w, ctx):
        """Возвращает (score 0-100, компоненты, неизвестные признаки)."""
        f = w["features"]
        comps, unknown = [], []
        dish = ctx.dish
        ax = dish["axes"] if dish else None
        tags = set(dish["tags"]) if dish else set()

        def add(key, v, msg, **kw):
            comps.append(dict(rule=key, value=max(-1.0, min(1.0, v)), weight=W[key], msg=msg, **kw))

        if ax:
            # 1. вес / интенсивность
            if f["body"] is not None:
                delta = abs(ax["intensity"] - f["body"])
                v = 1 - 2 * delta / 3
                add("weight", v, "вес", dish_i=ax["intensity"], wine_body=f["body"], delta=round(delta, 2))
            else:
                unknown.append("тело вина")

            # 2. кислотность режет жир
            if ax["fat"] >= 2:
                need = ax["fat"] / 4
                if f["acidity"] is not None:
                    v = need * (f["acidity"] - 2) / 1.5
                    add("fat_acid", v, "жир/кислота", fat=ax["fat"], acid=f["acidity"])
                else:
                    unknown.append("кислотность вина")

            # 3. сладость: вино не должно быть менее сладким, чем блюдо
            ws = f["sweetness"]
            ds = min(3.0, ax["sweet"])
            contrast_ok = bool({"сыр с плесенью", "печень"} & tags) or (ax["heat"] >= 2.5)
            if ws is None:
                if ds >= 1.0:
                    unknown.append("сладость вина")
                    add("sweet", -0.3, "сладость неизвестна при сладком блюде", dish_sweet=ds, wine_sweet=None)
                else:
                    unknown.append("сладость вина")
            else:
                # к несладкому блюду с «сладким акцентом» (соус) допускаем запас 0.6, к десерту — нет
                slack = 0.0 if ({"десерт", "фрукты", "шоколад"} & tags) else 0.6
                if ds - ws > slack:
                    v = -min(1.0, (ds - ws - slack) / 1.5)
                    add("sweet", v, "вино суше блюда", dish_sweet=ds, wine_sweet=ws)
                elif ds >= 1.0:
                    add("sweet", 0.5 if ws - ds <= 1 else 0.2, "сладость вина не ниже сладости блюда", dish_sweet=ds, wine_sweet=ws)
                elif contrast_ok and ws in (1, 2):
                    add("sweet", 0.6, "контраст с острым/солёным", dish_sweet=ds, wine_sweet=ws, contrast=True)
                elif ws >= 2 and ds < 1 and ctx.dish.get("course") not in ("аперитив", "десерт"):
                    add("sweet", -0.2 * ws, "сладкое вино к несладкому блюду", dish_sweet=ds, wine_sweet=ws)
                elif ws <= 1 and ds < 1:
                    add("sweet", 0.4 if ws == 0 else 0.2, "несладкое вино к несладкому блюду", dish_sweet=ds, wine_sweet=ws)

            # 4. танины
            t, est = f["tannin"], False
            if t is None and w["category"] == "Красное" and self.red_tannin_prior is not None:
                t, est = self.red_tannin_prior, True
                unknown.append("танины (оценка по красным каталога)")
            if t is not None:
                hot = max(ax["heat"] / 4, max(0.0, ax["salt"] - 1.5) / 2.5)
                pos = (ax["fat"] >= 2.5) or bool({"красное мясо", "сыр"} & tags and ax["fat"] >= 2)
                v = 0.0
                if t >= 2 and hot > 0.2:
                    v -= min(1.0, (t - 1) / 2.5 * hot * 1.6)
                if t >= 2 and pos:
                    v += 0.6 * min(1.0, t / 3.5) * min(1.0, ax["fat"] / 3)
                if est:
                    v *= 0.5
                if v != 0:
                    add("tannin", v, "танины", tannin=t, est=est, heat=ax["heat"], salt=ax["salt"], fat=ax["fat"])
            elif w["category"] != "Красное" and ax["heat"] >= 2.5:
                pass  # у белых/розовых танины в описании не заявлены — конфликта нет, но и вклада нет

            # 5. региональный принцип
            if dish.get("region") and dish["region"] == w["region"]:
                add("region", 1.0, "регион совпадает с кухней", region=w["region"])

            # 6. аромат: дополнение/контраст
            links = _aroma_links(tags, ax)
            if links and f["aroma"]:
                hit = sorted(set(links["complement"]) & set(f["aroma"]))
                if hit:
                    add("aroma", 0.8, "дополнение по аромату", aromas=hit, why=links["why"])
                hit2 = sorted(set(links["contrast"]) & set(f["aroma"]))
                if hit2:
                    add("aroma", 0.6, "контраст по аромату", aromas=hit2, why=links["why_contrast"])
            # 6b. блюда, которые само описание вина называет подходящими
            fd = set(f["food_in_desc"])
            if fd:
                match = _dish_food_tags(tags) & fd
                if match:
                    add("aroma", 1.0, "описание вина само упоминает такое блюдо", foods=sorted(match), quote=w["evidence"].get("food_in_desc"))

        # 7. повод
        oc = ctx.occasion
        if oc:
            sp = f["sparkling"]
            if oc == "celebration" and sp:
                add("occasion", 0.7, "праздничный повод — игристое", occasion=oc)
            elif oc == "aperitif":
                if sp or (w["category"] == "Белое" and f["body"] is not None and f["body"] < 2):
                    add("occasion", 0.7, "лёгкое/игристое подходит для аперитива", occasion=oc)
                elif f["body"] is not None and f["body"] > 3:
                    add("occasion", -0.5, "плотное вино тяжеловато для аперитива", occasion=oc)
            elif oc == "business":
                gs = self.grapes_of(w)
                if gs and max(self.grape_count[g] for g in gs) >= self.classic_threshold:
                    add("occasion", 0.5, "деловая встреча — узнаваемый сорт без сюрпризов", occasion=oc)
            elif oc in ("date", "gift") and f["sparkling"]:
                add("occasion", 0.3, "игристое — нейтральный «безопасный» формат", occasion=oc)

        # 8. предпочтения гостя
        pv, pmsg = 0.0, []
        gs = set(self.grapes_of(w))
        if ctx.grapes_like and gs & ctx.grapes_like:
            pv += 0.8
            pmsg.append("любимый сорт")
        if ctx.regions and w["region"] in ctx.regions:
            pv += 0.4
            pmsg.append("предпочтительный регион")
        if ctx.tannin_pref and f["tannin"] is not None:
            if ctx.tannin_pref == "high":
                pv += 0.5 if f["tannin"] >= 2.8 else -0.3
            else:
                pv += 0.5 if f["tannin"] <= 1.5 else -0.4
            pmsg.append("танины по вкусу")
        if ctx.body_pref and f["body"] is not None:
            target = 1.2 if ctx.body_pref == "light" else 3.5
            pv += 0.6 * (1 - abs(target - f["body"]) / 2.5)
            pmsg.append("тело по вкусу")
        if ctx.sweet_ok is not None and f["sweetness"] is None:
            pv -= 0.3
            pmsg.append("сладость не указана в базе")
        if pmsg:
            add("prefs", pv, ", ".join(pmsg))

        raw = sum(c["weight"] * c["value"] for c in comps)
        # нормировка по правилам, применимым к ЗАПРОСУ (а не к данным о вине): нет данных -> нет баллов,
        # поэтому вино с подтверждёнными признаками обходит вино, о котором в базе ничего не сказано
        possible = self._applicable(ctx)
        norm_score = 50 + 50 * raw / max(possible, 3.0)
        return round(max(0.0, min(100.0, norm_score)), 1), comps, sorted(set(unknown))

    @staticmethod
    def _applicable(ctx):
        p = 0.0
        if ctx.dish:
            ax, tags = ctx.dish["axes"], set(ctx.dish["tags"])
            p += W["weight"] + W["sweet"]
            if ax["fat"] >= 2:
                p += W["fat_acid"]
            if ax["fat"] >= 2.5 or ax["heat"] >= 1.2 or ax["salt"] >= 2.5:
                p += W["tannin"]
            if ctx.dish.get("region"):
                p += W["region"]
            if _aroma_links(tags, ax):
                p += W["aroma"]
        if ctx.occasion:
            p += W["occasion"]
        if (ctx.grapes_like or ctx.regions or ctx.tannin_pref or ctx.body_pref):
            p += W["prefs"]
        return p

    # ---- ранжирование + разнообразие ---------------------------------------------------------------
    def recommend(self, ctx, k=3):
        pool, dropped = self.hard_filter(ctx)
        scored = []
        for w in pool:
            s, comps, unk = self.score(w, ctx)
            scored.append(dict(wine=w, score=s, comps=comps, unknown=unk))
        scored.sort(key=lambda r: -r["score"])
        picks = self._diversify(scored, k)
        return dict(picks=picks, pool_size=len(pool), dropped=dict(dropped), total=len(self.wines))

    def _sim(self, a, b):
        wa, wb = a["wine"], b["wine"]
        s = 0.0
        if set(self.grapes_of(wa)) & set(self.grapes_of(wb)):
            s += 0.4
        if wa["category"] == wb["category"] and wa["features"]["sparkling"] == wb["features"]["sparkling"]:
            s += 0.3
        if wa["region"] == wb["region"]:
            s += 0.15
        if wa["winery"] == wb["winery"]:
            s += 0.15
        return s

    def _diversify(self, scored, k):
        if not scored:
            return []
        seen_names, cands = set(), []
        for r in scored:                       # одинаковые названия (разные урожаи) — один раз
            key = r["wine"]["name"].strip().lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            cands.append(r)
        cands = cands[:120]
        picks = [cands[0]]
        floor = cands[0]["score"] - 15         # разнообразие не должно тянуть в откровенно слабое
        while len(picks) < k:
            best, best_v = None, -1e9
            for r in cands:
                if r in picks or r["score"] < floor:
                    continue
                v = r["score"] - 25 * max(self._sim(r, p) for p in picks)
                if v > best_v:
                    best, best_v = r, v
            if best is None:
                break
            picks.append(best)
        for i, r in enumerate(picks):
            r["role"] = self._role(r, picks, i)
        return picks

    def _role(self, r, picks, i):
        w = r["wine"]
        gs = self.grapes_of(w)
        freq = max((self.grape_count[g] for g in gs), default=0)
        if i == 0:
            return "Лучшее совпадение по правилам"
        if gs and freq <= self.rare_threshold:
            return "Нестандартный выбор (редкий сорт в каталоге)"
        if gs and freq >= self.classic_threshold:
            return "Классический вариант"
        return "Другой стиль для сравнения"

    # ---- уместность типов вина для ситуации ---------------------------------------------------------
    def type_fit(self, ctx):
        """Какие ТИПЫ вина уместны к ситуации: среднее по 5 лучшим винам каждого типа + главная причина."""
        base = Context(**{**vars(ctx), "colors": set(), "sparkling": None, "sweet_ok": None,
                          "grapes_avoid": set(), "avoid_oak": False, "excluded_ids": set(), "grapes_like": set()})
        groups = defaultdict(list)
        for w in self.wines:
            f = w["features"]
            if f["sweetness"] is None:
                continue
            key = (w["category"], f["sparkling"], {0: "сухое", 1: "полусухое", 2: "полусладкое", 3: "сладкое"}[f["sweetness"]])
            s, comps, _ = self.score(w, base)
            groups[key].append((s, comps))
        rows = []
        for key, lst in groups.items():
            if len(lst) < 5:
                continue
            lst.sort(key=lambda x: -x[0])
            top = lst[:5]
            mean = sum(s for s, _ in top) / len(top)
            neg, pos = defaultdict(float), defaultdict(float)
            for _, comps in top:
                for c in comps:
                    (neg if c["value"] < 0 else pos)[c["rule"]] += c["value"] * c["weight"] / len(top)
            worst = min(neg.items(), key=lambda kv: kv[1])[0] if neg else None
            best = max(pos.items(), key=lambda kv: kv[1])[0] if pos else None
            rows.append(dict(cat=key[0], sparkling=key[1], sweet=key[2], mean=round(mean, 1), n=len(lst),
                             worst=worst, worst_val=round(neg.get(worst, 0), 2) if worst else 0, best=best))
        rows.sort(key=lambda r: -r["mean"])
        if not rows:
            return []
        top = rows[0]["mean"]
        for r in rows:
            gap = top - r["mean"]
            r["verdict"] = "подходит" if gap <= 6 else "допустимо, с оговорками" if gap <= 14 else "лучше избегать"
            if r["worst"] == "sweet" and r["worst_val"] < -1.0:
                r["verdict"] = "лучше избегать"
        return rows

    # ---- статистика по сортам/типам (для «в чём различие») -------------------------------------------
    def group_profile(self, ws, label):
        n = len(ws)
        fs = [w["features"] for w in ws]
        def mean(key):
            vs = [f[key] for f in fs if f[key] is not None]
            return (round(st.mean(vs), 2), len(vs)) if vs else (None, 0)
        allc = Counter(t for w in self.wines for t in w["features"]["aroma"])
        gc = Counter(t for f in fs for t in f["aroma"])
        N = len(self.wines)
        lift = {t: (c / n) / (allc[t] / N) for t, c in gc.items() if c >= max(3, 0.15 * n)}
        top_aroma = [t for t, _ in sorted(gc.items(), key=lambda kv: -kv[1])[:4]]
        distinct = [t for t, _ in sorted(lift.items(), key=lambda kv: -kv[1]) if lift[t] > 1.25][:3]
        def share(pred):
            return round(sum(1 for f in fs if pred(f)) / n, 2)
        return dict(
            label=label, n=n,
            body=mean("body"), acidity=mean("acidity"), tannin=mean("tannin"),
            tannin_mention=share(lambda f: f["tannin"] is not None),
            sweet_share=share(lambda f: f["sweetness"] is not None and f["sweetness"] >= 2),
            dry_share=share(lambda f: f["sweetness"] == 0),
            oak_share=share(lambda f: f["oak"]),
            sparkling_share=share(lambda f: f["sparkling"]),
            top_aroma=top_aroma, distinct_aroma=distinct,
            aroma_share={t: c / n for t, c in gc.items()},
            regions=Counter(w["region"] for w in ws).most_common(3),
            foods=[t for t, _ in Counter(t for f in fs for t in f["food_in_desc"]).most_common(3)],
            examples=[w["name"] for w in sorted(ws, key=lambda w: -len(w["description"]))[:3]],
        )


# ---- связки блюдо -> аромат (дополнение / контраст) --------------------------------------------------
def _aroma_links(tags, ax):
    rules = [
        ({"грибы"}, dict(complement=["землистое/грибное", "орехи"], contrast=[], why="землистые тона вина перекликаются с грибами", why_contrast="")),
        ({"баранина"}, dict(complement=["травы/зелень", "пряности/перец"], contrast=[], why="пряно-травяные ноты поддерживают баранину", why_contrast="")),
        ({"красное мясо", "гриль"}, dict(complement=["дым/смола", "пряности/перец", "тёмные ягоды"], contrast=[], why="пряные и тёмно-ягодные ноты дополняют мясо на огне", why_contrast="")),
        ({"утка"}, dict(complement=["красные ягоды"], contrast=[], why="красные ягоды перекликаются с фруктовым соусом", why_contrast="")),
        ({"сладко-кислый"}, dict(complement=["красные ягоды", "тёмные ягоды"], contrast=[], why="ягодные ноты вина перекликаются с ягодным соусом", why_contrast="")),
        ({"морепродукты"}, dict(complement=["минеральность"], contrast=["цитрус"], why="минеральность перекликается с солоноватостью морепродуктов", why_contrast="цитрусовая свежесть контрастирует с морской солью")),
        ({"рыба"}, dict(complement=["минеральность"], contrast=["цитрус"], why="минеральные тона поддерживают рыбу", why_contrast="цитрус освежает рыбное блюдо")),
        ({"сыр с плесенью"}, dict(complement=[], contrast=["мёд", "сухофрукты"], why="", why_contrast="медовая/сухофруктовая сладость контрастирует с солёным сыром")),
        ({"десерт"}, dict(complement=["косточковые", "мёд", "тропические", "сухофрукты", "красные ягоды"], contrast=[], why="фруктовые и медовые ноты перекликаются с десертом", why_contrast="")),
        ({"шоколад"}, dict(complement=["шоколад/кофе", "тёмные ягоды", "сухофрукты"], contrast=[], why="шоколадные и тёмно-ягодные ноты дополняют шоколад", why_contrast="")),
        ({"овощи"}, dict(complement=["травы/зелень"], contrast=["цитрус"], why="травянистые ноты перекликаются с овощами и зеленью", why_contrast="цитрусовая свежесть освежает овощи")),
        ({"острое"}, dict(complement=[], contrast=["тропические", "косточковые", "цветы"], why="", why_contrast="фруктовая мягкость смягчает остроту")),
    ]
    comp, cont, why, why_c = [], [], [], []
    for req, d in rules:
        if req & tags:
            comp += d["complement"]
            cont += d["contrast"]
            if d["why"] and d["why"] not in why:
                why.append(d["why"])
            if d["why_contrast"] and d["why_contrast"] not in why_c:
                why_c.append(d["why_contrast"])
    if not (comp or cont):
        return None
    return dict(complement=comp, contrast=cont, why="; ".join(why), why_contrast="; ".join(why_c))


def _dish_food_tags(tags):
    m = {"красное мясо": "красное мясо", "баранина": "красное мясо", "гриль": "красное мясо", "дичь": "красное мясо",
         "птица": "птица", "утка": "птица", "свинина": "свинина", "рыба": "рыба", "морепродукты": "морепродукты",
         "сыр": "сыр", "десерт": "десерт", "фрукты": "десерт", "шоколад": "десерт", "закуски": "закуски",
         "паста": "паста/пицца", "овощи": "овощи/салат", "грибы": "овощи/салат", "острое": "острое/азиатская", "азиатская": "острое/азиатская"}
    return {m[t] for t in tags if t in m}
