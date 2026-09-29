from app.pipelines.search.cascade.label_check import (
    KEEP,
    REJECT,
    SWITCH,
    CatalogCard,
    LabelReading,
    check_candidate,
    decide,
    norm,
    sweetness_class,
)


def card(slug, name, winery, grapes=(), category="Красное", type_="красное сухое"):
    return CatalogCard(slug=slug, name=name, winery=winery, grapes=tuple(grapes), category=category, type=type_)


RUBIN = card("rubin", "Рубин кларет . Красная стрелка", "Denisov Winery", ["Рубин Голодриги"], "Розовое", "розовое")
PINOT = card("pinot", "Пино Нуар кларет. Красная стрелка", "Denisov Winery", ["Пино Нуар"], "Розовое", "розовое")
YUZHNY = card("yuzhny", "Южный Лес", "Усадьба Дивноморское", ["Мерло"])


def test_norm_transliterates_latin():
    assert norm("DENISOV Pinot") == "денисов пино"
    assert norm("Cabernet Sauvignon") == "каберне совиньон"
    assert norm("Ёлка, брют!") == "елка брют"


def test_sweetness_class_prefers_specific_word():
    assert sweetness_class("вино полусухое белое") == "semidry"
    assert sweetness_class("розовое игристое полусладкое") == "semisweet"
    assert sweetness_class("экстра брют") == "dry"
    assert sweetness_class("розовое сладость не указана") is None


def test_switch_to_lower_candidate_confirmed_by_name():
    label = LabelReading(producer="DENISOV", name="Красная стрелка", text="Denisov Самара Красная стрелка Рубин")
    verdict = decide(label, [PINOT, RUBIN])
    assert verdict.action == SWITCH and verdict.slug == "rubin"


def test_reject_when_label_name_is_not_in_any_candidate():
    label = LabelReading(producer="Усадьба Дивноморское", name="Вечерница", text="Усадьба Дивноморское Терруар Вторая линия Вечерница")
    assert decide(label, [YUZHNY]).action == REJECT


def test_keep_when_label_confirms_top1():
    label = LabelReading(producer="Усадьба Дивноморское", name="Южный лес", grapes=("Мерло",), text="Южный лес Мерло")
    verdict = decide(label, [YUZHNY, RUBIN])
    assert verdict.action == KEEP and verdict.slug == "yuzhny"


def test_keep_when_label_is_unreadable():
    assert decide(LabelReading(), [YUZHNY]).action == KEEP


def test_sweetness_conflict():
    semisweet = card("ps", "Русское игристое полусладкое красное", "Абрау-Дюрсо", [], "Красное", "красное игристое полусладкое")
    check = check_candidate(LabelReading(producer="Абрау-Дюрсо", sweetness="брют", text="Русское игристое брют"), semisweet)
    assert any("сладость" in c for c in check.conflicts)


def test_neighbouring_sweetness_is_not_a_conflict():
    semidry = card("psh", "Зелёное вино", "Фанагория", [], "Белое", "белое полусухое")
    check = check_candidate(LabelReading(producer="Фанагория", sweetness="сухое", text="Зелёное вино сухое"), semidry)
    assert not check.conflicts


def test_grape_conflict():
    check = check_candidate(LabelReading(producer="Denisov", grapes=("Рислинг",), text="Denisov Рислинг"), PINOT)
    assert any("сорт" in c for c in check.conflicts)


def test_misread_grape_field_is_not_a_conflict_when_the_text_has_the_grape():
    kokur = card("kokur", "Кокур Классика", "Валерий Захарьин", ["Кокур Белый"], "Белое", "белое игристое сухое")
    label = LabelReading(producer="Валерий Захарьин", name="Кокур", grapes=("САМСОН",), text="Кокур белое брют classic Самсон")
    assert decide(label, [kokur]).action == KEEP
