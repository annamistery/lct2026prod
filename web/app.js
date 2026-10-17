import { loadDetector, detect, draw } from './detector.js';

// Сканер «Своё Вино»: фото -> каскад (/api/cascade/search) -> ответ «найдено» / «похоже» / «нет в каталоге» + сомелье.
// На сервер всегда уходит полный кадр или исходное фото: этикетку вырезает серверная YOLO, и пороги зон ответа
// откалиброваны именно на таких кропах. Детектор в браузере только рисует рамку-подсказку.

const config = window.LCT_CONFIG || { apiBase: '' };
const storage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* приватный режим */ } },
    remove(key) { try { localStorage.removeItem(key); } catch { /* приватный режим */ } },
};
// Адрес API: ?api=https://… из ссылки (для копии сканера на GitHub Pages) > сохранённый в настройках > config.js
const apiFromLink = new URLSearchParams(location.search).get('api');
const linkBase = apiFromLink && /^https?:\/\/\S+$/.test(apiFromLink) ? apiFromLink.replace(/\/+$/, '') : null;
if (linkBase) storage.set('vina_api_url', linkBase);
config.apiBase = linkBase || storage.get('vina_api_url') || config.apiBase || '';
const api = path => (/^(data:|blob:|https?:)/.test(path) ? path : `${config.apiBase || ''}${path}`);

const $ = id => document.getElementById(id);
const video = $('camera-feed');
const overlay = $('overlay');
const btnSnap = $('btn-snap');
const fileInput = $('file-input');
const scannerHint = $('scanner-hint');

const views = { scanner: $('scanner-view'), loading: $('loading-view'), result: $('result-view'), original: $('original-view'), batch: $('batch-view') };
let returnTo = 'scanner';    // куда ведёт «Назад» из ответа: к сканеру или к проверке набора
function showView(name) {
    Object.values(views).forEach(view => view.classList.remove('active'));
    views[name].classList.add('active');
    window.scrollTo({ top: 0 });
}

const STATUS = {
    found: { icon: '✓', title: 'Вино найдено в каталоге' },
    probable: { icon: '?', title: 'Похоже на это вино' },
    not_in_catalog: { icon: '✗', title: 'Данного вина нет в каталоге' },
};

let busy = false;
let request = 0;             // номер текущего распознавания: ответы от прошлых фото отбрасываются
let current = null;          // { data, slug, info }
let photoUrl = null;
let sommelierSession = null;

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

async function getJson(url, options) {
    const response = await fetch(url, options);
    let data = null;
    try { data = await response.json(); } catch { /* не JSON */ }
    if (!response.ok) throw new Error((data && data.detail && (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))) || `HTTP ${response.status}`);
    return data;
}

const fmt = (value, digits = 3) => (typeof value === 'number' ? value.toFixed(digits) : '—');

// ---------- Пороги зон ответа ----------
async function loadThresholds() {
    const body = $('thresholds-body');
    try {
        const t = await getJson(api('/api/cascade/thresholds'));
        const list = el('ul');
        list.append(
            el('li', '', `«Найдено»: сходство SigLIP 2 ≥ ${fmt(t.found_min_similarity)} и отрыв от второго кандидата ≥ ${fmt(t.found_min_margin)}`),
            el('li', '', `«Нет в каталоге»: сходство SigLIP 2 < ${fmt(t.reject_below_similarity)}`),
            el('li', '', '«Похоже»: всё между ними — лучший кандидат показан, его стоит сверить с этикеткой'),
        );
        if (typeof t.predict_threshold === 'number') list.append(el('li', '', `Дополнительный порог отказа: уверенность < ${fmt(t.predict_threshold)}`));
        const intro = el('div', '', `DINOv2 отбирает ${t.candidate_pool} кандидатов, оценка = SigLIP 2 (2 вида кропа) + ${t.v1_weight} · DINOv2 (2 вида).`);
        body.replaceChildren(intro, list);
        if (t.stage === 'v1_only') body.append(el('div', 'somm-note', 'SigLIP 2 не загружен: все ответы получают статус «похоже».'));
    } catch (error) {
        body.textContent = `Пороги недоступны: ${error.message}`;
    }
}

// ---------- Распознавание ----------
async function cascadeSearch(blob) {
    const body = new FormData();
    body.append('image', blob, blob.name || 'photo.jpg');
    body.append('k', '5');
    return getJson(api('/api/cascade/search'), { method: 'POST', body });
}

async function submit(blob) {
    if (!blob || busy) return;
    busy = true;
    const id = ++request;
    btnSnap.disabled = true;
    $('loading-text').textContent = 'Анализ этикетки…';
    showView('loading');
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = URL.createObjectURL(blob);
    try {
        const data = await cascadeSearch(blob);
        if (id !== request) return;
        returnTo = 'scanner';
        $('btn-back-label').textContent = 'Назад к сканеру';
        renderResult(data, id, photoUrl);
        showView('result');
    } catch (error) {
        scannerHint.textContent = `Ошибка распознавания: ${error.message}`;
        showView('scanner');
    } finally {
        busy = false;
        btnSnap.disabled = false;
    }
}

function renderResult(data, id, originalUrl) {
    const winner = data.winner;
    const view = STATUS[data.status] || STATUS.probable;
    current = { data, slug: winner ? winner.slug : null, info: null };

    // 1. Статус ответа и почему он выбран
    $('status-banner').className = `status-banner ${data.status}`;
    $('status-title').textContent = `${view.icon} ${data.status === 'probable' ? data.message : view.title}`;
    $('status-reason').textContent = data.decision ? data.decision.reason : '';
    const meta = $('status-meta');
    meta.replaceChildren();
    const metaItem = (label, value) => { const span = el('span', '', `${label}: `); span.append(el('b', '', value)); meta.append(span); };
    if (data.decision) {
        metaItem('сходство', fmt(data.decision.similarity));
        metaItem('отрыв', fmt(data.decision.margin));
    }
    if (data.timings) metaItem('время', `${Math.round(data.timings.total_ms)} мс`);
    if (data.stage_reached === 'v1_only') metaItem('режим', 'только DINOv2');

    // 2. Карточка вина — только когда каскад выбрал вино
    $('wine-block').classList.toggle('hidden', !winner);
    $('about-answer').classList.add('hidden');
    $('pairing-answer').textContent = '';
    $('pairing-answer').classList.remove('pairing-graphic');
    document.querySelectorAll('.btn-pairing').forEach(btn => btn.classList.remove('selected'));
    if (winner) renderWine(winner, data.status, id);

    // 3. Кандидаты: для «похоже» — другие вина-кандидаты, для «нет в каталоге» — ближайшие этикетки (другие вина)
    const others = winner ? data.final_results.slice(1) : data.final_results;
    const showCandidates = data.status !== 'found' && others.length > 0;
    $('candidates-section').classList.toggle('hidden', !showCandidates);
    if (showCandidates) {
        $('candidates-title').textContent = winner ? 'Другие кандидаты' : 'Похожие этикетки каталога';
        $('candidates-hint').textContent = winner
            ? 'Если на этикетке другое вино — оно может быть среди этих кандидатов.'
            : 'Это другие вина: такого низкого сходства не бывает у вин из каталога.';
        $('candidates-list').replaceChildren(...others.slice(0, 4).map(item => listItem(
            item.image_url, item.title, item.manufacturer, `сходство SigLIP 2: ${fmt(item.v4_similarity ?? item.v1_similarity)}`,
        )));
    }

    // 4. Чат с сомелье
    $('chat-hint').textContent = winner
        ? 'Спросите сомелье о блюде, вкусе или другом вине из каталога.'
        : 'Этого вина нет в каталоге. Опишите блюдо или вкус — сомелье подберёт вино из каталога.';
    $('chat-answer').textContent = '';

    // 5. Исходное фото и кроп этикетки
    $('original-img').src = originalUrl;
    $('crop-img').src = data.bbox_crop || originalUrl;
}

function listItem(imageUrl, title, subtitle, meta, reasons) {
    const row = el('div', 'list-item');
    if (imageUrl) {
        const img = el('img');
        img.src = api(imageUrl);
        img.alt = '';
        img.loading = 'lazy';
        img.addEventListener('click', () => openLightbox(img.src, title));
        row.append(img);
    } else {
        row.append(el('div', 'noimg', 'нет фото'));
    }
    const body = el('div');
    body.append(el('div', 'item-title', title));
    if (subtitle) body.append(el('div', 'item-meta', subtitle));
    if (meta) body.append(el('div', 'item-meta', meta));
    if (reasons) body.append(el('div', 'item-reasons', reasons));
    row.append(body);
    return row;
}

function setTag(node, value) {
    node.textContent = value || '';
    node.classList.toggle('hidden', !value);
}

async function renderWine(winner, status, id) {
    $('wine-title').textContent = winner.title;
    $('wine-winery').textContent = winner.manufacturer || '';
    $('wine-description').textContent = winner.description || '';
    $('wine-description').classList.remove('structured');
    $('wine-description').classList.toggle('hidden', !winner.description);
    ['wine-category', 'wine-color', 'wine-region'].forEach(key => setTag($(key), ''));
    $('wine-grape-row').classList.add('hidden');
    $('somm-note-probable').classList.toggle('hidden', status !== 'probable');
    const image = $('wine-image');
    image.src = api(winner.image_url);
    image.alt = winner.title;
    initRating(winner.slug || winner.product_id);

    const alternatives = $('alternatives-list');
    alternatives.replaceChildren(el('div', 'item-meta', 'Сомелье подбирает похожие вина…'));
    const slug = winner.slug ? encodeURIComponent(winner.slug) : null;
    const [product, info, alts] = await Promise.allSettled([
        getJson(api(`/api/products/${winner.product_id}`)),
        slug ? getJson(api(`/api/sommelier/wine/${slug}`)) : Promise.reject(new Error('нет slug')),
        slug ? getJson(api(`/api/sommelier/wine/${slug}/alternatives?limit=4`)) : Promise.reject(new Error('нет slug')),
    ]);
    if (id !== request) return;

    // Фото бутылки из каталога, если есть; иначе — кроп этикетки
    if (product.status === 'fulfilled' && product.value.image_url) image.src = api(product.value.image_url);

    if (info.status === 'fulfilled' && info.value.kind === 'wine_info') {
        current.info = info.value;
        const card = info.value.card || {};
        setTag($('wine-category'), card.category);
        setTag($('wine-color'), card.color && card.color.length <= 30 ? card.color : '');
        setTag($('wine-region'), card.region);
        if (card.grape) {
            $('wine-grape').textContent = card.grape;
            $('wine-grape-row').classList.remove('hidden');
        }
        if (info.value.profile) renderProfile($('wine-description'), info.value.profile);
    }

    const items = alts.status === 'fulfilled' ? alts.value.items : [];
    alternatives.replaceChildren(...(items.length
        ? items.map(item => listItem(item.image_url, item.name, item.winery, item.facts, item.reasons.join(' · ')))
        : [el('div', 'item-meta', 'Похожих вин в каталоге сомелье не нашёл.')]));
}

// Структурированное описание вина: блоки «Цвет / Аромат / Вкус / Гастрономия», шкалы вкуса, чипы ароматов и блюд.
// Тексты блоков — фразы карточки из базы без изменений (сервер только раскладывает их по смыслу).
const WINE_TONE = { 'Красное': 'red', 'Белое': 'white', 'Розовое': 'rose', 'Оранжевое': 'orange' };

function chips(items, className) {
    const box = el('div', 'wd-chips');
    items.forEach(text => box.append(el('span', `wd-chip ${className}`, text)));
    return box;
}

function renderProfile(node, profile) {
    const sections = profile.sections || [];
    const scales = (profile.scales || []).filter(s => s.value !== null && s.value !== undefined);
    if (!sections.length && !scales.length) return;  // нечего структурировать — остаётся исходный текст
    const tone = WINE_TONE[profile.category] || 'neutral';
    const parts = [];

    if (scales.length) {
        const box = el('div', `wd-scales tone-${tone}`);
        scales.forEach(s => {
            const row = el('div', `wd-scale scale-${s.key}`);
            const head = el('div', 'wd-scale-head');
            head.append(el('span', 'wd-scale-label', s.label), el('span', 'wd-scale-word', s.word || ''));
            const track = el('div', 'wd-track');
            const dot = el('span', 'wd-dot');
            dot.style.left = `${Math.round(s.value * 100)}%`;
            track.append(dot);
            const ends = el('div', 'wd-scale-ends');
            ends.append(el('span', '', s.left), el('span', '', s.right));
            row.append(head, track, ends);
            box.append(row);
        });
        parts.push(box);
    }

    sections.forEach(section => {
        const block = el('div', `wd-block wd-${section.key}${section.key === 'color' ? ` tone-${tone}` : ''}`);
        block.append(el('div', 'wd-title', section.title), el('p', 'wd-text', section.text));
        if (section.key === 'aroma' && profile.aromas && profile.aromas.length) block.append(chips(profile.aromas, 'aroma'));
        if (section.key === 'food' && profile.foods && profile.foods.length) block.append(chips(profile.foods, 'food'));
        parts.push(block);
    });
    // ароматы/блюда, распознанные в карточке, но без своего блока в тексте
    if (profile.aromas && profile.aromas.length && !sections.some(s => s.key === 'aroma')) {
        const block = el('div', 'wd-block wd-aroma');
        block.append(el('div', 'wd-title', 'Ароматы'), chips(profile.aromas, 'aroma'));
        parts.push(block);
    }
    if (profile.foods && profile.foods.length && !sections.some(s => s.key === 'food')) {
        const block = el('div', 'wd-block wd-food');
        block.append(el('div', 'wd-title', 'С чем сочетается по карточке'), chips(profile.foods, 'food'));
        parts.push(block);
    }

    if (profile.facts && profile.facts.length) {
        const facts = el('div', 'wd-facts');
        profile.facts.forEach(f => {
            const item = el('div', 'wd-fact');
            item.append(el('span', 'wd-fact-label', f.label), el('span', 'wd-fact-value', f.value));
            facts.append(item);
        });
        parts.push(facts);
    }

    node.replaceChildren(...parts);
    node.classList.add('structured');
    node.classList.remove('hidden');
}

// ---------- Сомелье ----------
async function pairWith(dish) {
    const answer = $('pairing-answer');
    answer.classList.remove('pairing-graphic');
    if (!current || !current.slug) {
        answer.textContent = 'Для этого вина нет карточки сомелье.';
        return;
    }
    answer.textContent = 'Сомелье оценивает сочетание…';
    const id = request;
    try {
        const result = await getJson(api(`/api/sommelier/wine/${encodeURIComponent(current.slug)}?dish=${encodeURIComponent(dish)}`));
        if (id !== request) return;
        if (result.kind !== 'wine_info') {
            answer.textContent = result.text;
            return;
        }
        if (result.pairing) {
            renderPairing(answer, result.pairing);
            return;
        }
        const start = result.text.indexOf('К этому блюду');  // справка о вине доступна по кнопке «О вине»
        answer.textContent = start >= 0 ? result.text.slice(start) : result.text;
    } catch (error) {
        answer.textContent = `Сомелье недоступен: ${error.message}`;
    }
}

// Инфографика сочетания «вино + блюдо»: кольцо с баллом и вердиктом, место среди вин каталога,
// доводы «работает / мешает» с силой вклада правила, блюда, к которым вино подходит лучше.
const SVG_NS = 'http://www.w3.org/2000/svg';

function scoreRing(score, level) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 80 80');
    svg.setAttribute('class', `pr-ring level-${level}`);
    const circle = (cls, extra) => {
        const c = document.createElementNS(SVG_NS, 'circle');
        Object.entries({ cx: 40, cy: 40, r: 34, class: cls, ...extra }).forEach(([k, v]) => c.setAttribute(k, v));
        return c;
    };
    const length = 2 * Math.PI * 34;
    svg.append(circle('pr-ring-bg'), circle('pr-ring-fg', {
        'stroke-dasharray': `${(length * Math.max(0, Math.min(100, score)) / 100).toFixed(1)} ${length.toFixed(1)}`,
        transform: 'rotate(-90 40 40)',
    }));
    const text = document.createElementNS(SVG_NS, 'text');
    Object.entries({ x: 40, y: 46, 'text-anchor': 'middle', class: 'pr-ring-text' }).forEach(([k, v]) => text.setAttribute(k, v));
    text.textContent = Math.round(score);
    svg.append(text);
    return svg;
}

function factorRow(f) {
    const row = el('div', `pr-factor ${f.sign}`);
    const head = el('div', 'pr-factor-head');
    head.append(el('span', 'pr-factor-icon', f.sign === 'plus' ? '+' : '−'), el('span', 'pr-factor-title', f.title));
    // сила довода: вклад правила (−1…+1) с учётом его веса; самый весомый — «Сладость», 3,5
    const meter = el('span', 'pr-meter');
    const fill = el('span', 'pr-meter-fill');
    fill.style.width = `${Math.round(Math.min(1, Math.abs(f.value) * f.weight / 3.5) * 100)}%`;
    meter.append(fill);
    head.append(meter);
    row.append(head, el('div', 'pr-factor-text', f.text));
    if (f.quote) row.append(el('div', 'pr-quote', `«${f.quote}»`));
    return row;
}

function renderPairing(node, p) {
    const parts = [];

    const hero = el('div', `pr-hero level-${p.verdict.level}`);
    const summary = el('div', 'pr-summary');
    summary.append(el('span', 'pr-verdict', p.verdict.title), el('div', 'pr-dish', p.dish.text));
    if (p.dish.traits && p.dish.traits.length) summary.append(chips(p.dish.traits, 'trait'));
    hero.append(scoreRing(p.score, p.verdict.level), summary);
    parts.push(hero);

    // место среди всех вин каталога к этому блюду
    const rank = el('div', 'pr-rank');
    rank.append(el('div', 'pr-rank-label', `Лучше, чем ${Math.round(p.beats * 100)} % вин каталога к этому блюду`));
    const track = el('div', 'pr-rank-track');
    const marker = el('span', 'pr-rank-marker');
    marker.style.left = `${Math.round(p.beats * 100)}%`;
    track.append(marker);
    const ends = el('div', 'wd-scale-ends');
    ends.append(el('span', '', 'хуже'), el('span', '', `лучшее в каталоге — ${Math.round(p.best)}`));
    rank.append(track, ends);
    parts.push(rank);

    const groups = [['plus', 'Что работает'], ['minus', 'Что мешает']];
    if (p.verdict.level === 'poor') groups.reverse();  // при плохой паре сначала причина
    groups.forEach(([sign, title]) => {
        const items = p.factors.filter(f => f.sign === sign);
        if (!items.length) return;
        const box = el('div', `pr-group pr-${sign}`);
        box.append(el('div', 'pr-group-title', title), ...items.map(factorRow));
        parts.push(box);
    });

    if (p.unknown && p.unknown.length) {
        const box = el('div', 'pr-unknown');
        box.append(el('span', 'pr-unknown-label', 'В базе не указано:'), chips(p.unknown, 'unknown'));
        parts.push(box);
    }

    if (p.better_dishes && p.better_dishes.length) {
        const box = el('div', 'pr-better');
        box.append(el('div', 'pr-group-title', 'С чем это вино раскроется лучше'));
        const list = el('div', 'pr-better-list');
        p.better_dishes.forEach(d => {
            const btn = el('button', 'pr-dish-btn');
            btn.type = 'button';
            btn.append(el('span', '', d.label), el('b', '', String(Math.round(d.score))));
            btn.title = 'Оценить это сочетание';
            btn.addEventListener('click', () => {
                document.querySelectorAll('.btn-pairing').forEach(other => other.classList.remove('selected'));
                $('dish-input').value = d.query;
                pairWith(d.query);
            });
            list.append(btn);
        });
        box.append(list);
        if (p.card_foods && p.card_foods.length) {
            const card = el('div', 'pr-card-foods');
            card.append(el('span', 'pr-unknown-label', 'Карточка вина советует:'), chips(p.card_foods, 'food'));
            box.append(card);
        }
        parts.push(box);
    }

    if (p.note) parts.push(el('div', 'pr-note', p.note));
    node.replaceChildren(...parts);
    node.classList.add('pairing-graphic');
}

document.querySelectorAll('.btn-pairing').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-pairing').forEach(other => other.classList.toggle('selected', other === btn));
    pairWith(btn.dataset.dish);
}));
$('dish-form').addEventListener('submit', event => {
    event.preventDefault();
    const dish = $('dish-input').value.trim();
    if (!dish) return;
    document.querySelectorAll('.btn-pairing').forEach(btn => btn.classList.remove('selected'));
    pairWith(dish);
});

$('btn-about').addEventListener('click', () => {
    const box = $('about-answer');
    box.classList.toggle('hidden');
    if (box.classList.contains('hidden')) return;
    box.classList.remove('pairing-graphic');
    if (!current || !current.info) box.textContent = 'В карточках сомелье этого вина нет.';
    else if (current.info.profile) renderAbout(box, current.info);
    else box.textContent = current.info.text;
});

// «О вине»: паспорт вина одним взглядом. Описание, шкалы и факты подачи уже показаны в карточке выше (renderProfile),
// здесь — сводка плитками, вкус и ароматы плашками, с чем подать (по нажатию — оценка сочетания), чего нет в базе.
function renderAbout(node, info) {
    const card = info.card || {};
    const profile = info.profile || {};
    const scales = profile.scales || [];
    const sweet = scales.find(s => s.key === 'sweetness');
    const parts = [];

    const hero = el('div', `ab-hero tone-${WINE_TONE[profile.category] || 'neutral'}`);
    hero.append(el('div', 'pr-group-title', 'Паспорт вина'), el('div', 'pr-dish', card.name || ''));
    if (card.winery) hero.append(el('div', 'rc-winery', card.winery));
    parts.push(hero);

    const type = [card.category && card.category.toLowerCase(), profile.sparkling ? 'игристое' : '', sweet && sweet.word].filter(Boolean).join(', ');
    const tiles = [['Тип', type], ['Сорт', card.grape], ['Регион', card.region],
        ...(profile.facts || []).map(f => [f.label, f.value])].filter(([, value]) => value);
    if (tiles.length) {
        const grid = el('div', 'ab-tiles');
        tiles.forEach(([label, value]) => {
            const tile = el('div', 'wd-fact');
            tile.append(el('span', 'wd-fact-label', label), el('span', 'wd-fact-value', value));
            grid.append(tile);
        });
        parts.push(grid);
    }

    const taste = scales.filter(s => s.key !== 'sweetness' && s.word && s.value !== null);
    if (taste.length || (profile.aromas || []).length) {
        const box = el('div', 'pr-group ab-taste');
        box.append(el('div', 'pr-group-title', 'Вкус и аромат'));
        if (taste.length) box.append(chips(taste.map(s => `${s.label}: ${s.word}`), 'trait'));
        if ((profile.aromas || []).length) box.append(chips(profile.aromas, 'aroma'));
        parts.push(box);
    }

    if ((info.best_dishes || []).length || (profile.foods || []).length) {
        const box = el('div', 'pr-better');
        box.append(el('div', 'pr-group-title', 'С чем подать'));
        if ((profile.foods || []).length) {
            const foods = el('div', 'pr-card-foods ab-foods');
            foods.append(el('span', 'pr-unknown-label', 'Советует карточка вина:'), chips(profile.foods, 'food'));
            box.append(foods);
        }
        if ((info.best_dishes || []).length) {
            box.append(el('div', 'pr-unknown-label ab-hint', 'Лучше всего по правилам сомелье — нажмите, чтобы оценить сочетание:'));
            const list = el('div', 'pr-better-list');
            info.best_dishes.forEach(d => {
                const btn = el('button', 'pr-dish-btn');
                btn.type = 'button';
                btn.append(el('span', '', d.label), el('b', '', String(Math.round(d.score))));
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.btn-pairing').forEach(other => other.classList.remove('selected'));
                    $('dish-input').value = d.query;
                    pairWith(d.query);
                    $('pairing-answer').scrollIntoView({ behavior: 'smooth', block: 'start' });
                });
                list.append(btn);
            });
            box.append(list);
        }
        parts.push(box);
    }

    const missing = scales.filter(s => s.value === null || s.value === undefined).map(s => s.label.toLowerCase());
    if (missing.length) {
        const box = el('div', 'pr-unknown');
        box.append(el('span', 'pr-unknown-label', 'В базе не указано:'), chips(missing, 'unknown'));
        parts.push(box);
    }
    if (card.url && /^https?:\/\//.test(card.url)) {
        const link = el('a', 'rc-link', 'Карточка вина на сайте →');
        link.href = card.url;
        link.target = '_blank';
        link.rel = 'noopener';
        parts.push(link);
    }
    if (info.note) parts.push(el('div', 'pr-note', info.note));

    node.replaceChildren(...parts);
    node.classList.add('pairing-graphic');
}

$('btn-chat').addEventListener('click', () => {
    $('chat-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('chat-input').focus({ preventScroll: true });
});

$('chat-form').addEventListener('submit', async event => {
    event.preventDefault();
    const message = $('chat-input').value.trim();
    if (!message) return;
    const answer = $('chat-answer');
    answer.classList.remove('pairing-graphic');
    answer.textContent = 'Сомелье думает…';
    try {
        const data = await getJson(api('/api/sommelier/ask'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sommelierSession ? { message, session_id: sommelierSession } : { message }),
        });
        sommelierSession = data.session_id;
        if (data.kind === 'recommendation' && data.picks && data.picks.length) {
            renderRecommendation(answer, data);
        } else {
            answer.classList.remove('pairing-graphic');
            answer.textContent = data.text;
        }
    } catch (error) {
        answer.classList.remove('pairing-graphic');
        answer.textContent = `Ошибка: ${error.message}`;
    }
});

// Подбор вин в чате: карточки с фото бутылки, ролью варианта, совпадением и доводами; уместность типов — плашками.
// Полный текст ответа (с LLM — живой пересказ) остаётся под «Пояснением сомелье».
const ROLE_TONE = { 'Лучшее совпадение по правилам': 'best', 'Классический вариант': 'classic', 'Нестандартный выбор': 'rare' };
const TYPE_VERDICTS = [['подходит', 'fit', 'Подходит'], ['допустимо, с оговорками', 'maybe', 'С оговорками'], ['лучше избегать', 'avoid', 'Лучше избегать']];

function pickCard(pick, index) {
    const card = el('article', `rc-card role-${ROLE_TONE[pick.role] || 'other'}`);
    const tone = WINE_TONE[pick.category] || 'neutral';

    const media = el('div', `rc-media tone-${tone}`);
    const image = document.createElement('img');
    image.alt = pick.name;
    image.loading = 'lazy';
    image.src = pick.image_url ? api(pick.image_url) : 'img/glass_filled_straight.png';
    if (!pick.image_url) media.classList.add('placeholder');
    image.onerror = () => { media.classList.add('placeholder'); image.onerror = null; image.src = 'img/glass_filled_straight.png'; };
    media.append(image, el('span', 'rc-num', String(index + 1)));

    const body = el('div', 'rc-body');
    body.append(el('span', 'rc-role', pick.role), el('div', 'rc-name', pick.name));
    if (pick.winery) body.append(el('div', 'rc-winery', pick.winery));
    const type = [pick.category && pick.category.toLowerCase(), pick.sparkling ? 'игристое' : '', pick.sweetness].filter(Boolean).join(', ');
    body.append(chips([type, pick.grape, pick.region].filter(Boolean), 'trait'));

    const match = el('div', 'rc-match');
    const meter = el('span', 'pr-meter');
    const fill = el('span', 'pr-meter-fill');
    fill.style.width = `${Math.max(0, Math.min(100, Math.round(pick.score)))}%`;
    meter.append(fill);
    match.append(el('span', 'rc-match-label', 'Совпадение'), meter, el('b', '', String(Math.round(pick.score))));
    body.append(match);

    const top = el('div', 'rc-top');
    top.append(media, body);
    card.append(top);

    const reasons = el('div', 'rc-reasons');
    (pick.pros || []).forEach(text => reasons.append(el('div', 'rc-reason plus', text)));
    (pick.cons || []).forEach(text => reasons.append(el('div', 'rc-reason minus', text)));
    if (reasons.children.length) card.append(reasons);
    if (pick.unknown && pick.unknown.length) {
        const box = el('div', 'pr-unknown');
        box.append(el('span', 'pr-unknown-label', 'В базе не указано:'), chips(pick.unknown, 'unknown'));
        card.append(box);
    }
    if (pick.quote) card.append(el('div', 'pr-quote rc-quote', `«${pick.quote}»`));
    if (pick.url && /^https?:\/\//.test(pick.url)) {
        const link = el('a', 'rc-link', 'Карточка вина на сайте →');
        link.href = pick.url;
        link.target = '_blank';
        link.rel = 'noopener';
        card.append(link);
    }
    return card;
}

function renderRecommendation(node, data) {
    const parts = [];
    const head = el('div', 'rc-head');
    const dish = data.context && data.context.dish_text;
    head.append(el('div', 'pr-group-title', dish ? 'Подбор к блюду' : 'Подбор вина'));
    if (dish) head.append(el('div', 'pr-dish', dish));
    if (data.dish_traits && data.dish_traits.length) head.append(chips(data.dish_traits, 'trait'));
    if (data.total) head.append(el('div', 'rc-pool', data.pool === data.total
        ? `Выбирал из всех ${data.total} вин каталога. Три варианта разных стилей:`
        : `Из ${data.total} вин каталога под ваши условия подходят ${data.pool}. Три варианта разных стилей:`));
    parts.push(head, ...data.picks.map(pickCard));

    if (data.types && data.types.length) {
        const box = el('div', 'pr-group rc-types');
        box.append(el('div', 'pr-group-title', 'Какие типы вина уместны'));
        TYPE_VERDICTS.forEach(([verdict, cls, title]) => {
            const rows = data.types.slice(0, 8).filter(t => t.verdict === verdict);
            if (!rows.length) return;
            const line = el('div', `rc-type-row ${cls}`);
            line.append(el('span', 'rc-type-verdict', title),
                chips(rows.map(t => [t.cat.toLowerCase(), t.sparkling ? 'игристое' : '', t.sweet].filter(Boolean).join(' ')), `type-${cls}`));
            box.append(line);
        });
        parts.push(box);
    }

    const details = document.createElement('details');
    details.className = 'rc-details';
    details.append(el('summary', '', 'Пояснение сомелье текстом'), el('div', 'somm-text', data.note ? data.text.replace(data.note, '').trim() : data.text));
    parts.push(details);
    if (data.note) parts.push(el('div', 'pr-note', data.note));

    node.replaceChildren(...parts);
    node.classList.add('pairing-graphic');
}

// ---------- Оценка бокалами (хранится на устройстве) ----------
function initRating(key) {
    const storageKey = `vina_rating_${key}`;
    let rating = parseInt(storage.get(storageKey) || '0', 10) || 0;
    const slots = document.querySelectorAll('.glass-slot');
    const paint = hover => {
        const value = hover || rating;
        slots.forEach(slot => {
            const active = Number(slot.dataset.index) <= value;
            slot.classList.toggle('active-bg', active);
            slot.querySelector('.glass-img').src = active ? 'img/glass_filled_straight.png' : 'img/glass_empty_straight.png';
        });
        $('badge-score').textContent = rating ? `${rating}/5` : '-';
        $('user-rating-text').textContent = rating ? `Ваша оценка: ${rating} из 5` : 'Ваша оценка: - из 5';
    };
    slots.forEach(slot => {
        const value = Number(slot.dataset.index);
        slot.onmouseenter = () => paint(value);
        slot.onmouseleave = () => paint(0);
        slot.onclick = () => {
            rating = value;
            storage.set(storageKey, String(rating));
            paint(0);
        };
    });
    paint(0);
}

$('people-rating-badge').addEventListener('click', () => $('rating-section').scrollIntoView({ behavior: 'smooth' }));

// ---------- Проверка набора фото: каждое фото по очереди проходит тот же каскад ----------
const IMAGE_FILE = /\.(jpe?g|png|webp)$/i;
let batchItems = [];
let batchRunning = false;
let batchStop = false;

async function runBatch(fileList) {
    const files = [...fileList]
        .filter(file => IMAGE_FILE.test(file.name) && !file.name.startsWith('.') && !(file.webkitRelativePath || '').includes('__MACOSX'))
        .sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
    if (!files.length) {
        scannerHint.textContent = 'В выбранном наборе нет фото JPG / PNG / WEBP.';
        return;
    }
    if (batchRunning || busy) return;
    batchItems.forEach(item => URL.revokeObjectURL(item.url));
    batchItems = files.map((file, index) => ({
        file,
        index,
        name: file.webkitRelativePath || file.name,
        url: URL.createObjectURL(file),
        state: 'pending',
    }));
    batchRunning = true;
    batchStop = false;
    $('btn-batch-stop').classList.remove('hidden');
    showView('batch');
    renderBatch();
    for (const item of batchItems) {
        if (batchStop) break;
        item.state = 'running';
        renderBatch();
        const started = performance.now();
        try {
            item.data = await cascadeSearch(item.file);
            item.state = 'done';
        } catch (error) {
            item.state = 'error';
            item.error = error.message;
        }
        item.ms = Math.round(performance.now() - started);
        renderBatch();
    }
    batchItems.filter(item => item.state === 'pending').forEach(item => { item.state = 'error'; item.error = 'остановлено'; });
    batchRunning = false;
    $('btn-batch-stop').classList.add('hidden');
    renderBatch();
}

function renderBatch() {
    const done = batchItems.filter(item => item.state === 'done' || item.state === 'error');
    const answered = batchItems.filter(item => item.state === 'done');
    const count = status => answered.filter(item => item.data.status === status).length;
    const times = answered.map(item => item.ms).sort((a, b) => a - b);
    $('batch-bar').style.width = `${batchItems.length ? (100 * done.length) / batchItems.length : 0}%`;
    $('batch-summary').textContent = [
        `Проверено ${done.length} из ${batchItems.length}${batchRunning ? '…' : ''}`,
        `найдено ${count('found')}, похоже ${count('probable')}, нет в каталоге ${count('not_in_catalog')}`,
        `ошибок ${done.length - answered.length}`,
        times.length ? `медиана ${times[Math.floor(times.length / 2)]} мс` : '',
    ].filter(Boolean).join(' · ');

    $('batch-list').replaceChildren(...batchItems.map(item => {
        const row = el('div', 'list-item batch-row');
        const img = el('img');
        img.src = item.url;
        img.alt = '';
        img.loading = 'lazy';
        const body = el('div');
        if (item.state === 'done') {
            const { data } = item;
            const winner = data.winner;
            body.append(
                el('span', `st ${data.status}`, (STATUS[data.status] || STATUS.probable).title),
                el('div', 'item-title', winner ? winner.title : 'Вина нет в каталоге'),
            );
            if (winner && winner.slug) body.append(el('div', 'slug', winner.slug));
            body.append(el('div', 'item-meta', `${item.name} · уверенность ${fmt(data.confidence)} · ${item.ms} мс`));
            row.addEventListener('click', () => openBatchItem(item));
        } else {
            const text = item.state === 'error' ? `Ошибка: ${item.error}` : (item.state === 'running' ? 'Распознаю…' : 'Ожидает');
            body.append(el('span', `st ${item.state === 'error' ? 'error' : ''}`, text), el('div', 'item-meta', item.name));
        }
        row.append(img, body);
        return row;
    }));
}

function openBatchItem(item) {
    returnTo = 'batch';
    $('btn-back-label').textContent = 'Назад к набору';
    renderResult(item.data, ++request, item.url);
    showView('result');
}

function download(filename, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = Object.assign(document.createElement('a'), { href: url, download: filename });
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

$('batch-files-input').addEventListener('change', event => { runBatch(event.target.files); event.target.value = ''; });
$('batch-folder-input').addEventListener('change', event => { runBatch(event.target.files); event.target.value = ''; });
$('btn-batch-stop').addEventListener('click', () => { batchStop = true; });
$('btn-batch-back').addEventListener('click', () => showView('scanner'));
$('btn-batch-tsv').addEventListener('click', () => {
    const clean = value => String(value ?? '').replace(/[\t\r\n]+/g, ' ');
    const header = ['image_path', 'status', 'slug', 'title', 'manufacturer', 'confidence', 'latency_ms', 'error'];
    const lines = batchItems.map(item => {
        const winner = item.data && item.data.winner;
        return [item.name, item.data ? item.data.status : '', winner ? winner.slug : '', winner ? winner.title : '', winner ? winner.manufacturer : '',
            item.data && typeof item.data.confidence === 'number' ? item.data.confidence.toFixed(4) : '', item.ms ?? '', item.error || ''].map(clean).join('\t');
    });
    download('results.tsv', [header.join('\t'), ...lines].join('\n') + '\n', 'text/tab-separated-values');
});
$('btn-batch-jsonl').addEventListener('click', () => {
    const lines = batchItems.filter(item => item.state === 'done' || item.state === 'error').map(item => JSON.stringify({
        query_id: `q-${String(item.index + 1).padStart(6, '0')}`,
        image_path: item.file.name,
        predicted_slug: item.data && item.data.winner ? item.data.winner.slug : null,
        latency_ms: item.ms ?? null,
    }));
    download('predictions.jsonl', lines.join('\n') + '\n', 'application/x-ndjson');
});

// ---------- Камера ----------
function frameBlob() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

async function loop() {
    if (video.videoWidth && !busy && views.scanner.classList.contains('active')) {
        try {
            const box = await detect(video);
            draw(overlay, box, video, 'cover');
            btnSnap.classList.toggle('detected', Boolean(box));
        } catch (error) {
            console.warn(error);
        }
    }
    requestAnimationFrame(loop);
}

async function start() {
    if (!window.isSecureContext) {
        scannerHint.textContent = 'Камера работает только по HTTPS — загрузите фото кнопкой «Выбрать фото».';
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        scannerHint.textContent = 'Браузер не даёт доступ к камере — загрузите фото кнопкой «Выбрать фото».';
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } } });
        video.srcObject = stream;
        video.style.display = 'block';
        await video.play();
        $('scan-instruction').classList.add('hidden');
        $('scanning-line').classList.remove('hidden');
        btnSnap.classList.remove('hidden');
        scannerHint.textContent = 'Наведите камеру на этикетку и нажмите на круглую кнопку.';
        if (window.ort && await loadDetector()) loop();
    } catch (error) {
        scannerHint.textContent = `Камера недоступна (${error.message}) — загрузите фото кнопкой «Выбрать фото».`;
    }
}

btnSnap.addEventListener('click', async () => {
    if (!video.videoWidth) return;
    submit(await frameBlob());
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) submit(fileInput.files[0]);
    fileInput.value = '';
});
$('btn-back').addEventListener('click', () => { scannerHint.textContent = ''; showView(returnTo); });
$('btn-show-original').addEventListener('click', () => showView('original'));
$('btn-back-to-result').addEventListener('click', () => showView('result'));

// ---------- Уведомления ----------
let toastTimeout = null;
function showToast(message) {
    const toast = $('toast');
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.remove('show'), 2500);
}

// ---------- Просмотр изображений ----------
const lightboxModal = $('lightbox-modal');
const lightboxImg = $('lightbox-img');
const lightboxCanvas = $('lightbox-canvas');
let zoomLevel = 1;
let dragging = false;
let startX = 0;
let startY = 0;
let translateX = 0;
let translateY = 0;

function updateLightboxTransform() {
    lightboxImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${zoomLevel})`;
}

function openLightbox(src, title) {
    if (!src) return;
    lightboxImg.src = src;
    $('lightbox-title').textContent = title || 'Просмотр изображения';
    zoomLevel = 1;
    translateX = 0;
    translateY = 0;
    updateLightboxTransform();
    lightboxModal.classList.remove('hidden');
}

function closeLightbox() {
    lightboxModal.classList.add('hidden');
    lightboxImg.src = '';
}

$('wine-image').addEventListener('click', event => openLightbox(event.target.src, $('wine-title').textContent));
$('original-img').addEventListener('click', event => openLightbox(event.target.src, 'Исходное фото'));
$('crop-img').addEventListener('click', event => openLightbox(event.target.src, 'Кроп этикетки'));
$('btn-close-lightbox').addEventListener('click', closeLightbox);
lightboxModal.addEventListener('click', event => { if (event.target === lightboxModal || event.target === lightboxCanvas) closeLightbox(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLightbox(); });
$('btn-zoom-in').addEventListener('click', () => { zoomLevel += 0.25; updateLightboxTransform(); });
$('btn-zoom-out').addEventListener('click', () => { zoomLevel = Math.max(0.25, zoomLevel - 0.25); updateLightboxTransform(); });
$('btn-zoom-reset').addEventListener('click', () => { zoomLevel = 1; translateX = 0; translateY = 0; updateLightboxTransform(); });
lightboxCanvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoomLevel = Math.min(8, Math.max(0.25, zoomLevel + (event.deltaY < 0 ? 0.25 : -0.25)));
    updateLightboxTransform();
}, { passive: false });
const dragStart = (x, y) => { dragging = true; startX = x - translateX; startY = y - translateY; };
const dragMove = (x, y) => { if (!dragging) return; translateX = x - startX; translateY = y - startY; updateLightboxTransform(); };
lightboxCanvas.addEventListener('mousedown', event => dragStart(event.clientX, event.clientY));
lightboxCanvas.addEventListener('mousemove', event => dragMove(event.clientX, event.clientY));
['mouseup', 'mouseleave', 'touchend'].forEach(name => lightboxCanvas.addEventListener(name, () => { dragging = false; }));
lightboxCanvas.addEventListener('touchstart', event => { if (event.touches.length === 1) dragStart(event.touches[0].clientX, event.touches[0].clientY); }, { passive: true });
lightboxCanvas.addEventListener('touchmove', event => { if (event.touches.length === 1) dragMove(event.touches[0].clientX, event.touches[0].clientY); }, { passive: true });

// ---------- Адрес сервера ----------
const settingsModal = $('settings-modal');
const inputApiUrl = $('input-api-url');
const feedbackBox = $('connection-feedback');

function showFeedback(text, type) {
    feedbackBox.textContent = text;
    feedbackBox.className = `feedback-box ${type}`;
}

function closeSettings() { settingsModal.classList.add('hidden'); }

// Ссылки на страницы сервера (каскадный поиск, добавление вина) ведут на выбранный сервер API
function updateServerLinks() {
    $('link-cascade').href = api('/search-cascade');
    $('link-add').href = config.apiBase ? api('/mobi/add.html') : 'add.html';
}

function applyApiBase(value) {
    if (value) storage.set('vina_api_url', value); else storage.remove('vina_api_url');
    config.apiBase = value;
    updateServerLinks();
    closeSettings();
    checkServerHealth();
    loadThresholds();
    showToast(value ? `Сервер: ${value}` : 'Сервер: этот сайт');
}

$('btn-settings').addEventListener('click', () => {
    inputApiUrl.value = storage.get('vina_api_url') || '';
    feedbackBox.className = 'feedback-box hidden';
    settingsModal.classList.remove('hidden');
});
$('btn-close-modal').addEventListener('click', closeSettings);
document.querySelector('.modal-backdrop').addEventListener('click', closeSettings);
$('btn-reset-local').addEventListener('click', () => applyApiBase(''));
$('btn-save-settings').addEventListener('click', () => applyApiBase(inputApiUrl.value.trim().replace(/\/+$/, '')));
$('btn-test-connection').addEventListener('click', async () => {
    const base = inputApiUrl.value.trim().replace(/\/+$/, '');
    showFeedback('Проверка соединения…', '');
    try {
        const response = await fetch(`${base}/api/ready`);
        showFeedback(response.ok ? 'Сервер доступен и готов к распознаванию.' : `Сервер ответил HTTP ${response.status}.`, response.ok ? 'success' : 'error');
    } catch {
        showFeedback('Не удалось связаться с сервером.', 'error');
    }
});

async function checkServerHealth() {
    const dot = document.querySelector('.status-dot');
    const text = $('status-text');
    dot.className = 'status-dot';
    text.textContent = 'Проверка…';
    try {
        const response = await fetch(api('/api/ready'));
        dot.className = `status-dot ${response.ok ? 'online' : 'offline'}`;
        text.textContent = response.ok ? 'Онлайн' : 'Не готов';
    } catch {
        dot.className = 'status-dot offline';
        text.textContent = 'Офлайн';
    }
}

updateServerLinks();
checkServerHealth();
loadThresholds();
start();
