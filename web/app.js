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
    }

    const items = alts.status === 'fulfilled' ? alts.value.items : [];
    alternatives.replaceChildren(...(items.length
        ? items.map(item => listItem(item.image_url, item.name, item.winery, item.facts, item.reasons.join(' · ')))
        : [el('div', 'item-meta', 'Похожих вин в каталоге сомелье не нашёл.')]));
}

// ---------- Сомелье ----------
async function pairWith(dish) {
    const answer = $('pairing-answer');
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
        const start = result.text.indexOf('К этому блюду');  // справка о вине доступна по кнопке «О вине»
        answer.textContent = start >= 0 ? result.text.slice(start) : result.text;
    } catch (error) {
        answer.textContent = `Сомелье недоступен: ${error.message}`;
    }
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
    box.textContent = current && current.info ? current.info.text : 'В карточках сомелье этого вина нет.';
    box.classList.toggle('hidden');
});

$('btn-chat').addEventListener('click', () => {
    $('chat-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('chat-input').focus({ preventScroll: true });
});

$('chat-form').addEventListener('submit', async event => {
    event.preventDefault();
    const message = $('chat-input').value.trim();
    if (!message) return;
    const answer = $('chat-answer');
    answer.textContent = 'Сомелье думает…';
    try {
        const data = await getJson(api('/api/sommelier/ask'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sommelierSession ? { message, session_id: sommelierSession } : { message }),
        });
        sommelierSession = data.session_id;
        answer.textContent = data.text;
    } catch (error) {
        answer.textContent = `Ошибка: ${error.message}`;
    }
});

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
