// export/js/cascade_mobi.js
// Mobile interactive camera scanner with Cascade (v1 + v4) recognition and AI-Sommelier link.

import { loadDetector, isDetectorReady, detect, mapBoxesToElement, drawBoxes } from './detector.js';

const API_BASE = (typeof window !== 'undefined' && window.MOBI_API_BASE) ? window.MOBI_API_BASE : '';
const CASCADE_SEARCH_URL = `${API_BASE}/api/cascade/search`;

let video = null;
let overlayCanvas = null;
let freezeCanvas = null;
let cameraWrap = null;
let shutterBtn = null;
let resumeOverlay = null;
let statusMsg = null;
let resultSection = null;
let heroCard = null;
let scanAgainBtn = null;

let isDetecting = true;
let isSearching = false;
let stream = null;
let lastBoxes = [];
let animId = null;

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

export async function initCascadeMobile(modelUrl = 'models/yolov8_label.onnx') {
  video = document.getElementById('camera-video');
  overlayCanvas = document.getElementById('overlay-canvas');
  freezeCanvas = document.getElementById('freeze-canvas');
  cameraWrap = document.getElementById('camera-wrap');
  shutterBtn = document.getElementById('camera-shutter');
  resumeOverlay = document.getElementById('resume-overlay');
  statusMsg = document.getElementById('camera-status');
  resultSection = document.getElementById('result-section');
  heroCard = document.getElementById('hero-card');
  scanAgainBtn = document.getElementById('scan-again');

  if (!video || !overlayCanvas || !shutterBtn) {
    console.error('Missing required mobile DOM elements');
    return;
  }

  statusMsg.textContent = 'Загрузка детектора этикеток (ONNX Web)...';
  const ready = await loadDetector(modelUrl);
  if (ready) {
    statusMsg.textContent = 'Детектор готов. Подключение камеры...';
  } else {
    statusMsg.textContent = 'Ошибка загрузки детектора, доступен ручной снимок.';
  }

  await startCamera();

  shutterBtn.addEventListener('click', onShutterClick);
  if (resumeOverlay) {
    resumeOverlay.addEventListener('click', resumeScanning);
  }
  if (scanAgainBtn) {
    scanAgainBtn.addEventListener('click', resumeScanning);
  }

  startDetectionLoop();
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    statusMsg.textContent = 'Наведите камеру на этикетку вина';
  } catch (err) {
    console.error('Camera access error:', err);
    statusMsg.textContent = 'Ошибка доступа к камере: ' + err.message;
  }
}

function startDetectionLoop() {
  async function frame() {
    if (isDetecting && !isSearching && video.readyState >= 2) {
      if (overlayCanvas.width !== video.clientWidth || overlayCanvas.height !== video.clientHeight) {
        overlayCanvas.width = video.clientWidth;
        overlayCanvas.height = video.clientHeight;
      }

      if (isDetectorReady()) {
        try {
          const rawBoxes = await detect(video);
          lastBoxes = mapBoxesToElement(rawBoxes, video);
          drawBoxes(overlayCanvas, lastBoxes);
        } catch (e) {
          console.warn('Detection frame error:', e);
        }
      }
    }
    animId = requestAnimationFrame(frame);
  }
  animId = requestAnimationFrame(frame);
}

async function onShutterClick() {
  if (isSearching) return;
  isSearching = true;
  isDetecting = false;

  freezePreview();
  statusMsg.textContent = 'Анализ финальным каскадом (v1 + v4)...';

  try {
    const blob = await captureFullFrameBlob();
    await performCascadeSearch(blob);
  } catch (err) {
    console.error('Search failed:', err);
    statusMsg.textContent = 'Ошибка распознавания: ' + err.message;
    unfreezePreview();
  } finally {
    isSearching = false;
  }
}

function freezePreview() {
  if (freezeCanvas && video) {
    freezeCanvas.width = video.clientWidth;
    freezeCanvas.height = video.clientHeight;
    const ctx = freezeCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, freezeCanvas.width, freezeCanvas.height);
    freezeCanvas.classList.remove('hidden');
    video.classList.add('hidden');
  }
  if (cameraWrap) {
    cameraWrap.classList.add('paused');
  }
  if (resumeOverlay) {
    resumeOverlay.classList.remove('hidden');
  }
}

function unfreezePreview() {
  if (freezeCanvas && video) {
    freezeCanvas.classList.add('hidden');
    video.classList.remove('hidden');
  }
  if (cameraWrap) {
    cameraWrap.classList.remove('paused');
  }
  if (resumeOverlay) {
    resumeOverlay.classList.add('hidden');
  }
  isDetecting = true;
  isSearching = false;
}

function resumeScanning() {
  unfreezePreview();
  if (resultSection) {
    resultSection.classList.add('hidden');
  }
  statusMsg.textContent = 'Наведите камеру на этикетку вина';
}

function captureFullFrameBlob() {
  return new Promise((resolve) => {
    const snap = document.createElement('canvas');
    snap.width = video.videoWidth || 1280;
    snap.height = video.videoHeight || 720;
    const ctx = snap.getContext('2d');
    ctx.drawImage(video, 0, 0, snap.width, snap.height);
    snap.toBlob((b) => resolve(b), 'image/jpeg', 0.92);
  });
}

async function performCascadeSearch(blob) {
  const formData = new FormData();
  formData.append('image', blob, 'camera_scan.jpg');

  const t0 = performance.now();
  const resp = await fetch(CASCADE_SEARCH_URL, {
    method: 'POST',
    body: formData
  });

  if (!resp.ok) {
    throw new Error(`API HTTP ${resp.status}`);
  }

  const data = await resp.json();
  const elapsed = Math.round(performance.now() - t0);
  renderResults(data, elapsed);
}

const MOBILE_STATUS = {
  found: { color: '#00e676', label: 'Найдено' },
  probable: { color: '#ffd740', label: 'Похоже — сверьте с этикеткой' },
  not_in_catalog: { color: '#ff5252', label: 'Нет в каталоге' },
};

function renderResults(res, clientElapsed) {
  if (!resultSection || !heroCard) return;

  resultSection.classList.remove('hidden');
  const winner = res.winner;
  const view = MOBILE_STATUS[res.status] || MOBILE_STATUS.probable;
  const confidence = typeof res.confidence === 'number' ? `${Math.round(res.confidence * 100)}%` : '—';

  if (!winner) {
    heroCard.innerHTML = `
      <div style="border: 2px solid ${view.color}; border-radius: 12px; padding: 14px; background: rgba(20,20,25,0.9); color: #fff;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <span style="background: ${view.color}; color: #000; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px;">${escapeHtml(view.label)}</span>
          <span style="color: #aaa; font-size: 0.8rem;">${clientElapsed} мс</span>
        </div>
        <h3 style="margin: 0 0 6px; font-size: 1.1rem;">${escapeHtml(res.message)}</h3>
        <div style="color: #bbb; font-size: 0.85rem;">Сходство с ближайшей этикеткой каталога: ${confidence}. Попробуйте снять этикетку ближе и без бликов.</div>
      </div>`;
    statusMsg.textContent = res.message;
    return;
  }

  heroCard.innerHTML = `
    <div style="border: 2px solid ${view.color}; border-radius: 12px; padding: 14px; background: rgba(20,20,25,0.9); color: #fff;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="background: ${view.color}; color: #000; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 12px;">${escapeHtml(view.label)}</span>
        <span style="color: #aaa; font-size: 0.8rem;">${clientElapsed} мс</span>
      </div>
      <div style="display: flex; gap: 12px; align-items: center;">
        <img src="${API_BASE}${winner.image_url}" alt="${escapeHtml(winner.title)}"
             style="width: 85px; height: 110px; object-fit: contain; border-radius: 6px; background: #fff; padding: 4px;"
             onclick="window.openMobileLightbox('${API_BASE}${winner.image_url}', '${escapeHtml(winner.title)}')">
        <div style="flex: 1;">
          <h3 style="margin: 0 0 4px; font-size: 1.1rem; line-height: 1.3;">${escapeHtml(winner.title)}</h3>
          <div style="color: #bbb; font-size: 0.85rem; margin-bottom: 6px;">${escapeHtml(winner.manufacturer)}</div>
          <div style="font-size: 0.8rem; color: #4fc3f7;">Сходство: <b>${confidence}</b></div>
        </div>
      </div>
      <div id="mobile-alternatives" style="margin-top: 12px; font-size: 0.85rem; color: #ccc;">AI-сомелье подбирает похожие вина…</div>
    </div>
  `;
  statusMsg.textContent = `${view.label}: ${winner.title} (${confidence})`;
  if (winner.slug) renderAlternatives(winner.slug);
}

async function renderAlternatives(slug) {
  const box = document.getElementById('mobile-alternatives');
  try {
    const resp = await fetch(`${API_BASE}/api/sommelier/wine/${encodeURIComponent(slug)}/alternatives?limit=3`);
    const data = resp.ok ? await resp.json() : { items: [] };
    if (!box) return;
    box.innerHTML = data.items.length
      ? '<div style="font-weight: 700; color: #fff; margin-bottom: 6px;">Похожие вина (AI-сомелье)</div>' + data.items.map(item => `
          <div style="margin-bottom: 6px;"><b>${escapeHtml(item.name)}</b> <span style="color: #999;">${escapeHtml(item.winery)}</span><br>
          <span style="font-size: 0.75rem; color: #b388ff;">${item.reasons.map(escapeHtml).join(' · ')}</span></div>`).join('')
      : '';
  } catch {
    if (box) box.textContent = '';
  }
}

// Lightbox helper
window.openMobileLightbox = function(src, caption) {
  const modal = document.getElementById('lightbox-modal');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  if (modal && img) {
    img.src = src;
    if (cap) cap.textContent = caption || '';
    modal.classList.remove('hidden');
  }
};

window.closeMobileLightbox = function(e) {
  if (e && e.target && e.target.tagName === 'IMG') return;
  const modal = document.getElementById('lightbox-modal');
  if (modal) modal.classList.add('hidden');
};
