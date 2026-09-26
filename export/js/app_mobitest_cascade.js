// export/app_mobitest_v5m.js
// High-performance client-side camera detection, freeze-frame search, and instant resume (v5m).

import {
  loadDetector,
  isDetectorReady,
  detect,
  mapBoxesToElement,
  drawBoxes,
  drawBox,
  cropLabel,
  isPointInBox,
  getDetectorStats
} from './v5m/detector.js?v=1';

const API_BASE = (typeof window !== 'undefined' && window.MOBI_API_BASE) ? window.MOBI_API_BASE : '';
const FETCH_TIMEOUT = 60000;
const SIMILARITY_HIGH = 0.90;
const SIMILARITY_MED = 0.75;
const INFER_INTERVAL_MS = 60; // Max ~16 FPS detector check

function fetchWithTimeout(url, options, timeout = FETCH_TIMEOUT) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id));
}

function formatError(e) {
  if (e.name === 'AbortError') {
    return 'Время ожидания истекло. Проверьте скорость сети или настройку MOBI_API_BASE.';
  }
  if (e.name === 'TypeError') {
    return 'Не удалось связаться с API. Если страница открыта по HTTPS — откройте по HTTP. Также проверьте настройку MOBI_API_BASE (config.js) или reverse-proxy.';
  }
  return e.message;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function sendClientLog(level, message, extra) {
  const payload = {
    level,
    message,
    userAgent: navigator.userAgent,
    location: window.location.href,
    apiBase: API_BASE,
    ...(extra || {}),
  };
  try {
    fetch(`${API_BASE}/api/client-log`, {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {});
  } catch (e) {}
}

function confidenceClass(sim) {
  if (sim >= SIMILARITY_HIGH) return 'high';
  if (sim >= SIMILARITY_MED) return 'medium';
  return 'low';
}

function confidenceText(sim) {
  if (sim >= SIMILARITY_HIGH) return 'Точно распознано';
  if (sim >= SIMILARITY_MED) return 'Возможно совпадение';
  return 'Совпадение неясно — попробуйте переснять';
}

function initCascadeMobile(modelUrl = 'models/yolov8_label.onnx') {
  const video = document.getElementById('camera-video');
  const shutter = document.getElementById('camera-shutter');
  const cameraWrap = document.getElementById('camera-wrap');
  const cameraStatus = document.getElementById('camera-status');
  const permissionMsg = document.getElementById('camera-permission');
  const resultSection = document.getElementById('result-section');
  const resultStatus = document.getElementById('result-status');
  const scanAgain = document.getElementById('scan-again');
  const overlay = document.getElementById('overlay-canvas');
  const freezeCanvas = document.getElementById('freeze-canvas');
  const resumeOverlay = document.getElementById('resume-overlay');
  const modelStatus = document.getElementById('model-status');
  const zoomWrap = document.getElementById('zoom-wrap');
  const zoomSlider = document.getElementById('zoom-slider');
  const zoomValue = document.getElementById('zoom-value');

  if (!video || !shutter || !overlay) return;

  let stream = null;
  let videoTrack = null;
  let isSearching = false;
  let lastInferTime = 0;
  let detectionActive = true;
  let rafId = null;

  // Interactive selection & freeze-frame state
  let isFrozen = false;
  let isPausedForResults = false;
  let selectedBoxIndex = -1;
  let frozenBoxes = [];
  let currentBoxes = [];
  let currentBox = null;

  async function setZoom(value) {
    if (!videoTrack) return;
    try {
      await videoTrack.applyConstraints({ advanced: [{ zoom: value }] });
    } catch (e) {
      console.error('apply zoom failed', e);
      sendClientLog('warn', 'camera zoom apply failed', { error: e.name + ': ' + e.message });
    }
  }

  function initZoom(track) {
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    if (!caps.zoom || zoomSlider === null) return;

    const { min = 1, max = min, step = 0.1 } = caps.zoom;
    zoomSlider.min = min;
    zoomSlider.max = max;
    zoomSlider.step = step;

    let defaultValue = 2.0;
    if (defaultValue < min) defaultValue = min;
    if (defaultValue > max) defaultValue = max;

    zoomSlider.value = defaultValue;
    if (zoomValue) zoomValue.textContent = Number(defaultValue).toFixed(1) + '×';

    zoomSlider.addEventListener('input', () => {
      const val = parseFloat(zoomSlider.value);
      if (zoomValue) zoomValue.textContent = val.toFixed(1) + '×';
      setZoom(val);
    });

    if (zoomWrap) zoomWrap.classList.remove('hidden');
    setZoom(defaultValue);
  }

  function setStatus(text) {
    if (cameraStatus) cameraStatus.textContent = text;
  }

  function showPermission(text) {
    if (permissionMsg) {
      permissionMsg.textContent = text;
      permissionMsg.classList.remove('hidden');
    }
    if (cameraWrap) cameraWrap.classList.add('hidden');
    if (shutter) shutter.classList.add('hidden');
    if (overlay) overlay.classList.add('hidden');
  }

  async function startCamera() {
    if (stream && video.srcObject) {
      video.play().catch(() => {});
      return;
    }

    setStatus('Подключение к камере…');
    const baseConstraints = {
      facingMode: 'environment',
      width: { ideal: 1280, max: 1920 },
      height: { ideal: 720, max: 1080 }
    };

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: baseConstraints });
    } catch (e1) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (e2) {
        if (e2.name === 'NotAllowedError') {
          showPermission('Доступ к камере запрещён. Разрешите доступ к камере в настройках браузера и обновите страницу.');
        } else if (e2.name === 'NotFoundError') {
          showPermission('Камера не найдена. Убедитесь, что устройство имеет камеру.');
        } else {
          showPermission('Не удалось включить камеру: ' + e2.message + '. Если вы открыли страницу по HTTP (не HTTPS), камера может быть недоступна.');
        }
        sendClientLog('error', 'camera init failed', { error: e2.name + ': ' + e2.message });
        return;
      }
    }
    if (stream) {
      videoTrack = stream.getVideoTracks()[0];
      initZoom(videoTrack);
    }
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play().catch(() => {});
      setStatus('Наведите этикетку');
      if (shutter) shutter.classList.remove('hidden');
      resizeOverlay();
    };
  }

  function resizeOverlay() {
    if (!overlay || !video) return;
    if (overlay.width !== video.clientWidth || overlay.height !== video.clientHeight) {
      overlay.width = video.clientWidth;
      overlay.height = video.clientHeight;
    }
    if (freezeCanvas && (freezeCanvas.width !== video.clientWidth || freezeCanvas.height !== video.clientHeight)) {
      freezeCanvas.width = video.clientWidth;
      freezeCanvas.height = video.clientHeight;
    }
  }

  async function detectionLoop() {
    if (!detectionActive || isSearching || isFrozen || isPausedForResults) {
      rafId = requestAnimationFrame(detectionLoop);
      return;
    }

    const now = performance.now();
    if (video.videoWidth > 0 && isDetectorReady() && (now - lastInferTime >= INFER_INTERVAL_MS)) {
      lastInferTime = now;
      detect(video).then((res) => {
        if (!isFrozen && !isPausedForResults && res && res.boxes !== undefined) {
          resizeOverlay();
          const ctx = overlay.getContext('2d');
          const stats = getDetectorStats();
          const mappedBoxes = mapBoxesToElement(res.boxes, overlay.width, overlay.height);
          currentBoxes = mappedBoxes;
          currentBox = mappedBoxes[0] || null;
          drawBoxes(ctx, mappedBoxes, 'label', stats, -1);
        }
      }).catch((e) => {
        console.error('detection error', e);
      });
    }

    rafId = requestAnimationFrame(detectionLoop);
  }

  function stopDetection() {
    detectionActive = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  function startDetection() {
    if (rafId) return;
    detectionActive = true;
    rafId = requestAnimationFrame(detectionLoop);
  }

  function freezeAtBox(boxIndex) {
    if (!currentBoxes || boxIndex < 0 || boxIndex >= currentBoxes.length) return;

    isFrozen = true;
    selectedBoxIndex = boxIndex;
    frozenBoxes = [...currentBoxes];
    stopDetection();

    // Freeze video frame onto the freezeCanvas
    if (freezeCanvas && video && video.videoWidth > 0) {
      resizeOverlay();
      const fCtx = freezeCanvas.getContext('2d');
      fCtx.drawImage(video, 0, 0, freezeCanvas.width, freezeCanvas.height);
      freezeCanvas.classList.remove('hidden');
    }

    // Redraw bounding boxes with selected box in bright red
    const ctx = overlay.getContext('2d');
    drawBoxes(ctx, frozenBoxes, 'label', null, selectedBoxIndex);

    setStatus('Кадр зафиксирован. Нажмите на красную рамку для поиска или мимо для отмены');
  }

  function unfreeze() {
    isFrozen = false;
    selectedBoxIndex = -1;
    frozenBoxes = [];

    if (freezeCanvas) {
      freezeCanvas.classList.add('hidden');
    }

    setStatus('Наведите этикетку');
    startDetection();
  }

  function pausePreviewForResults() {
    isPausedForResults = true;
    stopDetection();

    // If freezeCanvas was not shown yet, capture current frame to show frozen blurred preview
    if (freezeCanvas && video && video.videoWidth > 0 && freezeCanvas.classList.contains('hidden')) {
      resizeOverlay();
      const fCtx = freezeCanvas.getContext('2d');
      fCtx.drawImage(video, 0, 0, freezeCanvas.width, freezeCanvas.height);
      freezeCanvas.classList.remove('hidden');
    }

    // Clear bounding boxes from overlay
    if (overlay) {
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    if (cameraWrap) cameraWrap.classList.add('paused');
    if (resumeOverlay) resumeOverlay.classList.remove('hidden');
    setStatus('Нажмите на окно камеры, чтобы продолжить сканирование');
  }

  function resumePreview() {
    isPausedForResults = false;
    isFrozen = false;
    selectedBoxIndex = -1;
    frozenBoxes = [];

    if (cameraWrap) cameraWrap.classList.remove('paused');
    if (resumeOverlay) resumeOverlay.classList.add('hidden');
    if (freezeCanvas) freezeCanvas.classList.add('hidden');
    if (resultSection) resultSection.classList.add('hidden');

    setStatus('Наведите этикетку');
    startDetection();

    // Smooth scroll back to top camera view
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function handleOverlayTap(e) {
    if (isSearching || isPausedForResults || !video || !video.videoWidth) return;

    const rect = overlay.getBoundingClientRect();
    let clientX = e.clientX;
    let clientY = e.clientY;

    if (e.changedTouches && e.changedTouches.length > 0) {
      clientX = e.changedTouches[0].clientX;
      clientY = e.changedTouches[0].clientY;
    } else if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }

    const tapX = clientX - rect.left;
    const tapY = clientY - rect.top;

    if (isFrozen) {
      // FROZEN mode: check if clicked on the highlighted red box
      const targetBox = frozenBoxes[selectedBoxIndex];
      if (targetBox && isPointInBox(tapX, tapY, targetBox, 18)) {
        // Tapped on the RED box -> Trigger search from crop!
        captureAndSearch(targetBox, freezeCanvas || video);
      } else {
        // Tapped outside -> Unfreeze and resume live detection
        unfreeze();
      }
    } else {
      // LIVE mode: check if clicked inside any detected bounding box
      if (!currentBoxes || !currentBoxes.length) return;

      let hitIdx = -1;
      for (let i = 0; i < currentBoxes.length; i++) {
        if (isPointInBox(tapX, tapY, currentBoxes[i], 18)) {
          hitIdx = i;
          break;
        }
      }

      if (hitIdx >= 0) {
        freezeAtBox(hitIdx);
      }
    }
  }

  async function loadModel() {
    if (modelStatus) {
      modelStatus.textContent = 'Ожидание ONNX Runtime…';
      modelStatus.classList.remove('ready', 'error');
    }

    // Wait for ONNX Runtime to load from CDN
    let ortWait = 0;
    while (typeof window.ort === 'undefined' && ortWait < 5000) {
      await new Promise(r => setTimeout(r, 100));
      ortWait += 100;
    }
    if (typeof window.ort === 'undefined') {
      if (modelStatus) {
        modelStatus.textContent = 'ONNX Runtime не загрузился. Проверьте интернет / CDN.';
        modelStatus.classList.add('error');
      }
      return;
    }

    if (modelStatus) {
      modelStatus.textContent = 'Инициализация INT8 детектора…';
    }

    try {
      const ok = await loadDetector({ modelUrl });
      if (modelStatus) {
        if (ok) {
          const stats = getDetectorStats();
          modelStatus.textContent = `Детектор готов (${stats.provider.toUpperCase()})`;
          modelStatus.classList.add('ready');
        } else {
          modelStatus.textContent = 'Детектор не загрузился — будет серверная детекция';
          modelStatus.classList.add('error');
        }
      }
    } catch (e) {
      if (modelStatus) {
        modelStatus.textContent = 'Ошибка детектора: ' + (e.message || String(e));
        modelStatus.classList.add('error');
      }
      console.error('[loadModel]', e);
    }
  }

  async function captureAndSearch(targetBox = null, sourceElement = null) {
    if (isSearching || !video.videoWidth) return;
    isSearching = true;
    if (shutter) shutter.disabled = true;
    setStatus('Ищу…');

    try {
      let blob = null;
      let endpoint = `${API_BASE}/api/cascade/search`;
      const srcEl = sourceElement || (isFrozen && freezeCanvas ? freezeCanvas : video);
      const boxToCrop = targetBox || currentBox || (currentBoxes && currentBoxes[0]);

      if (isDetectorReady() && boxToCrop && boxToCrop.conf >= 0.25) {
        blob = await cropLabel(srcEl, boxToCrop, 518);
        if (blob) {
          endpoint = `${API_BASE}/api/cascade/search-from-crop`;
        }
      }

      if (!blob) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(srcEl, 0, 0, canvas.width, canvas.height);
        blob = await new Promise((resolve, reject) => {
          canvas.toBlob((b) => b ? resolve(b) : reject(new Error('не удалось сохранить кадр')), 'image/jpeg', 0.92);
        });
        endpoint = `${API_BASE}/api/cascade/search`;
      }


      const formData = new FormData();
      formData.append('image', blob, 'photo.jpg');
      formData.append('k', '5');

      sendClientLog('info', 'starting cascade camera search', {
        clientCrop: endpoint.includes('search-from-crop'),
        fromFrozen: isFrozen,
        blobSize: blob.size
      });

      const res = await fetchWithTimeout(endpoint, {
        method: 'POST',
        mode: 'cors',
        credentials: 'omit',
        body: formData,
      });

      if (!res.ok) {
        let msg = res.statusText;
        try {
          const data = await res.json();
          if (data.error) msg = data.error;
        } catch (_) {}
        throw new Error('HTTP ' + res.status + ': ' + msg);
      }

      const data = await res.json();

      // Pause preview with blur, keep camera alive & zoom untouched!
      pausePreviewForResults();

      if (resultSection) resultSection.classList.remove('hidden');
      renderResult(data);

      // Smoothly scroll down to results
      setTimeout(() => {
        if (resultSection) {
          resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);

    } catch (e) {
      console.error('search error', e);
      sendClientLog('error', 'cascade search failed', { error: e.name + ': ' + e.message, message: formatError(e) });
      setStatus('Ошибка: ' + formatError(e));
    } finally {
      isSearching = false;
      if (shutter) shutter.disabled = false;
    }
  }

  function renderResult(data) {
    const winner = data.winner;
    const timings = data.timings;
    const cropUrl = data.v4_query_crop || data.bbox_crop;
    const stageReached = data.stage_reached || '';
    const isV1Only = stageReached === 'v1_confident';

    if (resultStatus) {
      if (timings) {
        resultStatus.textContent = `⚡ Каскад за ${timings.total_ms} мс (Detect: ${timings.bbox_detect_ms || 0}ms, v1: ${timings.v1_total_ms}ms, v4: ${timings.v4_total_ms || 0}ms)`;
      } else if (data.time) {
        resultStatus.textContent = `Поиск занял ${data.time.toFixed(2)} с`;
      }
    }

    const cropImg = document.getElementById('result-crop-img');
    if (cropImg && cropUrl) {
      cropImg.src = cropUrl.startsWith('data:') ? cropUrl : API_BASE + cropUrl + '?v=' + Date.now();
      cropImg.onclick = () => window.openMobileLightbox(cropImg.src, 'Кроп этикетки (запрос)');
    }

    const hero = document.getElementById('hero-card');
    if (hero) {
      if (winner) {
        const title = winner.title || winner.product_id || 'Неизвестный товар';
        const mfg = winner.manufacturer || '';
        const scoreRaw = winner.final_score !== undefined ? winner.final_score : winner.dino_similarity;
        const score = scoreRaw !== undefined ? (scoreRaw * 100).toFixed(2) : '—';
        const imgSrc = winner.image_url || '';
        const badgeColor = isV1Only ? '#00e676' : '#a29bfe';
        const badgeText = isV1Only ? 'Этап 1: DINOv2' : 'Этап 2: SigLIP 2 / OCR';
        const lowConfidence = scoreRaw !== undefined && scoreRaw < 0.75;

        hero.innerHTML = `
          <div style="background:${lowConfidence ? '#fff8e1' : '#f0fdf4'};border:2px solid ${lowConfidence ? '#ffab00' : badgeColor};border-radius:12px;padding:14px;box-shadow:0 4px 14px rgba(0,0,0,0.1);">
            <div style="display:inline-block;background:${badgeColor};color:#000;font-size:11px;font-weight:800;padding:4px 10px;border-radius:20px;margin-bottom:8px;">${lowConfidence ? '⚠️ НИЗКАЯ УВЕРЕННОСТЬ' : '🏆 ' + badgeText}</div>
            <div style="display:flex;gap:12px;align-items:center;">
              <div style="width:80px;height:80px;background:#fff;border-radius:8px;border:1px solid #e2e8f0;overflow:hidden;display:flex;align-items:center;justify-content:center;flex-shrink:0;cursor:pointer;" onclick="openMobileLightbox('${imgSrc}', '${escapeHtml(title)}')">
                <img src="${imgSrc}" style="max-width:100%;max-height:100%;object-fit:contain;" alt="Winner">
              </div>
              <div style="flex:1;min-width:0;">
                <div style="font-size:15px;font-weight:700;color:#15803d;line-height:1.3;margin-bottom:2px;">${escapeHtml(title)}</div>
                ${mfg ? `<div style="font-size:12px;color:#166534;margin-bottom:4px;">${escapeHtml(mfg)}</div>` : ''}
                <div style="font-size:12px;font-weight:700;color:#15803d;">Уверенность: ${score}%</div>
                ${winner.product_id ? `<a href="/products/${winner.product_id}" style="font-size:11px;color:#2563eb;text-decoration:underline;">Открыть карточку</a>` : ''}
              </div>
            </div>
          </div>
        `;
      } else {
        hero.innerHTML = `<div style="background:#fff8e1;border:2px solid #ffab00;border-radius:12px;padding:14px;color:#b45309;font-weight:700;">Не найдено или низкая уверенность</div>`;
      }
    }

    const others = document.getElementById('other-results');
    if (others) {
      let html = '';
      const candidates = data.v4_results && data.v4_results.length ? data.v4_results : (data.final_results || []);
      if (candidates.length > 1) {
        html += '<h3 style="margin:14px 0 8px 0;font-size:14px;color:#334155;">Кандидаты</h3>';
        html += candidates.slice(1, 6).map((it, i) => {
          const t = it.title || '—';
          const img = it.image_url || '';
          const sim = it.final_score !== undefined ? (it.final_score * 100).toFixed(1) : (it.dino_similarity !== undefined ? (it.dino_similarity * 100).toFixed(1) : '—');
          return `
            <div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid #a29bfe;border-radius:8px;padding:8px 10px;display:flex;align-items:center;gap:10px;margin-bottom:6px;">
              <div style="font-size:11px;font-weight:700;color:#64748b;min-width:20px;">#${i+2}</div>
              <div style="width:40px;height:40px;background:#f8fafc;border-radius:6px;border:1px solid #cbd5e1;overflow:hidden;flex-shrink:0;display:flex;align-items:center;justify-content:center;cursor:pointer;" onclick="openMobileLightbox('${img}', '${escapeHtml(t)}')">
                <img src="${img}" style="max-width:100%;max-height:100%;object-fit:contain;" alt="">
              </div>
              <div style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;font-weight:600;">${escapeHtml(t)}</div>
              <div style="font-size:11px;font-weight:700;color:#2563eb;">${sim}%</div>
            </div>
          `;
        }).join('');
      }
      others.innerHTML = html;
    }
  }


  // Event Listeners
  if (shutter) shutter.addEventListener('click', () => captureAndSearch());
  if (scanAgain) scanAgain.addEventListener('click', resumePreview);
  if (resumeOverlay) resumeOverlay.addEventListener('click', resumePreview);

  // Overlay tap / click handler for freeze & crop selection
  overlay.addEventListener('click', handleOverlayTap);
  overlay.addEventListener('touchend', (e) => {
    e.preventDefault();
    handleOverlayTap(e);
  });

  const toggle = document.getElementById('details-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const details = document.getElementById('hero-details');
      if (!details) return;
      details.classList.toggle('hidden');
      toggle.textContent = details.classList.contains('hidden') ? 'Подробнее' : 'Скрыть';
    });
  }

  window.addEventListener('resize', resizeOverlay);

  startCamera();
  startDetection();
  loadModel();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCascadeMobile);
} else {
  initCascadeMobile();
}
