// export/v5m/detector.js
// High-performance client-side YOLOv8 label detector using Web Worker & WebGPU/WASM INT8 with main-thread fallback.

const CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
const ONNX_VERSION = (typeof window !== 'undefined' && window.MOBI_ONNX_VERSION) ? window.MOBI_ONNX_VERSION : 'v5m-int8-3';
const MODEL_URL = new URL(`./yolov8_label.onnx?${ONNX_VERSION}`, import.meta.url).href;
const WORKER_URL = new URL('./detector.worker.js', import.meta.url).href;

const DEFAULT_INPUT_SIZE = 384;
const CONF_THRESHOLD = 0.60; // 60% confidence threshold
const IOU_THRESHOLD = 0.45;

let worker = null;
let mainThreadSession = null;
let isReady = false;
let isInitializing = false;
let isWorkerMode = true;
let activeProvider = 'wasm';
let inputSize = DEFAULT_INPUT_SIZE;

// Concurrency control & stats
let isDetecting = false;
let requestIdCounter = 0;
let pendingResolve = null;
let lastBoxes = [];
let lastInferTimeMs = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

// Main thread fallback preallocations
let mainCanvas = null;
let mainCtx = null;
let mainTensorData = null;

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function postprocessMain(output, size, confThresh = CONF_THRESHOLD) {
  const [batch, channels, anchors] = output.dims;
  const raw = output.data;
  const boxes = [];

  for (let i = 0; i < anchors; i++) {
    const rawConf = raw[i + anchors * 4];
    const conf = (rawConf > 1.0 || rawConf < 0.0) ? (1.0 / (1.0 + Math.exp(-rawConf))) : rawConf;
    if (conf >= confThresh) {
      const cx = raw[i];
      const cy = raw[i + anchors];
      const w = raw[i + anchors * 2];
      const h = raw[i + anchors * 3];
      boxes.push({
        x1: Math.max(0, cx - w / 2),
        y1: Math.max(0, cy - h / 2),
        x2: Math.min(size, cx + w / 2),
        y2: Math.min(size, cy + h / 2),
        conf: conf
      });
    }
  }

  if (boxes.length === 0) return [];
  boxes.sort((a, b) => b.conf - a.conf);

  const selected = [];
  for (const box of boxes) {
    let keep = true;
    for (const s of selected) {
      if (iou(box, s) > IOU_THRESHOLD) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push(box);
    }
  }
  return selected;
}

async function loadMainThreadFallback(modelUrl) {
  console.warn('[Detector v5m] Initializing main thread fallback...');
  try {
    if (typeof window.ort === 'undefined') {
      console.error('[Detector v5m] window.ort is undefined');
      return false;
    }
    window.ort.env.wasm.wasmPaths = CDN_BASE;
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.simd = true;

    mainCanvas = document.createElement('canvas');
    mainCanvas.width = inputSize;
    mainCanvas.height = inputSize;
    mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });
    mainTensorData = new Float32Array(3 * inputSize * inputSize);

    mainThreadSession = await window.ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all'
    });
    activeProvider = 'wasm (main-thread)';
    isWorkerMode = false;
    isReady = true;
    console.log('[Detector v5m] Main thread fallback loaded successfully');
    return true;
  } catch (err) {
    console.error('[Detector v5m] Main thread fallback failed:', err);
    return false;
  }
}

export async function loadDetector(options = {}) {
  if (isReady) return true;
  if (isInitializing) return false;
  isInitializing = true;

  const modelUrl = options.modelUrl || MODEL_URL;
  inputSize = options.inputSize || DEFAULT_INPUT_SIZE;
  const conf = options.confThreshold || CONF_THRESHOLD;
  const iou = options.iouThreshold || IOU_THRESHOLD;

  // Try Web Worker first
  const workerSuccess = await new Promise((resolve) => {
    try {
      worker = new Worker(WORKER_URL);

      const timeoutId = setTimeout(() => {
        console.warn('[Detector v5m] Worker initialization timeout (5s)');
        resolve(false);
      }, 5000);

      worker.onmessage = (e) => {
        const msg = e.data;
        if (!msg) return;

        if (msg.type === 'init_done') {
          clearTimeout(timeoutId);
          if (msg.ok) {
            isReady = true;
            isWorkerMode = true;
            activeProvider = msg.provider || 'wasm';
            console.log(`[Detector v5m] Ready via Worker. Provider: ${activeProvider}, inputSize: ${inputSize}, confThresh: ${conf}`);
            resolve(true);
          } else {
            console.warn('[Detector v5m] Worker init reported failure:', msg.error);
            resolve(false);
          }
        } else if (msg.type === 'detect_result') {
          isDetecting = false;
          lastBoxes = msg.boxes || (msg.box ? [msg.box] : []);
          lastInferTimeMs = msg.inferTime || 0;
          if (msg.provider) activeProvider = msg.provider;

          // FPS counter
          frameCount++;
          const now = performance.now();
          if (now - lastFpsTime >= 1000) {
            currentFps = Math.round((frameCount * 1000) / (now - lastFpsTime));
            frameCount = 0;
            lastFpsTime = now;
          }

          if (pendingResolve) {
            const cb = pendingResolve;
            pendingResolve = null;
            cb({
              boxes: lastBoxes,
              box: lastBoxes[0] || null,
              inferTime: lastInferTimeMs,
              provider: activeProvider,
              fps: currentFps
            });
          }
        }
      };

      worker.onerror = (err) => {
        clearTimeout(timeoutId);
        console.warn('[Detector v5m] Worker error during init:', err);
        resolve(false);
      };

      worker.postMessage({
        type: 'init',
        modelUrl: modelUrl,
        inputSize: inputSize,
        confThreshold: conf,
        iouThreshold: iou
      });
    } catch (e) {
      console.warn('[Detector v5m] Failed to spawn worker:', e);
      resolve(false);
    }
  });

  if (workerSuccess) {
    isInitializing = false;
    return true;
  }

  // Fallback to main thread
  if (worker) {
    try { worker.terminate(); } catch (_) {}
    worker = null;
  }

  const mainOk = await loadMainThreadFallback(modelUrl);
  isInitializing = false;
  return mainOk;
}

export function isDetectorReady() {
  return isReady;
}

export function getDetectorStats() {
  return {
    isReady,
    isBusy: isDetecting,
    isWorker: isWorkerMode,
    provider: activeProvider,
    inferTimeMs: lastInferTimeMs,
    fps: currentFps,
    inputSize,
    confThreshold: CONF_THRESHOLD
  };
}

/**
 * Async non-blocking detection on video stream.
 * Returns { boxes: Array<Box>, box: TopBox, inferTime, provider, fps }
 */
export async function detect(videoElement) {
  if (!isReady || !videoElement || !videoElement.videoWidth) {
    return { boxes: lastBoxes, box: lastBoxes[0] || null, skipped: true };
  }

  if (isDetecting) {
    return { boxes: lastBoxes, box: lastBoxes[0] || null, skipped: true, busy: true };
  }

  isDetecting = true;
  const reqId = ++requestIdCounter;

  // Worker mode
  if (isWorkerMode && worker) {
    try {
      const bitmap = await createImageBitmap(videoElement, {
        resizeWidth: inputSize,
        resizeHeight: inputSize,
        resizeQuality: 'low'
      });

      return new Promise((resolve) => {
        pendingResolve = resolve;
        worker.postMessage({
          type: 'detect',
          id: reqId,
          bitmap: bitmap
        }, [bitmap]);
      });
    } catch (err) {
      isDetecting = false;
      return { boxes: lastBoxes, box: lastBoxes[0] || null, error: err.message };
    }
  }

  // Main thread fallback mode
  if (mainThreadSession) {
    const t0 = performance.now();
    try {
      mainCtx.drawImage(videoElement, 0, 0, inputSize, inputSize);
      const imgData = mainCtx.getImageData(0, 0, inputSize, inputSize).data;
      const total = inputSize * inputSize;

      for (let i = 0; i < total; i++) {
        const i4 = i * 4;
        mainTensorData[i] = imgData[i4] / 255.0;
        mainTensorData[i + total] = imgData[i4 + 1] / 255.0;
        mainTensorData[i + total * 2] = imgData[i4 + 2] / 255.0;
      }

      const tensor = new window.ort.Tensor('float32', mainTensorData, [1, 3, inputSize, inputSize]);
      const results = await mainThreadSession.run({ images: tensor });
      const output = results.output0 || results[Object.keys(results)[0]];
      lastBoxes = postprocessMain(output, inputSize, CONF_THRESHOLD);
      lastInferTimeMs = Math.round(performance.now() - t0);

      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        currentFps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        frameCount = 0;
        lastFpsTime = now;
      }

      isDetecting = false;
      return {
        boxes: lastBoxes,
        box: lastBoxes[0] || null,
        inferTime: lastInferTimeMs,
        provider: activeProvider,
        fps: currentFps
      };
    } catch (mErr) {
      isDetecting = false;
      return { boxes: lastBoxes, box: lastBoxes[0] || null, error: mErr.message };
    }
  }

  isDetecting = false;
  return { boxes: [], box: null, skipped: true };
}

export function mapBoxesToElement(boxes, elementWidth, elementHeight, customInputSize = inputSize) {
  if (!boxes || !boxes.length) return [];
  const sx = elementWidth / customInputSize;
  const sy = elementHeight / customInputSize;
  return boxes.map((box) => ({
    x1: box.x1 * sx,
    y1: box.y1 * sy,
    x2: box.x2 * sx,
    y2: box.y2 * sy,
    conf: box.conf
  }));
}

export function mapBoxToElement(box, elementWidth, elementHeight, customInputSize = inputSize) {
  if (!box) return null;
  const sx = elementWidth / customInputSize;
  const sy = elementHeight / customInputSize;
  return {
    x1: box.x1 * sx,
    y1: box.y1 * sy,
    x2: box.x2 * sx,
    y2: box.y2 * sy,
    conf: box.conf
  };
}

/**
 * Checks if a click/touch point (x, y) falls inside bounding box (with optional touch padding).
 */
export function isPointInBox(x, y, box, padding = 10) {
  if (!box) return false;
  return (
    x >= box.x1 - padding &&
    x <= box.x2 + padding &&
    y >= box.y1 - padding &&
    y <= box.y2 + padding
  );
}

export function drawBoxes(ctx, boxes, label, stats = null, selectedIndex = -1) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!boxes || !boxes.length) return;

  boxes.forEach((box, idx) => {
    const isSelected = (idx === selectedIndex);

    let color = box.conf >= 0.80 ? '#00e676' : box.conf >= 0.65 ? '#ffab00' : '#29b6f6';
    let lineWidth = 3;

    if (selectedIndex >= 0) {
      if (isSelected) {
        color = '#ff1744'; // Bright red for selected box
        lineWidth = 4;
      } else {
        color = 'rgba(255, 255, 255, 0.35)'; // Dim unselected boxes
        lineWidth = 2;
      }
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

    // Label badge
    ctx.fillStyle = color;
    ctx.font = 'bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

    let badgeText = `${label} ${(box.conf * 100).toFixed(0)}%`;
    if (isSelected) {
      badgeText = `🔍 Нажмите для поиска (${(box.conf * 100).toFixed(0)}%)`;
    } else if (idx === 0 && stats && stats.inferTimeMs && selectedIndex < 0) {
      badgeText += ` • ${stats.inferTimeMs}ms (${stats.provider || 'wasm'})`;
    }

    const textWidth = ctx.measureText(badgeText).width;
    ctx.fillRect(box.x1, Math.max(0, box.y1 - 24), textWidth + 14, 24);
    ctx.fillStyle = isSelected ? '#ffffff' : '#000000';
    ctx.fillText(badgeText, box.x1 + 6, Math.max(17, box.y1 - 7));
  });
}

export function drawBox(ctx, box, label, stats = null, isSelected = false) {
  if (Array.isArray(box)) {
    drawBoxes(ctx, box, label, stats, isSelected ? 0 : -1);
  } else if (box) {
    drawBoxes(ctx, [box], label, stats, isSelected ? 0 : -1);
  } else {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }
}

/**
 * Crops a label region from either a <video> or a <canvas> element into canonical square (256x256).
 */
export async function cropLabel(sourceElement, box, size = 256) {
  if (!sourceElement || !box) return null;

  const rect = sourceElement.getBoundingClientRect();
  const srcWidth = sourceElement.videoWidth || sourceElement.width;
  const srcHeight = sourceElement.videoHeight || sourceElement.height;

  const sx = srcWidth / rect.width;
  const sy = srcHeight / rect.height;

  const x1 = Math.max(0, Math.floor(box.x1 * sx));
  const y1 = Math.max(0, Math.floor(box.y1 * sy));
  const x2 = Math.min(srcWidth, Math.ceil(box.x2 * sx));
  const y2 = Math.min(srcHeight, Math.ceil(box.y2 * sy));
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) return null;

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceElement, x1, y1, w, h, 0, 0, size, size);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92);
  });
}
