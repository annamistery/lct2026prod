// export/v5m/detector.worker.js
// Dedicated Web Worker for YOLOv8 INT8 inference with WebGPU / WASM SIMD.

const CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
importScripts(CDN_BASE + 'ort.all.min.js');

let session = null;
let inputSize = 384;
let confThreshold = 0.60;
let iouThreshold = 0.45;
let activeProvider = 'wasm';
let offscreenCanvas = null;
let offscreenCtx = null;
let preallocatedTensorData = null;

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

function postprocess(output) {
  const [batch, channels, anchors] = output.dims;
  const raw = output.data;
  const boxes = [];

  for (let i = 0; i < anchors; i++) {
    const rawConf = raw[i + anchors * 4];
    const conf = (rawConf > 1.0 || rawConf < 0.0) ? (1.0 / (1.0 + Math.exp(-rawConf))) : rawConf;
    if (conf >= confThreshold) {
      const cx = raw[i];
      const cy = raw[i + anchors];
      const w = raw[i + anchors * 2];
      const h = raw[i + anchors * 3];
      boxes.push({
        x1: Math.max(0, cx - w / 2),
        y1: Math.max(0, cy - h / 2),
        x2: Math.min(inputSize, cx + w / 2),
        y2: Math.min(inputSize, cy + h / 2),
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
      if (iou(box, s) > iouThreshold) {
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

async function initSession(modelUrl, size = 384, confThresh = 0.60, iouThresh = 0.45) {
  inputSize = size;
  confThreshold = confThresh;
  iouThreshold = iouThresh;

  if (typeof OffscreenCanvas !== 'undefined') {
    offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
    offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
  }
  preallocatedTensorData = new Float32Array(3 * inputSize * inputSize);

  // CRITICAL: Point ORT WASM loader to CDN path inside Worker
  if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = CDN_BASE;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
  }

  // Try WebGPU first, then fallback to WASM
  const providersToTry = [
    { name: 'webgpu', eps: ['webgpu', 'wasm'] },
    { name: 'wasm', eps: ['wasm'] }
  ];

  let loaded = false;
  let lastErr = null;

  for (const p of providersToTry) {
    try {
      console.log(`[Worker] Attempting ORT load with: ${p.name}`);
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: p.eps,
        graphOptimizationLevel: 'all'
      });
      activeProvider = p.name;
      loaded = true;
      console.log(`[Worker] Successfully loaded model with provider: ${activeProvider}`);
      break;
    } catch (err) {
      console.warn(`[Worker] Provider ${p.name} failed:`, err);
      lastErr = err;
    }
  }

  if (!loaded || !session) {
    throw new Error(`Failed to load ONNX model: ${lastErr ? (lastErr.message || lastErr) : 'unknown error'}`);
  }

  // Warmup run
  try {
    const dummy = new ort.Tensor('float32', preallocatedTensorData, [1, 3, inputSize, inputSize]);
    await session.run({ images: dummy });
    console.log('[Worker] Warmup inference completed');
  } catch (wErr) {
    console.warn('[Worker] Warmup error:', wErr);
  }

  return { ok: true, provider: activeProvider, inputSize, confThreshold };
}

async function runInference(bitmap, id) {
  if (!session) {
    if (bitmap && bitmap.close) bitmap.close();
    return { id, boxes: [], inferTime: 0, error: 'Model not initialized' };
  }

  const t0 = performance.now();
  try {
    if (!offscreenCtx) {
      offscreenCanvas = new OffscreenCanvas(inputSize, inputSize);
      offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
    }

    offscreenCtx.drawImage(bitmap, 0, 0, inputSize, inputSize);
    if (bitmap.close) bitmap.close();

    const imgData = offscreenCtx.getImageData(0, 0, inputSize, inputSize).data;
    const total = inputSize * inputSize;

    for (let i = 0; i < total; i++) {
      const i4 = i * 4;
      preallocatedTensorData[i] = imgData[i4] / 255.0;
      preallocatedTensorData[i + total] = imgData[i4 + 1] / 255.0;
      preallocatedTensorData[i + total * 2] = imgData[i4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', preallocatedTensorData, [1, 3, inputSize, inputSize]);
    const feeds = { images: tensor };
    const results = await session.run(feeds);
    const output = results.output0 || results[Object.keys(results)[0]];
    const boxes = postprocess(output);
    const inferTime = Math.round(performance.now() - t0);

    return { id, boxes, inferTime, provider: activeProvider };
  } catch (e) {
    if (bitmap && bitmap.close) bitmap.close();
    return { id, boxes: [], inferTime: Math.round(performance.now() - t0), error: e.message };
  }
}

self.onmessage = async function(e) {
  const msg = e.data;
  if (!msg) return;

  if (msg.type === 'init') {
    try {
      const res = await initSession(msg.modelUrl, msg.inputSize, msg.confThreshold, msg.iouThreshold);
      self.postMessage({ type: 'init_done', ...res });
    } catch (err) {
      self.postMessage({ type: 'init_done', ok: false, error: err.message || String(err) });
    }
  } else if (msg.type === 'detect') {
    const res = await runInference(msg.bitmap, msg.id);
    self.postMessage({ type: 'detect_result', ...res });
  }
};
