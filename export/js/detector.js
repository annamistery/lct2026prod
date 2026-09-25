// export/js/detector.js
// Client-side YOLOv8 label detector using ONNX Runtime Web.

const DEFAULT_INPUT_SIZE = 384;
const CONF_THRESHOLD = 0.55;
const IOU_THRESHOLD = 0.45;

let session = null;
let isReady = false;
let isInitializing = false;
let inputSize = DEFAULT_INPUT_SIZE;

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

function postprocess(output, size, confThresh = CONF_THRESHOLD) {
  const dims = output.dims;
  const anchors = dims[2] || dims[1];
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
    if (keep) selected.push(box);
  }
  return selected;
}

export async function loadDetector(modelUrl = 'models/yolov8_label.onnx') {
  if (isReady) return true;
  if (isInitializing) return false;
  isInitializing = true;

  try {
    if (typeof window.ort === 'undefined') {
      console.error('[Detector] ONNX Runtime Web is not loaded');
      isInitializing = false;
      return false;
    }

    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.simd = true;

    mainCanvas = document.createElement('canvas');
    mainCanvas.width = inputSize;
    mainCanvas.height = inputSize;
    mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });
    mainTensorData = new Float32Array(3 * inputSize * inputSize);

    session = await window.ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all'
    });

    isReady = true;
    isInitializing = false;
    console.log('[Detector] Loaded successfully with model:', modelUrl);
    return true;
  } catch (err) {
    console.error('[Detector] Failed to load:', err);
    isInitializing = false;
    return false;
  }
}

export function isDetectorReady() {
  return isReady;
}

export async function detect(videoOrCanvas) {
  if (!isReady || !session) return [];

  mainCtx.drawImage(videoOrCanvas, 0, 0, inputSize, inputSize);
  const imgData = mainCtx.getImageData(0, 0, inputSize, inputSize);
  const data = imgData.data;
  const channelSize = inputSize * inputSize;

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    mainTensorData[p] = data[i] / 255.0;
    mainTensorData[channelSize + p] = data[i + 1] / 255.0;
    mainTensorData[2 * channelSize + p] = data[i + 2] / 255.0;
  }

  const inputTensor = new window.ort.Tensor('float32', mainTensorData, [1, 3, inputSize, inputSize]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  return postprocess(output, inputSize);
}

export function mapBoxesToElement(boxes, element, nativeSize = inputSize) {
  const rect = element.getBoundingClientRect();
  const elW = rect.width;
  const elH = rect.height;
  const scale = Math.min(elW / nativeSize, elH / nativeSize);
  const offsetX = (elW - nativeSize * scale) / 2;
  const offsetY = (elH - nativeSize * scale) / 2;

  return boxes.map(b => ({
    x1: b.x1 * scale + offsetX,
    y1: b.y1 * scale + offsetY,
    x2: b.x2 * scale + offsetX,
    y2: b.y2 * scale + offsetY,
    conf: b.conf
  }));
}

export function drawBoxes(canvas, boxes, isSelectedFn) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  boxes.forEach((b, idx) => {
    const isSelected = isSelectedFn ? isSelectedFn(idx) : false;
    ctx.strokeStyle = isSelected ? '#ffeb3b' : '#00e676';
    ctx.lineWidth = isSelected ? 4 : 2;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

    ctx.fillStyle = isSelected ? '#ffeb3b' : '#00e676';
    ctx.font = '12px sans-serif';
    ctx.fillText(`${Math.round(b.conf * 100)}%`, b.x1 + 4, Math.max(14, b.y1 - 4));
  });
}
