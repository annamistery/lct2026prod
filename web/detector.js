const SIZE = 640;
const THRESHOLD = 0.25;
let session = null;

export async function loadDetector() {
  try {
    session = await ort.InferenceSession.create('./models/yolov8_label.onnx', { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return true;
  } catch (error) {
    console.warn('Client detector unavailable', error);
    return false;
  }
}

// Box in 0..SIZE coordinates of the stretched frame, or null
export async function detect(video) {
  if (!session) return null;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, SIZE, SIZE);
  const pixels = context.getImageData(0, 0, SIZE, SIZE).data;
  const tensor = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    tensor[i] = pixels[i * 4] / 255;
    tensor[i + SIZE * SIZE] = pixels[i * 4 + 1] / 255;
    tensor[i + 2 * SIZE * SIZE] = pixels[i * 4 + 2] / 255;
  }
  const output = Object.values(await session.run({ images: new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]) }))[0];
  const anchors = output.dims[2];
  const data = output.data;
  let best = null;
  for (let i = 0; i < anchors; i++) {
    const confidence = 1 / (1 + Math.exp(-data[i + anchors * 4]));
    if (confidence >= THRESHOLD && (!best || confidence > best.confidence)) {
      const cx = data[i], cy = data[i + anchors], w = data[i + anchors * 2], h = data[i + anchors * 3];
      best = { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, confidence };
    }
  }
  return best;
}

// The box is mapped into the picture area of the video: letterboxed for object-fit: contain,
// cropped around the centre for object-fit: cover
export function draw(canvas, box, video, fit = 'contain') {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!box || !video.videoWidth) return;
  const pick = fit === 'cover' ? Math.max : Math.min;
  const scale = pick(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  const left = (canvas.width - width) / 2;
  const top = (canvas.height - height) / 2;
  context.strokeStyle = '#20d45a';
  context.lineWidth = 3;
  context.strokeRect(left + box.x1 / SIZE * width, top + box.y1 / SIZE * height, (box.x2 - box.x1) / SIZE * width, (box.y2 - box.y1) / SIZE * height);
}
