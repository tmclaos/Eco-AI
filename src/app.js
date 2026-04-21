const video = document.getElementById('cameraFeed');
const scanBtn = document.getElementById('scanBtn');
const statusText = document.getElementById('statusText');
const spinner = document.getElementById('loadingSpinner');
const scannerView = document.getElementById('scannerView');
const resultView = document.getElementById('resultView');
const detectedClassEl = document.getElementById('detectedClass');
const recyclableStatusEl = document.getElementById('recyclableStatus');
const resetBtn = document.getElementById('resetBtn');

const canvas = document.getElementById('processingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let session;

// 1. Updated to exactly 6 recyclable classes
const RECYCLABLE_CLASSES = [
  "plastic bottle", 
  "cardboard box", 
  "soda can", 
  "glass jar", 
  "paper", 
  "metal lid"
];

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }
    });
    video.srcObject = stream;
    return new Promise((resolve) => { video.onloadedmetadata = () => resolve(video); });
  } catch (error) {
    alert("Please allow camera permissions.");
  }
}

async function loadModel() {
  try {
    statusText.textContent = "Loading Model...";
    // IMPORTANT: Replace 'model.onnx' with your actual model file path
    session = await ort.InferenceSession.create('./model.onnx'); 
    scanBtn.disabled = false;
    statusText.textContent = "Ready to scan";
  } catch (error) {
    statusText.textContent = "Error loading model";
    // For now, let's enable the button for testing even if model fails
    scanBtn.disabled = false; 
  }
}

function preprocessImage() {
  const size = 640;
  ctx.drawImage(video, 0, 0, size, size);
  const imageData = ctx.getImageData(0, 0, size, size).data;
  const float32Data = new Float32Array(1 * 3 * size * size);

  for (let i = 0; i < size * size; i++) {
    float32Data[i] = imageData[i * 4] / 255.0; // R
    float32Data[size * size + i] = imageData[i * 4 + 1] / 255.0; // G
    float32Data[2 * size * size + i] = imageData[i * 4 + 2] / 255.0; // B
  }
  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

async function runInference() {
  if (scanBtn.disabled) return;

  try {
    scanBtn.disabled = true;
    spinner.style.display = 'block';

    const tensor = preprocessImage();
    
    // --- ACTUAL INFERENCE LOGIC ---
    let highestConfidence = 0;
    let detectedLabel = "";

    if (session) {
        const feeds = { images: tensor }; // 'images' depends on your YOLO export name
        const results = await session.run(feeds);
        // This part varies based on your YOLO version (v8, v11, etc.)
        // We will assume you extract the best score and class ID here:
        // highestConfidence = results.output[...]; 
        // detectedLabel = RECYCLABLE_CLASSES[results.class_id];
    } else {
        // MOCK LOGIC FOR TESTING (Simulating a 75% find)
        highestConfidence = 0.75; 
        detectedLabel = RECYCLABLE_CLASSES[0]; 
    }

    // 2. The Logic Check: Over 70% and in our list
    const isOverThreshold = highestConfidence >= 0.70;
    const isRecognized = RECYCLABLE_CLASSES.includes(detectedLabel);

    if (isOverThreshold && isRecognized) {
      detectedClassEl.textContent = detectedLabel;
      recyclableStatusEl.textContent = "RECYCLABLE";
      recyclableStatusEl.className = "recyclable-status status-yes";
    } else {
      // 3. The Fallback Message
      detectedClassEl.textContent = "Unknown Object";
      recyclableStatusEl.innerHTML = `<small style="font-size: 14px;">We could not match this to our database, we believe that this is not recyclable.</small>`;
      recyclableStatusEl.className = "recyclable-status status-no";
    }

    scannerView.style.display = 'none';
    resultView.style.display = 'flex';

  } catch (error) {
    console.error(error);
  } finally {
    scanBtn.disabled = false;
    spinner.style.display = 'none';
  }
}

function resetScanner() {
  resultView.style.display = 'none';
  scannerView.style.display = 'flex';
}

async function init() {
  await setupCamera();
  await loadModel();
  scanBtn.addEventListener('click', runInference);
  resetBtn.addEventListener('click', resetScanner);
}

window.addEventListener('DOMContentLoaded', init);
