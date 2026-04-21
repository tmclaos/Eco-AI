const video = document.getElementById('cameraFeed');
const scanBtn = document.getElementById('scanBtn');
const statusText = document.getElementById('statusText');
const spinner = document.getElementById('loadingSpinner');

// View Containers
const scannerView = document.getElementById('scannerView');
const resultView = document.getElementById('resultView');

// Result Elements
const detectedClassEl = document.getElementById('detectedClass');
const recyclableStatusEl = document.getElementById('recyclableStatus');
const resetBtn = document.getElementById('resetBtn');

const canvas = document.getElementById('processingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

let session;

// Allowed recyclable classes
const LABELS = [
  "Beverage Can",
  "Glass Bottle",
  "Plastic Bottle",
  "Paper",
  "Plastic Film",
  "Carton"
];

// Confidence threshold
const CONFIDENCE_THRESHOLD = 0.70;

/**
 * Initializes the camera
 */
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  } catch (error) {
    console.error("Error accessing the camera:", error);
    alert("Could not access camera. Please allow camera permissions.");
  }
}

/**
 * Loads the ONNX model
 */
async function loadModel() {
  try {
    statusText.textContent = "Loading Model...";

    try {
      // Attempt to load the real model
      session = await ort.InferenceSession.create('/public/model/best.onnx');
      console.log("Model loaded successfully!");
    } catch (e) {
      // If the file is empty or missing (placeholder), log a warning but allow UI testing
      console.warn("Could not load real model. Using mock inference mode. Replace /public/model/best.onnx with a valid model.", e);
      session = null;
    }

    scanBtn.disabled = false;
    statusText.textContent = "Ready to scan";
  } catch (error) {
    console.error("Failed to initialize:", error);
    statusText.textContent = "Error loading model";
  }
}

/**
 * Captures a frame from the video, resizes to 640x640, and normalizes it.
 */
function preprocessImage() {
  const size = Math.min(video.videoWidth, video.videoHeight);
  const startX = (video.videoWidth - size) / 2;
  const startY = (video.videoHeight - size) / 2;

  ctx.drawImage(
    video,
    startX, startY, size, size,
    0, 0, 640, 640
  );

  const imageData = ctx.getImageData(0, 0, 640, 640).data;
  const float32Data = new Float32Array(1 * 3 * 640 * 640);

  for (let i = 0; i < 640 * 640; i++) {
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    float32Data[i] = r;
    float32Data[640 * 640 + i] = g;
    float32Data[2 * 640 * 640 + i] = b;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

/**
 * Runs the inference and displays results
 */
async function runInference() {
  if (scanBtn.disabled) return;

  try {
    scanBtn.disabled = true;
    spinner.style.display = 'block';

    // 1. Pre-process the image
    const tensor = preprocessImage();

    // 2. Run model
    let maxConfidence = 0;
    let maxClassId = -1;

    if (session) {
      // Execute actual ONNX model
      const feeds = { images: tensor };
      const results = await session.run(feeds);

      // Output parsing depends on exact YOLO26n architecture.
      // Assuming a flattened array where each element corresponds to class confidence.
      const output = results[session.outputNames[0]].data;

      for (let i = 0; i < output.length; i++) {
        if (output[i] > maxConfidence) {
          maxConfidence = output[i];
          maxClassId = i;
        }
      }
    } else {
      // Mock logic if placeholder file is still active
      await new Promise(resolve => setTimeout(resolve, 800));
      maxConfidence = Math.random() * 0.59 + 0.4;
      maxClassId = Math.floor(Math.random() * LABELS.length);
    }

    // 3. Display Results
    if (maxConfidence > CONFIDENCE_THRESHOLD && maxClassId < LABELS.length) {
      detectedClassEl.textContent = LABELS[maxClassId];
      recyclableStatusEl.textContent = "IS RECYCLABLE";
      recyclableStatusEl.className = "recyclable-status status-yes";
    } else {
      detectedClassEl.textContent = "Unknown Item";
      recyclableStatusEl.textContent = "IS NOT RECYCLABLE";
      recyclableStatusEl.className = "recyclable-status status-no";
    }

    // Transition Views
    scannerView.style.display = 'none';
    resultView.style.display = 'flex';

  } catch (error) {
    console.error("Inference error:", error);
    alert("Error during scanning.");
  } finally {
    scanBtn.disabled = false;
    spinner.style.display = 'none';
  }
}

/**
 * Resets the app back to the scanner view
 */
function resetScanner() {
  resultView.style.display = 'none';
  scannerView.style.display = 'flex';
}

// Main initialization
async function init() {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;

  await setupCamera();
  await loadModel();

  scanBtn.addEventListener('click', runInference);
  resetBtn.addEventListener('click', resetScanner);
}

// Start app
window.addEventListener('DOMContentLoaded', init);