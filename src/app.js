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

// Recycling specific dummy labels with their recyclability status
const LABELS = [
  { name: "plastic bottle", recyclable: true },
  { name: "cardboard box", recyclable: true },
  { name: "soda can", recyclable: true },
  { name: "glass jar", recyclable: true },
  { name: "paper", recyclable: true },
  { name: "styrofoam cup", recyclable: false },
  { name: "plastic bag", recyclable: false },
  { name: "apple core", recyclable: false },
  { name: "candy wrapper", recyclable: false },
  { name: "chip bag", recyclable: false }
];

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

    // Simulate model load for placeholder scenario
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log("Model 'loaded' successfully!");

    scanBtn.disabled = false;
    statusText.textContent = "Ready to scan";
  } catch (error) {
    console.error("Failed to load the model:", error);
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

    // 2. Run model (Simulated)
    await new Promise(resolve => setTimeout(resolve, 800));

    const randomClassId = Math.floor(Math.random() * LABELS.length);
    const result = LABELS[randomClassId];

    // 3. Display Results
    detectedClassEl.textContent = result.name;

    if (result.recyclable) {
      recyclableStatusEl.textContent = "IS RECYCLABLE";
      recyclableStatusEl.className = "recyclable-status status-yes";
    } else {
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