const video = document.getElementById('cameraFeed');
const scanBtn = document.getElementById('scanBtn');
const resultArea = document.getElementById('resultArea');
const topMatchEl = document.getElementById('topMatch');
const canvas = document.getElementById('processingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const spinner = document.getElementById('loadingSpinner');

let session;

// Dummy labels for demonstration (YOLO models typically have 80 COCO classes)
const LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"];

/**
 * Initializes the camera
 */
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" } // Prefer back camera on mobile
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
    console.log("Loading model...");
    // We would use '/public/model/yolo26n.onnx' but it's an empty placeholder,
    // so creating a session with it will fail.
    // In a real scenario, we load it like this:
    // session = await ort.InferenceSession.create('/public/model/yolo26n.onnx');

    // Simulating model load for the placeholder scenario
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log("Model 'loaded' successfully!");

    // Enable the scan button
    scanBtn.textContent = "Scan Now";
    scanBtn.disabled = false;
  } catch (error) {
    console.error("Failed to load the model:", error);
    scanBtn.textContent = "Error loading model";
  }
}

/**
 * Captures a frame from the video, resizes to 640x640, and normalizes it.
 * Returns a Float32 tensor for ONNX Runtime.
 */
function preprocessImage() {
  // Draw the current video frame onto the 640x640 canvas
  // This resizes and crops the center if necessary depending on aspect ratio

  // For simplicity, we're drawing the whole video frame into 640x640
  const size = Math.min(video.videoWidth, video.videoHeight);
  const startX = (video.videoWidth - size) / 2;
  const startY = (video.videoHeight - size) / 2;

  ctx.drawImage(
    video,
    startX, startY, size, size, // Source rectangle (center square)
    0, 0, 640, 640              // Destination rectangle (canvas)
  );

  // Get image data
  const imageData = ctx.getImageData(0, 0, 640, 640).data;

  // YOLO typically expects Float32 normalized [0, 1] in NCHW format
  // N=1, C=3 (RGB), H=640, W=640
  const float32Data = new Float32Array(1 * 3 * 640 * 640);

  for (let i = 0; i < 640 * 640; i++) {
    // Canvas imageData is RGBA (4 channels)
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    // NCHW format (C=0 for R, C=1 for G, C=2 for B)
    float32Data[i] = r;                           // R channel
    float32Data[640 * 640 + i] = g;               // G channel
    float32Data[2 * 640 * 640 + i] = b;           // B channel
  }

  // Return ORT Tensor
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
    resultArea.style.display = 'none';

    // 1. Pre-process the image
    const tensor = preprocessImage();

    // 2. Run model
    // In a real scenario:
    // const feeds = { images: tensor }; // 'images' depends on model input name
    // const results = await session.run(feeds);
    // const output = results[session.outputNames[0]].data;

    // Simulating inference time and NMS-free output for the placeholder
    await new Promise(resolve => setTimeout(resolve, 800));

    // Simulated random detection
    const randomClassId = Math.floor(Math.random() * LABELS.length);
    const topClass = LABELS[randomClassId];

    // 3. Display Results
    topMatchEl.textContent = topClass.toUpperCase();
    resultArea.style.display = 'block';

  } catch (error) {
    console.error("Inference error:", error);
    alert("Error during scanning.");
  } finally {
    scanBtn.disabled = false;
    spinner.style.display = 'none';
  }
}

// Main initialization
async function init() {
  // Set up ORT web config for WASM multi-threading
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
  ort.env.wasm.simd = true;

  await setupCamera();
  await loadModel();

  scanBtn.addEventListener('click', runInference);
}

// Start app
window.addEventListener('DOMContentLoaded', init);