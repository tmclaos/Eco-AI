const video = document.getElementById('webcam');
const scanBtn = document.getElementById('scanBtn');
const info = document.getElementById('info');
const loader = document.getElementById('loader');
const resultView = document.getElementById('resultView');
const scannerView = document.getElementById('scannerView');
const display = document.getElementById('display');
const ctx = document.getElementById('canvas').getContext('2d');

let session;
// --- STEP 1: DEFINE YOUR 6 CLASSES ---
const CLASSES = ["plastic bottle", "cardboard box", "soda can", "glass jar", "paper", "metal lid"];

async function startApp() {
  try {
    // Setup Camera
    const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment", width: 640, height: 640 } 
    });
    video.srcObject = stream;

    // --- STEP 2: LOAD THE MODEL ---
    // We use the path you provided from the root
    session = await ort.InferenceSession.create('public/model/best.onnx', { 
        executionProviders: ['wasm'] 
    });
    
    loader.style.display = 'none';
    scanBtn.disabled = false;
    info.textContent = "Ready to identify items";
  } catch (err) {
    info.textContent = "Error: Check model path or camera";
    console.error(err);
  }
}

scanBtn.onclick = async () => {
  info.textContent = "Analyzing object...";
  scanBtn.disabled = true;

  // Prepare Image
  ctx.drawImage(video, 0, 0, 640, 640);
  const data = ctx.getImageData(0, 0, 640, 640).data;
  
  const r = [], g = [], b = [];
  for (let i = 0; i < data.length; i += 4) {
    r.push(data[i] / 255.0);
    g.push(data[i+1] / 255.0);
    b.push(data[i+2] / 255.0);
  }
  
  const input = new ort.Tensor('float32', new Float32Array([...r, ...g, ...b]), [1, 3, 640, 640]);

  try {
    // Run AI Inference
    const inputName = session.inputNames[0];
    const results = await session.run({ [inputName]: input });
    const output = results[session.outputNames[0]].data; 

    let maxScore = 0;
    let classIdx = -1;

    // Post-process YOLOv8 output (checking 8400 boxes)
    for (let i = 0; i < 8400; i++) {
      for (let j = 0; j < CLASSES.length; j++) {
        const score = output[(j + 4) * 8400 + i];
        if (score > maxScore) {
          maxScore = score;
          classIdx = j;
        }
      }
    }

    showResult(classIdx, maxScore);
  } catch (e) {
    console.error(e);
    info.textContent = "Something went wrong.";
    scanBtn.disabled = false;
  }
};

function showResult(id, score) {
  scannerView.style.display = 'none';
  resultView.style.display = 'block';
  
  // --- STEP 3: THE 70% THRESHOLD CHECK ---
  if (score >= 0.70 && id !== -1) {
    display.innerHTML = `
      <p style="margin-bottom:0; color:#8a9bb2;">Detected Item:</p>
      <h2 style="color:var(--primary); text-transform:uppercase; margin-top:5px;">${CLASSES[id]}</h2>
      <p style="margin-top:20px;">This item is officially</p>
      <div class="status-yes">RECYCLABLE</div>
      <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; font-size:0.8em; margin-top:20px;">
         AI Certainty: ${(score * 100).toFixed(1)}%
      </div>
    `;
  } else {
    // --- STEP 4: THE FALLBACK MESSAGE ---
    display.innerHTML = `
      <div class="status-no">
        <div style="font-size: 3em; margin-bottom: 15px;">🔍</div>
        We could not match this to our database, <br>
        we believe that this is not recyclable.
      </div>
    `;
  }
  info.textContent = "Result Found";
}

function restart() {
  resultView.style.display = 'none';
  scannerView.style.display = 'block';
  scanBtn.disabled = false;
  info.textContent = "Ready to scan";
}

window.onload = startApp;
