// frontend/app-fixed.js
// Replaces your previous script: uses endpoint /api/detect and form key "image",
// checks HTTP status, shows server error bodies, updates recent list.

const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const snapBtn = document.getElementById('snapBtn');
const previewImg = document.getElementById('previewImg');
const uploadBtn = document.getElementById('uploadBtn');
const uploadInput = document.getElementById('uploadInput');
const resultBox = document.getElementById('result');
const recentList = document.getElementById('recentList');

// Backend endpoint — adjust if your backend runs on a different host/port
const BACKEND_BASE = "http://127.0.0.1:5000";
const DETECT_ENDPOINT = `${BACKEND_BASE}/api/detect`; // <- backend expects /api/detect

// Start camera
startBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    startBtn.disabled = true;
    resultBox.innerText = "Camera started.";
  } catch (err) {
    alert('Camera access denied or unavailable. Check permissions or use HTTPS/localhost.');
    console.error(err);
  }
};

// Capture frame
snapBtn.onclick = async () => {
  if (!video.srcObject) {
    resultBox.innerText = "Start the camera first.";
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  previewImg.src = URL.createObjectURL(blob);
  resultBox.innerText = "Analyzing...";
  await sendToBackend(blob);
};

// Upload image
uploadBtn.onclick = () => uploadInput.click();
uploadInput.onchange = async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  previewImg.src = URL.createObjectURL(file);
  resultBox.innerText = "Analyzing...";
  await sendToBackend(file);
};

// Send captured/uploaded image to backend
async function sendToBackend(fileBlob) {
  try {
    const formData = new FormData();
    // IMPORTANT: backend expects key "image"
    formData.append("image", fileBlob, "image.jpg");

    const resp = await fetch(DETECT_ENDPOINT, {
      method: "POST",
      body: formData
    });

    // If backend returns HTML error (500) or any non-JSON, show raw text
    if (!resp.ok) {
      const text = await resp.text();
      console.error("Server error body:", text);
      resultBox.innerHTML = `<span style="color:red">Server error ${resp.status}:</span><pre style="white-space:pre-wrap">${escapeHtml(text)}</pre>`;
      return;
    }

    // parse JSON
    const result = await resp.json();

    // Basic defensive checks
    if (result.error) {
      resultBox.innerHTML = `<span style="color:red">${escapeHtml(result.error)}</span>`;
      return;
    }

    // The backend might use keys: prediction/confidence or label/confidence — handle both
    const label = (result.prediction || result.label || "unknown").toString();
    const confidence = (typeof result.confidence === "number") ? (result.confidence * 100).toFixed(1) + "%" : (result.confidence || "N/A");
    const timeMs = result.processing_time_ms ?? result.processing_time ?? "N/A";

    resultBox.innerHTML = `
      <div><strong>Prediction:</strong> ${escapeHtml(label.toUpperCase())}</div>
      <div><strong>Confidence:</strong> ${escapeHtml(String(confidence))}</div>
      <div><strong>Processing time:</strong> ${escapeHtml(String(timeMs))} ms</div>
    `;

    appendRecent({
      label,
      confidence: (typeof result.confidence === "number") ? result.confidence : 0,
      timestamp: new Date().toLocaleString()
    }, previewImg.src);

  } catch (err) {
    console.error("Network / parse error:", err);
    resultBox.innerText = "Detection failed. Is backend running? See console for details.";
  }
}

// Add recent detections list
function appendRecent(item, imgSrc) {
  const d = document.createElement('div');
  d.style.display = 'flex';
  d.style.alignItems = 'center';
  d.style.gap = '8px';
  d.style.marginBottom = '5px';

  const img = document.createElement('img');
  img.src = imgSrc;
  img.width = 80;
  img.height = 60;
  img.style.objectFit = 'cover';
  img.style.borderRadius = '6px';

  const info = document.createElement('div');
  info.innerHTML = `
    <div style="font-weight:700">${escapeHtml(item.label.toUpperCase())}</div>
    <div style="font-size:12px;color:#777">${escapeHtml(item.timestamp)}</div>
    <div style="font-size:12px;color:#777">conf ${(item.confidence*100).toFixed(1)}%</div>
  `;

  d.appendChild(img);
  d.appendChild(info);
  recentList.prepend(d);

  // keep last 10
  while (recentList.children.length > 10) recentList.removeChild(recentList.lastChild);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
