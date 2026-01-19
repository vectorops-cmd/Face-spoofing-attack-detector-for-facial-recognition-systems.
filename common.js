const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const snapBtn = document.getElementById('snapBtn');
const previewImg = document.getElementById('previewImg');
const uploadBtn = document.getElementById('uploadBtn');
const uploadInput = document.getElementById('uploadInput');
const resultBox = document.getElementById('result');
const modelStatus = document.getElementById('modelStatus');

// Ping backend model
async function pingModel() {
  try {
    const r = await fetch('http://127.0.0.1:5000/api/ping');
    const j = await r.json();
    modelStatus.innerText = `${j.model_name || 'none'} (loaded=${j.model_loaded})`;
  } catch (e) {
    modelStatus.innerText = 'backend unreachable';
  }
}
pingModel();

// Start camera
startBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    startBtn.disabled = true;
  } catch {
    alert('Camera access denied or unavailable.');
  }
};

// Capture frame
snapBtn.onclick = async () => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const data = canvas.toDataURL('image/jpeg');
  previewImg.src = data;
  resultBox.innerHTML = "Analyzing...";
  await autoDetect(data, 'camera');
};

// Upload image
uploadBtn.onclick = () => uploadInput.click();
uploadInput.onchange = async (ev) => {
  const file = ev.target.files[0];
  const reader = new FileReader();
  reader.onload = async () => {
    previewImg.src = reader.result;
    resultBox.innerHTML = "Analyzing...";
    await autoDetect(reader.result, 'upload');
  };
  reader.readAsDataURL(file);
};

// Auto detect
async function autoDetect(src, sourceType) {
  try {
    const resp = await fetch('http://127.0.0.1:5000/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: src, source: sourceType })
    });
    const data = await resp.json();
    resultBox.innerHTML = `
      <div><strong>Result:</strong> ${data.prediction.toUpperCase()} (${data.attack_type})</div>
      <div><strong>Confidence:</strong> ${Math.round(data.confidence * 100)}%</div>
      <div><strong>Time:</strong> ${data.processing_time_ms} ms</div>`;
    appendRecent(data);
  } catch {
    resultBox.innerHTML = 'Detection failed. Is backend running?';
  }
}

// Append recent detection
function appendRecent(item) {
  const r = document.getElementById('recentList');
  const d = document.createElement('div');

  const img = document.createElement('img');
  img.src = item.image_path
    ? `http://127.0.0.1:5000/uploads/${item.image_path.split('/').pop()}`
    : previewImg.src;
  img.className = 'img-thumb';

  const info = document.createElement('div');
  info.innerHTML = `
    <div style="font-weight:700">${item.prediction.toUpperCase()} • ${item.attack_type}</div>
    <div style="color:#777;font-size:12px">${item.timestamp}</div>
    <div style="font-size:12px;color:#777">conf ${Math.round(item.confidence * 100)}% • ${item.processing_time_ms} ms</div>`;

  d.appendChild(img);
  d.appendChild(info);
  r.prepend(d);
}

// Load recent logs
(async () => {
  try {
    const logs = await fetch('http://127.0.0.1:5000/api/logs?limit=10');
    const j = await logs.json();
    j.rows.forEach(r => appendRecent(r));
  } catch {}
})();
