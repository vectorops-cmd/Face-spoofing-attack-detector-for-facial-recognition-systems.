// frontend/app.js — GitHub Pages + Render compatible

const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const snapBtn = document.getElementById('snapBtn');
const previewImg = document.getElementById('previewImg');
const uploadBtn = document.getElementById('uploadBtn');
const uploadInput = document.getElementById('uploadInput');
const resultBox = document.getElementById('result');
const recentList = document.getElementById('recentList');

// ✅ MUST be Render backend (NOT localhost)
const BACKEND_BASE = "https://face-spoofing-attack-detector-for-facial.onrender.com";
const DETECT_ENDPOINT = `${BACKEND_BASE}/api/detect`;

// -------------------- CAMERA --------------------
startBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    startBtn.disabled = true;
    resultBox.innerText = "Camera started.";
  } catch (err) {
    alert("Camera access denied. Use HTTPS or allow permission.");
    console.error(err);
  }
};

// -------------------- SNAP --------------------
snapBtn.onclick = async () => {
  if (!video.srcObject) {
    resultBox.innerText = "Start the camera first.";
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);

  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.9));
  previewImg.src = URL.createObjectURL(blob);

  resultBox.innerText = "Analyzing...";
  await sendToBackend(blob);
};

// -------------------- UPLOAD --------------------
uploadBtn.onclick = () => uploadInput.click();

uploadInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  previewImg.src = URL.createObjectURL(file);
  resultBox.innerText = "Analyzing...";
  await sendToBackend(file);
};

// -------------------- API CALL --------------------
async function sendToBackend(blob) {
  try {
    const formData = new FormData();
    formData.append("image", blob, "image.jpg");

    const resp = await fetch(DETECT_ENDPOINT, {
      method: "POST",
      body: formData
    });

    if (!resp.ok) {
      const text = await resp.text();
      resultBox.innerHTML = `<span style="color:red">Server error:</span><pre>${escapeHtml(text)}</pre>`;
      return;
    }

    const result = await resp.json();

    const label = (result.prediction || "unknown").toUpperCase();
    const conf = Math.round((result.confidence || 0) * 100);
    const time = result.processing_time_ms ?? "N/A";

    resultBox.innerHTML = `
      <div><b>Prediction:</b> ${label}</div>
      <div><b>Confidence:</b> ${conf}%</div>
      <div><b>Time:</b> ${time} ms</div>
    `;

    appendRecent(label, conf, previewImg.src);

  } catch (err) {
    console.error(err);
    resultBox.innerText = "Network error. Backend unreachable.";
  }
}

// -------------------- RECENT LIST --------------------
function appendRecent(label, conf, imgSrc) {
  const d = document.createElement("div");
  d.style.display = "flex";
  d.style.gap = "8px";

  d.innerHTML = `
    <img src="${imgSrc}" width="80" style="border-radius:6px"/>
    <div>
      <div><b>${label}</b></div>
      <div style="font-size:12px">${conf}%</div>
    </div>
  `;

  recentList.prepend(d);
  while (recentList.children.length > 10) recentList.removeChild(recentList.lastChild);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])
  );
}
