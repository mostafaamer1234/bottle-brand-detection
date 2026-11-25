// YOLO11n (python/train_yolo11.py, exported to ONNX) running in the browser.
const $ = (id) => document.getElementById(id);
const SIZE = 640, NAMES = ["Branded", "Generic"], CLS = ["branded", "generic"];
const SAMPLES = Array.from({ length: 20 }, (_, i) => `samples/IMG_${9091 + i}.jpg`);
let session = null, current = null, raw = null;

function letterbox(img) {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)";
  g.fillRect(0, 0, SIZE, SIZE);
  const r = SIZE / Math.max(img.width, img.height);
  g.drawImage(img, 0, 0, Math.round(img.width * r), Math.round(img.height * r));
  const px = g.getImageData(0, 0, SIZE, SIZE).data, n = SIZE * SIZE, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { t[i] = px[i * 4] / 255; t[n + i] = px[i * 4 + 1] / 255; t[2 * n + i] = px[i * 4 + 2] / 255; }
  return { tensor: new ort.Tensor("float32", t, [1, 3, SIZE, SIZE]), r };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

function decode(out, r, thr) {
  const d = out.data, N = 8400, boxes = [];
  for (let i = 0; i < N; i++) {
    const s0 = d[4 * N + i], s1 = d[5 * N + i], score = Math.max(s0, s1);
    if (score < thr) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    boxes.push({ x1: (cx - w / 2) / r, y1: (cy - h / 2) / r, x2: (cx + w / 2) / r, y2: (cy + h / 2) / r, score, cls: s0 >= s1 ? 0 : 1 });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every((k) => iou(k, b) < 0.45)) keep.push(b);
  return keep;
}

function draw(img, dets) {
  const cv = $("view"), g = cv.getContext("2d");
  cv.width = img.width; cv.height = img.height;
  g.drawImage(img, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const lw = Math.max(3, img.width / 130), fs = Math.max(18, img.width / 22);
  dets.forEach((b) => {
    const col = css.getPropertyValue(`--${CLS[b.cls]}`).trim();
    g.strokeStyle = col; g.lineWidth = lw;
    g.beginPath(); g.roundRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, lw * 2); g.stroke();
    const label = `${NAMES[b.cls]} ${Math.round(b.score * 100)}%`;
    g.font = `600 ${fs}px -apple-system, BlinkMacSystemFont, Inter, sans-serif`;
    const tw = g.measureText(label).width + fs * 0.9, th = fs * 1.6, ty = Math.max(0, b.y1 - th - lw);
    g.fillStyle = col; g.beginPath(); g.roundRect(b.x1, ty, tw, th, th / 2); g.fill();
    g.fillStyle = "#fff"; g.fillText(label, b.x1 + fs * 0.45, ty + th * 0.7);
  });
  $("dets").innerHTML = dets.length
    ? dets.map((b) => `<div class="det"><b class="${CLS[b.cls]}">${NAMES[b.cls]}</b><span>${Math.round(b.score * 100)}%</span></div>`).join("")
    : `<p class="q-small">No bottle above the threshold.</p>`;
}

async function detect(img) {
  current = img;
  if (!session) { draw(img, []); return; }
  const { tensor, r } = letterbox(img);
  const t0 = performance.now();
  const out = await session.run({ images: tensor });
  $("speed").textContent = `Inference took ${Math.round(performance.now() - t0)} ms on this device.`;
  raw = { out: out.output0, r };
  redraw();
}

function redraw() {
  if (!current || !raw) return;
  draw(current, decode(raw.out, raw.r, $("thr").value / 100));
}

function loadUrl(url) {
  const img = new Image();
  img.onload = () => detect(img);
  img.src = url;
}

$("strip").innerHTML = SAMPLES.map((s, i) => `<button aria-pressed="${i === 0}" aria-label="Test photo ${i + 1}"><img src="${s}" alt="" loading="lazy" /></button>`).join("");
$("strip").querySelectorAll("button").forEach((b, i) => b.addEventListener("click", () => {
  $("strip").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  loadUrl(SAMPLES[i]);
}));
$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("strip").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
  loadUrl(URL.createObjectURL(f));
});
$("thr").addEventListener("input", () => { $("thr-v").textContent = `${$("thr").value}%`; redraw(); });

loadUrl(SAMPLES[0]);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.InferenceSession.create("bottles.onnx", { executionProviders: ["wasm"] }).then((s) => {
  session = s;
  $("loading").remove();
  if (current) detect(current);
}).catch((err) => { $("loading").textContent = `Could not load the model: ${err.message}`; });
