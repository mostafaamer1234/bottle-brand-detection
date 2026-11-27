// Bottle Detector camera: YOLO11n (python/train_yolo11.py, exported to ONNX) on live video or photos,
// in the browser with ONNX Runtime Web. Letterboxing and non-max suppression match the about page.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const SIZE = 640, NAMES = ["Branded", "Generic"], COLORS = ["#ff375f", "#30d158"];
  const SAMPLES = Array.from({ length: 20 }, (_, i) => `samples/IMG_${9091 + i}.jpg`);
  let session = null, source = null, raw = null, live = false, stream = null, facing = "environment", busy = false;
  const lb = document.createElement("canvas"); lb.width = lb.height = SIZE;
  const lbg = lb.getContext("2d", { willReadFrequently: true });

  function letterbox(el, w, h) {
    lbg.fillStyle = "rgb(114,114,114)"; lbg.fillRect(0, 0, SIZE, SIZE);
    const r = SIZE / Math.max(w, h);
    lbg.drawImage(el, 0, 0, Math.round(w * r), Math.round(h * r));
    const px = lbg.getImageData(0, 0, SIZE, SIZE).data, n = SIZE * SIZE, t = new Float32Array(3 * n);
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

  function fit() {
    const cv = $("view"), f = $("finder");
    if (!cv.width) return;
    const s = Math.min(f.clientWidth / cv.width, f.clientHeight / cv.height);
    cv.style.width = `${cv.width * s}px`; cv.style.height = `${cv.height * s}px`;
  }
  function draw(el, w, h, dets) {
    const cv = $("view"), g = cv.getContext("2d");
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; fit(); }
    g.drawImage(el, 0, 0, w, h);
    const lw = Math.max(3, w / 150), fs = Math.max(16, w / 26);
    for (const b of dets) {
      const col = COLORS[b.cls];
      g.strokeStyle = col; g.lineWidth = lw;
      const seg = Math.min(b.x2 - b.x1, b.y2 - b.y1) * 0.22;
      g.beginPath(); // focus-bracket corners, like the viewfinder
      [[b.x1, b.y1, 1, 1], [b.x2, b.y1, -1, 1], [b.x1, b.y2, 1, -1], [b.x2, b.y2, -1, -1]].forEach(([x, y, sx, sy]) => { g.moveTo(x + sx * seg, y); g.lineTo(x, y); g.lineTo(x, y + sy * seg); });
      g.stroke();
      g.globalAlpha = 0.12; g.fillStyle = col; g.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1); g.globalAlpha = 1;
      const label = `${NAMES[b.cls].toUpperCase()} ${Math.round(b.score * 100)}%`;
      g.font = `700 ${fs}px "Space Grotesk", sans-serif`;
      const tw = g.measureText(label).width + fs, th = fs * 1.7, ty = Math.max(0, b.y1 - th - lw);
      g.fillStyle = col; g.beginPath(); g.roundRect(b.x1, ty, tw, th, th / 2); g.fill();
      g.fillStyle = "#fff"; g.fillText(label, b.x1 + fs / 2, ty + th * 0.68);
    }
    const n = [0, 1].map((c) => dets.filter((d) => d.cls === c).length);
    $("tally").innerHTML = dets.length ? n.map((k, c) => (k ? `<span><i style="background:${COLORS[c]}"></i>${k} ${NAMES[c].toLowerCase()}</span>` : "")).join("") : `<span>No bottle in view</span>`;
  }

  async function infer(el, w, h) {
    const { tensor, r } = letterbox(el, w, h);
    const t0 = performance.now();
    const out = await session.run({ images: tensor });
    return { out: out.output0, r, ms: performance.now() - t0 };
  }

  async function photo(url) {
    stopLive();
    const img = new Image();
    img.onload = async () => {
      source = img;
      if (!session) { draw(img, img.width, img.height, []); return; }
      raw = await infer(img, img.width, img.height);
      $("fps").textContent = `${Math.round(raw.ms)} MS`;
      redraw();
    };
    img.src = url;
  }
  function redraw() {
    if (live || !source || !raw) return;
    draw(source, source.width, source.height, decode(raw.out, raw.r, $("thr").value / 100));
  }

  async function startLive() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { $("status").textContent = "This browser can't open the camera. Use a photo instead."; return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
    } catch (err) {
      $("status").textContent = err.name === "NotAllowedError" ? "Camera permission was declined. Use a photo instead." : "No camera available. Use a photo instead.";
      return;
    }
    const v = $("video");
    v.srcObject = stream; await v.play();
    live = true; $("live").setAttribute("aria-pressed", "true"); $("flip").disabled = false; $("mode-tag").textContent = "LIVE";
    document.querySelectorAll("#roll button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    $("status").textContent = "";
    loop();
  }
  async function loop() {
    if (!live) return;
    const v = $("video"), w = v.videoWidth, h = v.videoHeight;
    if (w && session && !busy) {
      busy = true;
      try {
        const res = await infer(v, w, h);
        if (live) { draw(v, w, h, decode(res.out, res.r, $("thr").value / 100)); $("fps").textContent = `${(1000 / Math.max(1, res.ms)).toFixed(1)} FPS`; }
      } finally { busy = false; }
    }
    requestAnimationFrame(loop);
  }
  function stopLive() {
    live = false;
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    $("live").setAttribute("aria-pressed", "false"); $("flip").disabled = true; $("mode-tag").textContent = "PHOTO";
  }

  $("roll").innerHTML = SAMPLES.map((s, i) => `<button type="button" aria-pressed="${i === 0}" aria-label="Sample photo ${i + 1}"><img src="${s}" alt="" loading="lazy" /></button>`).join("");
  $("roll").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll("#roll button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    photo(SAMPLES[[...b.parentNode.children].indexOf(b)]);
  });
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    document.querySelectorAll("#roll button").forEach((x) => x.setAttribute("aria-pressed", "false"));
    photo(URL.createObjectURL(f)); e.target.value = "";
  });
  $("live").addEventListener("click", () => (live ? (stopLive(), source && redraw()) : startLive()));
  $("flip").addEventListener("click", () => { facing = facing === "environment" ? "user" : "environment"; stopLive(); startLive(); });
  $("thr").addEventListener("input", () => { $("thr-v").textContent = `${$("thr").value}%`; redraw(); });
  window.addEventListener("resize", fit);
  document.addEventListener("visibilitychange", () => { if (document.hidden && live) stopLive(); });

  photo(SAMPLES[0]);
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.InferenceSession.create("bottles.onnx", { executionProviders: ["wasm"] }).then((s) => {
    session = s;
    $("status").textContent = "Tap the shutter for the live camera";
    setTimeout(() => { if ($("status").textContent.startsWith("Tap")) $("status").textContent = ""; }, 4000);
    if (source && !live) photo(source.src);
  }).catch((err) => { $("status").textContent = `Could not load the model: ${err.message}`; });
})();
