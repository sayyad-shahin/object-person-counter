(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const post = async (url, body) => {
    const r = await fetch(url, body instanceof FormData ? { method: "POST", body }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    return d;
  };
  const showError = (m) => { const e = $("error"); e.textContent = m || ""; e.classList.toggle("hidden", !m); };

  function chosenClasses() {
    const boxes = [...document.querySelectorAll("#classes input:checked")].map((b) => b.value);
    return boxes.includes("all") ? [] : boxes;
  }

  async function start() {
    showError("");
    const btn = $("startBtn"); btn.disabled = true;
    try {
      let source = "webcam:" + $("cam").value;
      if (document.querySelector("input[name=src]:checked").value === "file") {
        const f = $("file").files[0];
        if (!f) throw new Error("Choose a video file first.");
        const fd = new FormData(); fd.append("video", f);
        source = "upload:" + (await post("/api/upload", fd)).upload_id;
      }
      await post("/api/start", { source, conf: +$("conf").value, classes: chosenClasses(), line: $("line").value, flip: $("flip").checked });
    } catch (e) { showError(e.message); } finally { btn.disabled = false; }
  }

  function spark(values) {
    const c = $("spark"), g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    if (values.length < 2) return;
    const max = Math.max(3, ...values), step = c.width / (values.length - 1);
    g.strokeStyle = "#4cc9f0"; g.lineWidth = 2; g.beginPath();
    values.forEach((v, i) => { const x = i * step, y = c.height - 6 - (v / max) * (c.height - 12); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
  }

  function render(s) {
    const st = $("status"); st.textContent = s.status; st.className = "pill " + (s.status === "running" ? "running" : s.status === "error" ? "error" : "");
    $("fps").textContent = s.status === "running" ? `${s.fps} FPS${s.source ? " · " + s.source : ""}` : (s.source || "");
    if (s.error) showError(s.error);
    $("tNow").textContent = s.total_current; $("tSeen").textContent = s.total_unique;
    $("tIn").textContent = s.line.in_total; $("tOut").textContent = s.line.out_total;
    const labels = [...new Set([...Object.keys(s.current), ...Object.keys(s.unique), ...Object.keys(s.peak)])];
    const rows = $("rows"); rows.innerHTML = "";
    labels.sort((a, b) => (s.unique[b] || 0) - (s.unique[a] || 0)).forEach((l) => {
      const tr = document.createElement("tr");
      [l, s.current[l] || 0, s.unique[l] || 0, s.peak[l] || 0].forEach((v) => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
      rows.appendChild(tr);
    });
    if (!labels.length) { const tr = document.createElement("tr"), td = document.createElement("td"); td.colSpan = 4; td.className = "muted"; td.textContent = "Nothing detected yet."; tr.appendChild(td); rows.appendChild(tr); }
    const ev = $("events"); ev.innerHTML = "";
    if (!s.line.enabled) { const li = document.createElement("li"); li.className = "muted"; li.textContent = "Counting line is off."; ev.appendChild(li); }
    s.events.forEach((e) => { const li = document.createElement("li"); li.className = e.direction; li.textContent = `${e.direction.toUpperCase()} · ${e.label} #${e.id}`; ev.appendChild(li); });
    spark(s.history);
  }

  async function poll() {
    try { render(await (await fetch("/api/stats")).json()); } catch (_) { /* server busy */ }
    setTimeout(poll, 600);
  }

  $("startBtn").addEventListener("click", start);
  $("stopBtn").addEventListener("click", () => post("/api/stop").catch((e) => showError(e.message)));
  $("resetBtn").addEventListener("click", () => post("/api/reset").catch((e) => showError(e.message)));
  $("conf").addEventListener("input", () => { $("confVal").textContent = (+$("conf").value).toFixed(2); });
  document.querySelectorAll("#classes input").forEach((b) => b.addEventListener("change", () => {
    const all = document.querySelector("#classes input[value=all]");
    if (b.value === "all" && b.checked) document.querySelectorAll("#classes input:not([value=all])").forEach((x) => { x.checked = false; });
    else if (b.checked) all.checked = false;
    if (!document.querySelector("#classes input:checked")) all.checked = true;
  }));
  poll();
})();
