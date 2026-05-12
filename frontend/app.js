(function () {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const fileCount = document.getElementById("file-count");
  const uploadBtn = document.getElementById("upload-btn");
  const statusPanel = document.getElementById("status-panel");
  const stepLabel = document.getElementById("step-label");
  const progressFill = document.getElementById("progress-fill");
  const progressPct = document.getElementById("progress-pct");
  const errorMsg = document.getElementById("error-msg");
  const doneActions = document.getElementById("done-actions");
  const viewBtn = document.getElementById("view-btn");
  const downloadBtn = document.getElementById("download-btn");
  const newBtn = document.getElementById("new-btn");
  const versionEl = document.getElementById("version");

  let selectedFiles = [];
  let pollTimer = null;

  // Fetch and display version
  fetch("/api/version")
    .then(r => r.json())
    .then(data => {
      versionEl.textContent = `v${data.version}`;
    })
    .catch(() => {
      versionEl.textContent = "version error";
    });

  // File selection
  fileInput.addEventListener("change", () => {
    selectedFiles = Array.from(fileInput.files);
    updateFileCount();
  });

  function updateFileCount() {
    if (selectedFiles.length === 0) {
      fileCount.textContent = "";
      uploadBtn.disabled = true;
      uploadBtn.textContent = "Select files to start";
      return;
    }
    const isVideo = selectedFiles.length === 1 && selectedFiles[0].type.startsWith("video/");
    if (isVideo) {
      fileCount.textContent = `1 video selected: ${selectedFiles[0].name}`;
    } else {
      fileCount.textContent = `${selectedFiles.length} image${selectedFiles.length !== 1 ? "s" : ""} selected`;
    }
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload & Reconstruct Room";
  }

  // Drag and drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  ["dragleave", "dragend"].forEach((evt) =>
    dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
  );

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    selectedFiles = Array.from(e.dataTransfer.files);
    updateFileCount();
  });

  // Upload
  uploadBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;
    await startUpload(selectedFiles);
  });

  async function startUpload(files) {
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";
    showStatus();

    const formData = new FormData();
    const isVideo = files.length === 1 && files[0].type.startsWith("video/");

    if (isVideo) {
      formData.append("video", files[0]);
    } else {
      files.forEach((f) => formData.append("files", f));
    }

    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      startPolling(data.job_id);
    } catch (err) {
      showError(err.message);
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Upload & Reconstruct Room";
    }
  }

  function showStatus() {
    statusPanel.removeAttribute("hidden");
    errorMsg.setAttribute("hidden", "");
    doneActions.setAttribute("hidden", "");
    newBtn.setAttribute("hidden", "");
    setProgress(0, "Uploading...");
  }

  function setProgress(pct, message) {
    progressFill.style.width = `${pct}%`;
    progressPct.textContent = `${pct}%`;
    stepLabel.textContent = message || "";
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.removeAttribute("hidden");
    newBtn.removeAttribute("hidden");
    stepLabel.textContent = "Reconstruction failed";
  }

  function startPolling(jobId) {
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        handleStatus(data, jobId);
      } catch (_) {
        // network blip — keep polling
      }
    }, 2000);
  }

  function handleStatus(data, jobId) {
    const { status, progress, message, error } = data;
    setProgress(progress, message);

    if (status === "done") {
      clearInterval(pollTimer);
      viewBtn.href = `/viewer/${jobId}`;
      downloadBtn.href = `/download/${jobId}`;
      doneActions.removeAttribute("hidden");
      newBtn.removeAttribute("hidden");
      uploadBtn.textContent = "Upload & Reconstruct Room";
    } else if (status === "failed") {
      clearInterval(pollTimer);
      showError(error || "Unknown error — check server logs");
    }
  }

  newBtn.addEventListener("click", () => {
    if (pollTimer) clearInterval(pollTimer);
    statusPanel.setAttribute("hidden", "");
    selectedFiles = [];
    fileInput.value = "";
    updateFileCount();
  });
})();
