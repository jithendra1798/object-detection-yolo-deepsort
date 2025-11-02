const socket = io();

const videoInput = document.getElementById('videoInput');
const uploadBtn = document.getElementById('uploadBtn');
const status = document.getElementById('status');
const originalVideo = document.getElementById('originalVideo');
const processedVideo = document.getElementById('processedVideo');
const progressBar = document.querySelector('#progressBar > i');
const percentText = document.getElementById('percent');

let uploadedFilename = null;
let processingUrl = null;
let originalPreviewUrl = null;

const MAX_PROCESSED_RETRIES = 20;
const RETRY_DELAY_MS = 1500;

function resetProgress() {
  progressBar.style.width = '0%';
  percentText.innerText = '0%';
}

function disableUiWhileProcessing(disabled) {
  uploadBtn.disabled = disabled;
  videoInput.disabled = disabled;
}

function showOriginalPreview(file, serverUrl) {
  if (originalPreviewUrl) {
    URL.revokeObjectURL(originalPreviewUrl);
    originalPreviewUrl = null;
  }

  if (file) {
    originalPreviewUrl = URL.createObjectURL(file);
    originalVideo.src = originalPreviewUrl;
  } else if (serverUrl) {
    originalVideo.src = `${serverUrl}${serverUrl.includes('?') ? '&' : '?'}t=${Date.now()}`;
  }

  originalVideo.load();
  originalVideo.onloadeddata = () => {
    // Attempt to show the first frame; autoplay may be blocked so ignore failures.
    originalVideo.play().catch(() => {});
  };
}

function loadProcessedVideo(url, attempt = 0) {
  const targetUrl = url || processingUrl;
  if (!targetUrl) {
    status.innerText = 'Processed file not yet available; retrying...';
    return;
  }

  const cacheBustedUrl = `${targetUrl}${targetUrl.includes('?') ? '&' : '?'}t=${Date.now()}`;
  processedVideo.onerror = null;
  processedVideo.onloadeddata = null;
  processedVideo.src = cacheBustedUrl;
  processedVideo.load();

  processedVideo.onloadeddata = () => {
    status.innerText = 'Processing done';
    disableUiWhileProcessing(false);
    processedVideo.play().catch(() => {});
  };

  processedVideo.onerror = () => {
    if (attempt + 1 >= MAX_PROCESSED_RETRIES) {
      status.innerText = 'Processed file not yet available; please try again later.';
      disableUiWhileProcessing(false);
      return;
    }
    setTimeout(() => loadProcessedVideo(targetUrl, attempt + 1), RETRY_DELAY_MS);
  };
}

uploadBtn.addEventListener('click', async () => {
  if (!videoInput.files || videoInput.files.length === 0) {
    alert('Choose a video first');
    return;
  }

  const file = videoInput.files[0];
  const form = new FormData();
  form.append('video', file);

  status.innerText = 'Uploading...';
  disableUiWhileProcessing(true);
  resetProgress();
  processingUrl = null;
  processedVideo.removeAttribute('src');
  processedVideo.load();

  showOriginalPreview(file);

  let res;
  try {
    res = await fetch('/upload', { method: 'POST', body: form });
  } catch (err) {
    status.innerText = 'Upload failed';
    disableUiWhileProcessing(false);
    return;
  }

  if (!res.ok) {
    status.innerText = 'Upload failed';
    disableUiWhileProcessing(false);
    return;
  }

  const j = await res.json();
  uploadedFilename = j.filename;
  status.innerText = 'Upload complete. Starting processing...';
  socket.emit('start_processing', { filename: uploadedFilename });
});

socket.on('processing_started', (data) => {
  processingUrl = data && data.processing_url ? data.processing_url : null;
  status.innerText = 'Processing started';
  resetProgress();
});

socket.on('processing_progress', (data) => {
  const value = Number(data && data.percent ? data.percent : 0);
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  progressBar.style.width = `${clamped}%`;
  percentText.innerText = `${clamped}%`;
});

socket.on('processing_done', (data) => {
  progressBar.style.width = '100%';
  percentText.innerText = '100%';
  const url = data && data.output ? `/processed/${data.output}` : processingUrl;
  loadProcessedVideo(url);
});

socket.on('processing_error', (data) => {
  status.innerText = `Error: ${data && data.error ? data.error : 'unknown'}`;
  disableUiWhileProcessing(false);
});

socket.on('connect_error', () => {
  status.innerText = 'Socket connection error. Please refresh the page.';
  disableUiWhileProcessing(false);
});
