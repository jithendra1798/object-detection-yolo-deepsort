const socket = io();

const videoInput = document.getElementById('videoInput');
const uploadBtn = document.getElementById('uploadBtn');
const status = document.getElementById('status');
const originalVideo = document.getElementById('originalVideo');
const processedVideo = document.getElementById('processedVideo');
const progressBar = document.querySelector('#progressBar > i');
const percentText = document.getElementById('percent');

let uploadedFilename = null;

uploadBtn.addEventListener('click', async () => {
  if (!videoInput.files || videoInput.files.length === 0) {
    alert('Choose a video first');
    return;
  }
  const file = videoInput.files[0];
  const form = new FormData();
  form.append('video', file);
  status.innerText = 'Uploading...';
  const res = await fetch('/upload', { method: 'POST', body: form });
  if (!res.ok) {
    status.innerText = 'Upload failed';
    return;
  }
  const j = await res.json();
  uploadedFilename = j.filename;
  originalVideo.src = j.url;
  status.innerText = 'Upload complete. Starting processing...';
  socket.emit('start_processing', { filename: uploadedFilename });
});

socket.on('processing_started', (d) => {
  status.innerText = 'Processing started';
});
socket.on('processing_progress', (d) => {
  const p = d.percent || 0;
  progressBar.style.width = p + '%';
  percentText.innerText = p + '%';
});
socket.on('processing_done', (d) => {
  status.innerText = 'Processing done';
  if (d.output) {
    const url = '/processed/' + d.output;
    // ensure the file is available (server may still be finalizing)
    fetch(url, { method: 'HEAD' }).then(res => {
      if (res.ok) {
        // create a source element with explicit type to help browsers
        processedVideo.innerHTML = '';
        const src = document.createElement('source');
        src.src = url;
        src.type = 'video/mp4';
        processedVideo.appendChild(src);
        processedVideo.load();
      } else {
        status.innerText = 'Processed file not yet available; try again in a few seconds.';
      }
    }).catch(err => {
      status.innerText = 'Error fetching processed file: ' + err;
    });
  }
});
socket.on('processing_error', (d) => {
  status.innerText = 'Error: ' + (d.error || 'unknown');
});
