document.addEventListener('DOMContentLoaded', () => {
    // ── STATE ──
    let currentFile = null;
    let isRecording = false;
    let isTranscribing = false;
    let transcriptText = '';
    let translationText = '';
    let mediaRecorder = null;
    let recordChunks = [];
    let audioContext = null;
    let segments = [];

    // DOM Elements
    const elements = {
        langSelect: document.getElementById('langSelect'),
        transSelect: document.getElementById('transSelect'),
        beamSelect: document.getElementById('beamSelect'),
        tempInput: document.getElementById('tempInput'),
        dropZone: document.getElementById('dropZone'),
        fileInput: document.getElementById('fileInput'),
        dropFilename: document.getElementById('dropFilename'),
        dropHint: document.getElementById('dropHint'),
        refText: document.getElementById('refText'),
        recordBtn: document.getElementById('recordBtn'),
        recordLabel: document.getElementById('recordLabel'),
        transcribeBtn: document.getElementById('transcribeBtn'),
        transcribeIcon: document.getElementById('transcribeIcon'),
        transcribeLabel: document.getElementById('transcribeLabel'),
        werDisplay: document.getElementById('werDisplay'),
        cerDisplay: document.getElementById('cerDisplay'),
        durationDisplay: document.getElementById('durationDisplay'),
        confDisplay: document.getElementById('confDisplay'),
        werDetail: document.getElementById('werDetail'),
        werBig: document.getElementById('werBig'),
        cerBig: document.getElementById('cerBig'),
        werBar: document.getElementById('werBar'),
        cerBar: document.getElementById('cerBar'),
        transcriptPlaceholder: document.getElementById('transcriptPlaceholder'),
        transcriptBody: document.getElementById('transcriptBody'),
        translationPanel: document.getElementById('translationPanel'),
        translationBody: document.getElementById('translationBody'),
        waveformStrip: document.getElementById('waveformStrip'),
        totalTime: document.getElementById('totalTime'),
        statusBadge: document.getElementById('statusBadge'),
        statusText: document.getElementById('statusText'),
        copyBtn: document.getElementById('copyBtn'),
        copyTransBtn: document.getElementById('copyTransBtn'),
        clearBtn: document.getElementById('clearBtn'),
        settingsToggle: document.getElementById('settingsToggle'),
        settingsDrawer: document.getElementById('settingsDrawer'),
        settingsChevron: document.getElementById('settingsChevron'),
        deviceLabel: document.getElementById('deviceLabel'),
        exportTxt: document.getElementById('exportTxt'),
        exportSrt: document.getElementById('exportSrt'),
        exportJson: document.getElementById('exportJson')
    };

    // ── FILE HANDLING ──
    function handleFile(file) {
        if (!file) return;
        currentFile = file;
        elements.dropZone.classList.add('has-file');
        document.getElementById('dropIcon').textContent = '🎵';
        elements.dropFilename.textContent = file.name;
        elements.dropHint.textContent = formatBytes(file.size) + ' · ' + file.type.split('/')[1]?.toUpperCase();
        showToast('📁', file.name + ' loaded');

        elements.waveformStrip.style.display = 'flex';
        drawStaticWaveform();

        const url = URL.createObjectURL(file);
        const audio = new Audio(url);
        audio.addEventListener('loadedmetadata', () => {
            elements.totalTime.textContent = formatTime(audio.duration);
            elements.durationDisplay.textContent = formatTime(audio.duration);
        });
    }

    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.style.borderColor = 'var(--accent)';
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    function formatBytes(b) {
        if (b > 1e6) return (b / 1e6).toFixed(1) + ' MB';
        return (b / 1e3).toFixed(0) + ' KB';
    }

    function formatTime(s) {
        if (!s || isNaN(s)) return '0:00';
        const m = Math.floor(s / 60), sec = Math.floor(s % 60);
        return m + ':' + String(sec).padStart(2, '0');
    }

    // ── WAVEFORM ──
    function drawStaticWaveform() {
        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.parentElement.offsetWidth;
        const H = 36;
        canvas.width = W; canvas.height = H;

        const bars = Math.floor(W / 4);
        ctx.clearRect(0, 0, W, H);
        for (let i = 0; i < bars; i++) {
            const h = (Math.sin(i * 0.3) * 0.4 + Math.random() * 0.6) * H * 0.85;
            const x = i * 4;
            const gradient = ctx.createLinearGradient(0, H / 2 - h / 2, 0, H / 2 + h / 2);
            gradient.addColorStop(0, 'rgba(110,86,255,0.7)');
            gradient.addColorStop(1, 'rgba(0,229,192,0.4)');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, H / 2 - h / 2, 2, h);
        }
    }

    // ── RECORDING & SILENCE DETECTION ──
    let silenceTimer = null;
    let audioStream = null;
    let animationId = null;

    elements.recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(audioStream);
                recordChunks = [];

                // Initialize Audio Context for Silence Detection
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(audioStream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 2048;
                source.connect(analyser);

                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);

                let lastVoiceTime = Date.now();
                const SILENCE_THRESHOLD = 15; // 15 seconds
                const VOLUME_THRESHOLD = 0.015; // Adjust based on testing

                const checkSilence = () => {
                    if (!isRecording) return;

                    analyser.getByteFrequencyData(dataArray);

                    // Calculate RMS Volume
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i] * dataArray[i];
                    }
                    const rms = Math.sqrt(sum / bufferLength) / 255.0;

                    const now = Date.now();
                    if (rms > VOLUME_THRESHOLD) {
                        lastVoiceTime = now;
                        elements.statusText.textContent = 'Listening...';
                    } else {
                        const silentSecs = (now - lastVoiceTime) / 1000;
                        if (silentSecs > 1) {
                            elements.statusText.textContent = `Quiet... (${Math.ceil(SILENCE_THRESHOLD - silentSecs)}s)`;
                        }

                        if (silentSecs >= SILENCE_THRESHOLD) {
                            showToast('⚡', 'Auto-transcribing due to silence...');
                            elements.recordBtn.click(); // Stop recording
                            return;
                        }
                    }
                    animationId = requestAnimationFrame(checkSilence);
                };

                mediaRecorder.ondataavailable = e => recordChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(recordChunks, { type: 'audio/webm' });
                    const file = new File([blob], 'recording-' + Date.now() + '.webm', { type: 'audio/webm' });
                    handleFile(file);

                    // Cleanup
                    if (audioStream) audioStream.getTracks().forEach(t => t.stop());
                    if (audioContext) audioContext.close();
                    cancelAnimationFrame(animationId);

                    // Auto-transcribe live recording
                    if (file) {
                        // Force English translation as requested
                        elements.transSelect.value = 'English';
                        elements.transcribeBtn.click();
                    }
                };

                mediaRecorder.start();
                isRecording = true;
                elements.recordBtn.classList.add('recording');
                elements.recordLabel.textContent = 'Stop · 0:00';
                setStatus('ready');
                elements.statusText.textContent = 'Listening...';

                let secs = 0;
                window._recTimer = setInterval(() => {
                    secs++;
                    elements.recordLabel.textContent = 'Stop · ' + formatTime(secs);
                }, 1000);

                checkSilence();
                showToast('🔴', 'Recording started...');
            } catch (e) {
                console.error(e);
                showToast('⚠', 'Microphone access denied');
            }
        } else {
            isRecording = false;
            clearInterval(window._recTimer);
            if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            elements.recordBtn.classList.remove('recording');
            elements.recordLabel.textContent = 'Record Live';
            showToast('✓', 'Recording stopped');
        }
    });

    // ── TRANSCRIBE ──
    elements.transcribeBtn.addEventListener('click', async () => {
        if (!currentFile || isTranscribing) return;
        isTranscribing = true;

        elements.transcribeBtn.classList.add('loading');
        elements.transcribeIcon.innerHTML = '<span class="spin">◌</span>';
        elements.transcribeLabel.textContent = 'Processing...';
        setStatus('processing');

        const formData = new FormData();
        formData.append('audio', currentFile);
        formData.append('language', elements.langSelect.value);
        formData.append('target_language', elements.transSelect.value);
        formData.append('reference_text', elements.refText.value);
        formData.append('beam_size', elements.beamSelect.value);
        formData.append('temperature', elements.tempInput.value);

        const tsActive = document.querySelector('#tsToggle .active');
        if (tsActive) formData.append('timestamps', tsActive.dataset.val);

        try {
            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.error) throw new Error(data.error);

            transcriptText = data.transcription;
            translationText = data.translation;
            renderTranscript(transcriptText);

            if (translationText) {
                elements.translationPanel.classList.remove('hidden');
                elements.translationBody.textContent = translationText;
            } else {
                elements.translationPanel.classList.add('hidden');
            }

            // Metrics
            updateMetric('wer', data.wer);
            updateMetric('cer', data.cer);
            elements.durationDisplay.textContent = formatTime(data.duration);
            elements.confDisplay.textContent = data.confidence.toFixed(2);
            elements.confDisplay.className = 'metric-value accent';

            if (data.wer !== null) showWerDetail(data.wer / 100, data.cer / 100);

            setStatus('ready');
            showToast('⚡', 'Complete');
        } catch (e) {
            console.error(e);
            showToast('⚠', 'Transcription failed: ' + e.message);
            setStatus('idle');
        } finally {
            isTranscribing = false;
            elements.transcribeBtn.classList.remove('loading');
            elements.transcribeIcon.textContent = '⚡';
            elements.transcribeLabel.textContent = 'Transcribe';
        }
    });

    function updateMetric(id, val) {
        const el = document.getElementById(id + 'Display');
        if (val === null) {
            el.textContent = '—';
            el.className = 'metric-value muted';
        } else {
            el.textContent = val.toFixed(1) + '%';
            el.className = 'metric-value ' + (val < 10 ? 'accent' : 'warn');
        }
    }

    function renderTranscript(text) {
        if (elements.transcriptPlaceholder) elements.transcriptPlaceholder.remove();
        elements.transcriptBody.innerHTML = text.split(' ').map((w, i) =>
            `<span class="word" data-idx="${i}">${w} </span>`
        ).join('');
    }

    function showWerDetail(wer, cer) {
        elements.werDetail.classList.add('visible');
        elements.werBig.textContent = (wer * 100).toFixed(1) + '%';
        elements.cerBig.textContent = (cer * 100).toFixed(1) + '%';
        setTimeout(() => {
            elements.werBar.style.width = (wer * 100) + '%';
            elements.cerBar.style.width = (cer * 100) + '%';
        }, 100);
    }

    function setStatus(state) {
        badge = elements.statusBadge;
        text = elements.statusText;
        badge.className = 'transcript-status';
        if (state === 'processing') {
            text.textContent = 'Processing';
            badge.style.borderColor = 'rgba(255,179,0,0.35)';
            badge.style.background = 'rgba(255,179,0,0.1)';
            badge.style.color = '#ffb300';
            badge.querySelector('.status-dot').style.background = '#ffb300';
        } else if (state === 'ready') {
            text.textContent = 'Ready';
            badge.style.borderColor = 'rgba(0,229,192,0.35)';
            badge.style.background = 'rgba(0,229,192,0.1)';
            badge.style.color = 'var(--accent2)';
            badge.querySelector('.status-dot').style.background = 'var(--accent2)';
        } else {
            badge.classList.add('idle');
            text.textContent = 'Idle';
        }
    }

    // ── SETTINGS ──
    elements.settingsToggle.addEventListener('click', () => {
        elements.settingsDrawer.classList.toggle('open');
        elements.settingsChevron.classList.toggle('open');
    });

    document.querySelectorAll('.toggle-opt').forEach(el => {
        el.addEventListener('click', () => {
            el.parentElement.querySelectorAll('.toggle-opt').forEach(o => o.classList.remove('active'));
            el.classList.add('active');
        });
    });

    // ── CLIPBOARD ──
    elements.copyBtn.addEventListener('click', () => {
        if (!transcriptText) return;
        navigator.clipboard.writeText(transcriptText);
        showToast('⎘', 'Copied');
    });

    elements.copyTransBtn.addEventListener('click', () => {
        if (!translationText) return;
        navigator.clipboard.writeText(translationText);
        showToast('⎘', 'Copied');
    });

    elements.clearBtn.addEventListener('click', () => {
        transcriptText = '';
        translationText = '';
        elements.transcriptBody.innerHTML = `<div class="transcript-placeholder" id="transcriptPlaceholder">
            <div class="placeholder-icon">🎙</div>
            <div class="placeholder-text">Upload audio or record to begin transcription</div>
        </div>`;
        elements.translationPanel.classList.add('hidden');
        elements.werDetail.classList.remove('visible');
        updateMetric('wer', null);
        updateMetric('cer', null);
        elements.durationDisplay.textContent = '—';
        elements.confDisplay.textContent = '—';
        elements.durationDisplay.className = elements.confDisplay.className = 'metric-value muted';
        setStatus('idle');
        showToast('✕', 'Cleared');
    });

    // ── EXPORT ──
    function download(content, filename, mime) {
        const blog = new Blob([content], { type: mime });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blog);
        a.download = filename;
        a.click();
    }

    elements.exportTxt.addEventListener('click', () => {
        if (!transcriptText) return;
        download(transcriptText, 'transcript.txt', 'text/plain');
    });

    elements.exportSrt.addEventListener('click', () => {
        if (!transcriptText) return;
        const srt = `1\n00:00:00,000 --> 00:00:10,000\n${transcriptText}`;
        download(srt, 'transcript.srt', 'text/plain');
    });

    elements.exportJson.addEventListener('click', () => {
        if (!transcriptText) return;
        const data = { transcription: transcriptText, translation: translationText, date: new Date().toISOString() };
        download(JSON.stringify(data, null, 2), 'transcript.json', 'application/json');
    });

    // ── TOAST ──
    function showToast(icon, msg) {
        const t = document.getElementById('toast');
        document.getElementById('toastIcon').textContent = icon;
        document.getElementById('toastMsg').textContent = msg;
        t.classList.add('show');
        clearTimeout(window._toastTimer);
        window._toastTimer = setTimeout(() => t.classList.remove('show'), 2800);
    }

    // shortcuts
    document.addEventListener('keydown', e => {
        if ((e.metaKey || e.ctrlKey) && e.key === 't') { e.preventDefault(); elements.transcribeBtn.click(); }
        if ((e.metaKey || e.ctrlKey) && e.key === 'r') { e.preventDefault(); elements.recordBtn.click(); }
    });
});
