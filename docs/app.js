/* ============================================================
   CuantumWiki — Tokenizer App — Core Logic
   
   Features:
   - Simple word-level tokenizer with vocabulary
   - Web Serial API connection to ESP32-S3
   - JSONL protocol (infer, result, error)
   - Simulated mode when no serial connected
   ============================================================ */

// ── Tokenizer ──
const Tokenizer = (() => {
    // Simple word-level vocab (maps words to IDs)
    // In production, this would be loaded from a vocab.json file
    const vocab = {};
    let nextId = 1;

    // Special tokens
    const PAD_ID = 0;
    const UNK_ID = 9999;

    // Build vocab from common spanish + english words
    const seedWords = [
        'hola', 'chau', 'bueno', 'malo', 'bien', 'mal', 'gracias', 'por', 'favor',
        'urgente', 'necesito', 'ayuda', 'ahora', 'ya', 'rapido', 'pronto',
        'me', 'molesta', 'no', 'funciona', 'terrible', 'horrible', 'peor',
        'excelente', 'genial', 'perfecto', 'increible', 'hermoso', 'lindo',
        'estimado', 'le', 'informo', 'solicito', 'atentamente', 'cordialmente',
        'que', 'como', 'cuando', 'donde', 'quien', 'porque', 'para', 'con',
        'el', 'la', 'los', 'las', 'un', 'una', 'es', 'son', 'de', 'en',
        'y', 'o', 'si', 'pero', 'este', 'esta', 'ese', 'esa', 'todo', 'nada',
        'hello', 'thanks', 'please', 'help', 'urgent', 'now', 'good', 'bad',
        'the', 'is', 'a', 'an', 'to', 'of', 'and', 'or', 'not', 'it',
        'i', 'you', 'we', 'they', 'my', 'your', 'his', 'her', 'our',
        'can', 'will', 'do', 'have', 'be', 'was', 'are', 'has', 'had',
        'need', 'want', 'like', 'love', 'hate', 'think', 'know', 'see',
        'ok', 'yes', 'no', 'maybe', 'sure', 'fine', 'great', 'nice',
        'problema', 'error', 'falla', 'sistema', 'respuesta', 'tiempo',
        'muy', 'mucho', 'poco', 'bastante', 'demasiado', 'siempre', 'nunca',
        'quiero', 'puedo', 'debo', 'tengo', 'soy', 'estoy', 'voy', 'hay',
        'señor', 'señora', 'doctor', 'director', 'gerente', 'jefe',
        'por', 'medio', 'presente', 'adjunto', 'envio', 'remito',
        'disculpe', 'perdon', 'lamento', 'siento', 'pido',
    ];

    seedWords.forEach(w => { vocab[w] = nextId++; });

    function tokenize(text) {
        // Normalize: lowercase, remove punctuation except basic
        const clean = text.toLowerCase()
            .replace(/[^\w\sáéíóúñü]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        if (!clean) return [];

        const words = clean.split(' ');
        return words.map(w => {
            if (vocab[w] !== undefined) return { word: w, id: vocab[w] };
            // Auto-add to vocab (dynamic)
            vocab[w] = nextId++;
            return { word: w, id: vocab[w] };
        });
    }

    function tokenizeAndPad(text, maxLen = 64) {
        const tokens = tokenize(text);
        const ids = tokens.map(t => t.id);

        // Truncate if too long
        if (ids.length > maxLen) ids.length = maxLen;

        // Pad
        while (ids.length < maxLen) ids.push(PAD_ID);

        return { tokens, ids, padded: ids, actualLen: tokens.length };
    }

    return { tokenize, tokenizeAndPad, PAD_ID, UNK_ID, vocab };
})();


// ── Serial Manager ──
const SerialManager = (() => {
    let port = null;
    let reader = null;
    let readableStreamClosed = null;
    let writableStreamClosed = null;
    let writer = null;
    let connected = false;
    let lineBuffer = '';

    const listeners = {
        connect: [],
        disconnect: [],
        message: [],
        error: []
    };

    function on(event, cb) { listeners[event].push(cb); }
    function emit(event, data) { listeners[event].forEach(cb => cb(data)); }

    async function connect(baudRate = 115200) {
        try {
            port = await navigator.serial.requestPort();
            await port.open({ baudRate });
            connected = true;

            // Writer
            const encoder = new TextEncoderStream();
            writableStreamClosed = encoder.readable.pipeTo(port.writable);
            writer = encoder.writable.getWriter();

            // Reader
            const decoder = new TextDecoderStream();
            readableStreamClosed = port.readable.pipeTo(decoder.writable);
            reader = decoder.readable.getReader();

            emit('connect', {});
            readLoop();
            return true;
        } catch (err) {
            emit('error', { message: err.message });
            return false;
        }
    }

    async function readLoop() {
        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                if (value) {
                    lineBuffer += value;
                    const lines = lineBuffer.split('\n');
                    lineBuffer = lines.pop(); // keep incomplete line
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed) processLine(trimmed);
                    }
                }
            }
        } catch (err) {
            // Port disconnected
        }
        emit('disconnect', {});
        connected = false;
    }

    function processLine(line) {
        try {
            const msg = JSON.parse(line);
            emit('message', msg);
        } catch {
            // Not JSON, treat as raw log
            emit('message', { type: 'raw', data: line });
        }
    }

    async function send(obj) {
        const line = JSON.stringify(obj) + '\n';
        if (connected && writer) {
            await writer.write(line);
            return true;
        }
        return false;
    }

    async function disconnect() {
        try {
            if (reader) { await reader.cancel(); await readableStreamClosed.catch(() => { }); }
            if (writer) { await writer.close(); await writableStreamClosed; }
            if (port) await port.close();
        } catch { }
        connected = false;
        port = null;
        reader = null;
        writer = null;
        emit('disconnect', {});
    }

    function isConnected() { return connected; }

    return { connect, disconnect, send, isConnected, on };
})();


// ── Simulator (when no serial connected) ──
const Simulator = (() => {
    const labels = ['neutral', 'positivo', 'urgente', 'molesto', 'formal'];

    function classifyText(text) {
        const lower = text.toLowerCase();
        let scores = { neutral: 0.3, positivo: 0.1, urgente: 0.1, molesto: 0.1, formal: 0.1 };

        // Simple heuristic keyword matching
        const keywords = {
            positivo: ['bien', 'excelente', 'genial', 'perfecto', 'increible', 'hermoso', 'lindo', 'gracias', 'great', 'good', 'nice', 'love', 'amazing', 'thanks', 'bueno', 'buena'],
            urgente: ['urgente', 'ahora', 'ya', 'rapido', 'necesito', 'ayuda', 'pronto', 'urgent', 'help', 'now', 'asap', 'inmediato', 'emergencia'],
            molesto: ['molesta', 'terrible', 'horrible', 'peor', 'malo', 'mala', 'no funciona', 'odio', 'hate', 'bad', 'awful', 'angry', 'furioso', 'inaceptable', 'error', 'falla'],
            formal: ['estimado', 'informo', 'solicito', 'atentamente', 'cordialmente', 'presente', 'adjunto', 'señor', 'señora', 'dear', 'regards', 'sincerely', 'director', 'gerente'],
        };

        for (const [label, words] of Object.entries(keywords)) {
            for (const w of words) {
                if (lower.includes(w)) scores[label] += 0.25;
            }
        }

        // Normalize
        const total = Object.values(scores).reduce((a, b) => a + b, 0);
        for (const k in scores) scores[k] /= total;

        // Pick top label
        let topLabel = 'neutral';
        let topScore = 0;
        for (const [label, score] of Object.entries(scores)) {
            if (score > topScore) { topScore = score; topLabel = label; }
        }

        // Add some randomness to confidence
        const jitter = (Math.random() - 0.5) * 0.1;
        const confidence = Math.min(0.99, Math.max(0.5, topScore + jitter));

        return {
            label: topLabel,
            confidence: parseFloat(confidence.toFixed(2)),
            node: `c3-${Math.floor(Math.random() * 4)}`,
            latency_ms: Math.floor(20 + Math.random() * 60),
        };
    }

    return { classifyText };
})();


// ── App Controller ──
const App = (() => {
    let reqCounter = 0;
    let maxLen = 64;
    let routeMode = 'auto';
    let modelVersion = 'tone-v1';
    let pendingRequests = {};
    let simulationMode = true;

    // Nodes (simulated state)
    let nodes = [
        { id: 'c3-0', addr: '0x10', status: 'unknown', lastSeen: null, infers: 0, errors: 0, avgLatency: 0 },
        { id: 'c3-1', addr: '0x11', status: 'unknown', lastSeen: null, infers: 0, errors: 0, avgLatency: 0 },
        { id: 'c3-2', addr: '0x12', status: 'unknown', lastSeen: null, infers: 0, errors: 0, avgLatency: 0 },
        { id: 'c3-3', addr: '0x13', status: 'unknown', lastSeen: null, infers: 0, errors: 0, avgLatency: 0 },
    ];

    function init() {
        bindUI();
        setupSerial();
        renderNodes();
        updateSerialStatus();
        addLog('sys', 'CuantumWiki Tokenizer iniciado');
        addLog('sys', 'Modo simulación activo (sin serial)');
    }

    function bindUI() {
        const textarea = document.getElementById('text-input');
        const sendBtn = document.getElementById('btn-send');
        const clearBtn = document.getElementById('btn-clear');
        const connectBtn = document.getElementById('btn-connect');
        const disconnectBtn = document.getElementById('btn-disconnect');
        const maxLenInput = document.getElementById('config-maxlen');
        const routeSelect = document.getElementById('config-route');

        textarea.addEventListener('input', () => {
            updateCharCount();
            updateTokenPreview();
        });

        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                sendInfer();
            }
        });

        sendBtn.addEventListener('click', sendInfer);
        clearBtn.addEventListener('click', clearResults);
        connectBtn.addEventListener('click', connectSerial);
        disconnectBtn.addEventListener('click', disconnectSerial);

        maxLenInput.addEventListener('change', () => {
            maxLen = parseInt(maxLenInput.value) || 64;
            updateTokenPreview();
        });

        routeSelect.addEventListener('change', () => {
            routeMode = routeSelect.value;
        });

        updateCharCount();
        updateTokenPreview();
    }

    function setupSerial() {
        SerialManager.on('connect', () => {
            simulationMode = false;
            updateSerialStatus();
            addLog('sys', 'Serial conectado');
            // Send hello
            SerialManager.send({ type: 'hello' });
        });

        SerialManager.on('disconnect', () => {
            simulationMode = true;
            updateSerialStatus();
            addLog('sys', 'Serial desconectado — modo simulación');
        });

        SerialManager.on('message', (msg) => {
            handleSerialMessage(msg);
        });

        SerialManager.on('error', (err) => {
            addLog('err', `Serial error: ${err.message}`);
        });
    }

    async function connectSerial() {
        if (!('serial' in navigator)) {
            addLog('err', 'Web Serial API no disponible. Usa Chrome/Edge.');
            return;
        }
        addLog('sys', 'Solicitando puerto serial...');
        await SerialManager.connect();
    }

    async function disconnectSerial() {
        await SerialManager.disconnect();
    }

    function updateSerialStatus() {
        const el = document.getElementById('serial-status');
        const connectBtn = document.getElementById('btn-connect');
        const disconnectBtn = document.getElementById('btn-disconnect');

        if (SerialManager.isConnected()) {
            el.className = 'serial-status connected';
            el.innerHTML = '<span class="dot"></span> Conectado';
            connectBtn.style.display = 'none';
            disconnectBtn.style.display = 'inline-flex';
        } else {
            el.className = 'serial-status disconnected';
            el.innerHTML = '<span class="dot"></span> Desconectado';
            connectBtn.style.display = 'inline-flex';
            disconnectBtn.style.display = 'none';
        }
    }

    function updateCharCount() {
        const text = document.getElementById('text-input').value;
        document.getElementById('char-count').textContent = `${text.length} chars`;
    }

    function updateTokenPreview() {
        const text = document.getElementById('text-input').value;
        const container = document.getElementById('token-chips');
        const countEl = document.getElementById('token-count');

        if (!text.trim()) {
            container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">Escribe algo para ver los tokens...</span>';
            countEl.textContent = '0 / ' + maxLen;
            return;
        }

        const result = Tokenizer.tokenizeAndPad(text, maxLen);
        countEl.textContent = `${result.actualLen} / ${maxLen}`;

        let html = '';
        // Show actual tokens
        result.tokens.forEach((t, i) => {
            if (i >= maxLen) return;
            html += `<span class="token-chip" title="ID: ${t.id}">${t.word}<span class="token-id">${t.id}</span></span>`;
        });

        // Show padding (max 8 visible)
        const padCount = maxLen - Math.min(result.actualLen, maxLen);
        const showPad = Math.min(padCount, 8);
        for (let i = 0; i < showPad; i++) {
            html += `<span class="token-chip pad">[PAD]<span class="token-id">0</span></span>`;
        }
        if (padCount > showPad) {
            html += `<span class="token-chip pad">+${padCount - showPad} PAD</span>`;
        }

        container.innerHTML = html;
    }

    async function sendInfer() {
        const textarea = document.getElementById('text-input');
        const text = textarea.value.trim();
        if (!text) return;

        const sendBtn = document.getElementById('btn-send');
        sendBtn.disabled = true;

        reqCounter++;
        const reqId = `r-${String(reqCounter).padStart(4, '0')}`;
        const result = Tokenizer.tokenizeAndPad(text, maxLen);

        const msg = {
            type: 'infer',
            req_id: reqId,
            model: modelVersion,
            tokens: result.ids,
            max_len: maxLen,
            route: routeMode,
        };

        addLog('tx', `→ infer ${reqId}: "${text.substring(0, 40)}${text.length > 40 ? '...' : ''}" (${Math.min(result.actualLen, maxLen)} tokens, padded to ${maxLen})`);

        if (simulationMode) {
            // Simulate
            const delay = 30 + Math.random() * 80;
            setTimeout(() => {
                const simResult = Simulator.classifyText(text);
                handleResult({
                    type: 'result',
                    req_id: reqId,
                    label: simResult.label,
                    confidence: simResult.confidence,
                    node: simResult.node,
                    latency_ms: simResult.latency_ms,
                }, text);
                sendBtn.disabled = false;
            }, delay);
        } else {
            pendingRequests[reqId] = { text, sentAt: Date.now() };
            await SerialManager.send(msg);
            // Timeout
            setTimeout(() => {
                if (pendingRequests[reqId]) {
                    delete pendingRequests[reqId];
                    addLog('err', `Timeout para ${reqId}`);
                    sendBtn.disabled = false;
                }
            }, 5000);
        }
    }

    function handleSerialMessage(msg) {
        if (msg.type === 'result') {
            const pending = pendingRequests[msg.req_id];
            const text = pending ? pending.text : '';
            delete pendingRequests[msg.req_id];
            handleResult(msg, text);
            document.getElementById('btn-send').disabled = false;
        } else if (msg.type === 'error') {
            addLog('err', `← error ${msg.req_id}: ${msg.code} — ${msg.message}`);
            delete pendingRequests[msg.req_id];
            document.getElementById('btn-send').disabled = false;
        } else if (msg.type === 'raw') {
            addLog('rx', `← ${msg.data}`);
        } else if (msg.type === 'hello') {
            addLog('rx', `← HELLO OK: ${JSON.stringify(msg)}`);
        }
    }

    function handleResult(msg, text) {
        addLog('rx', `← result ${msg.req_id}: ${msg.label} (${(msg.confidence * 100).toFixed(0)}%) via ${msg.node} in ${msg.latency_ms}ms`);

        // Update node
        const nodeIdx = nodes.findIndex(n => n.id === msg.node);
        if (nodeIdx >= 0) {
            const node = nodes[nodeIdx];
            node.status = 'online';
            node.lastSeen = Date.now();
            node.infers++;
            node.avgLatency = node.avgLatency === 0 ? msg.latency_ms : Math.round((node.avgLatency + msg.latency_ms) / 2);
            renderNodes();
        }

        // Render result card
        renderResult(msg, text);
    }

    function renderResult(msg, text) {
        const container = document.getElementById('results-container');
        const empty = document.getElementById('results-empty');
        if (empty) empty.style.display = 'none';

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
      <div class="result-header">
        <span class="result-label ${msg.label}">${msg.label}</span>
        <span class="result-confidence" style="color: var(--${msg.label}-color)">${(msg.confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="confidence-bar">
        <div class="confidence-fill ${msg.label}" style="width: ${msg.confidence * 100}%"></div>
      </div>
      <div class="result-meta">
        <div class="result-meta-item">
          <span class="meta-icon">🔶</span>
          <span class="meta-value">${msg.node}</span>
        </div>
        <div class="result-meta-item">
          <span class="meta-icon">⚡</span>
          <span class="meta-value">${msg.latency_ms}ms</span>
        </div>
        <div class="result-meta-item">
          <span class="meta-icon">🏷️</span>
          <span class="meta-value">${msg.req_id}</span>
        </div>
      </div>
      ${text ? `<div class="result-text">${escapeHtml(text.substring(0, 120))}</div>` : ''}
    `;

        container.insertBefore(card, container.firstChild);

        // Max 20 results
        while (container.children.length > 20) {
            container.removeChild(container.lastChild);
        }
    }

    function clearResults() {
        const container = document.getElementById('results-container');
        container.innerHTML = `
      <div id="results-empty" class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>Sin resultados aún</h3>
        <p>Escribe un texto y presiona "Analizar" para detectar el tono.</p>
      </div>
    `;
    }

    function renderNodes() {
        nodes.forEach((node, i) => {
            const statusEl = document.getElementById(`node-${i}-status`);
            const infersEl = document.getElementById(`node-${i}-infers`);
            const latencyEl = document.getElementById(`node-${i}-latency`);
            const lastEl = document.getElementById(`node-${i}-last`);

            if (statusEl) statusEl.className = `node-status ${node.status}`;
            if (infersEl) infersEl.textContent = node.infers;
            if (latencyEl) latencyEl.textContent = node.avgLatency ? `${node.avgLatency}ms` : '--';
            if (lastEl) {
                if (node.lastSeen) {
                    const ago = Math.round((Date.now() - node.lastSeen) / 1000);
                    lastEl.textContent = ago < 60 ? `${ago}s ago` : `${Math.round(ago / 60)}m ago`;
                } else {
                    lastEl.textContent = '--';
                }
            }
        });
    }

    function addLog(type, text) {
        const console = document.getElementById('log-console');
        const now = new Date();
        const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        const line = document.createElement('div');
        line.className = 'log-line';
        line.innerHTML = `<span class="log-time">${time}</span><span class="log-msg ${type}">${escapeHtml(text)}</span>`;
        console.appendChild(line);
        console.scrollTop = console.scrollHeight;

        // Max 100 lines
        while (console.children.length > 100) {
            console.removeChild(console.firstChild);
        }
    }

    function escapeHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Refresh node "last seen" times
    setInterval(renderNodes, 5000);

    return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
