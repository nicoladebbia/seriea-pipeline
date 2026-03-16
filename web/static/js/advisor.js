/**
 * SerieAI Advisor — Chat UI with SSE streaming
 *
 * Supports two modes:
 *   1. Floating panel (injected via base.html on all pages)
 *   2. Fullpage (/advisor route)
 *
 * Uses fetch + ReadableStream for SSE (POST body support).
 * Session stored in sessionStorage (clears on tab close — intentional for time-sensitive data).
 */

window.SerieAI = window.SerieAI || {};

SerieAI.Advisor = (() => {
  let _messagesEl = null;
  let _inputEl = null;
  let _sendBtn = null;
  let _stopBtn = null;
  let _abortController = null;
  let _streaming = false;
  let _sessionId = null;
  let _history = [];  // local mirror for sessionStorage
  let _renderTimer = null;

  // -----------------------------------------------------------------------
  // Session management
  // -----------------------------------------------------------------------
  function _getSessionId() {
    if (!_sessionId) {
      _sessionId = sessionStorage.getItem('advisor_session') || ('s_' + Date.now());
      sessionStorage.setItem('advisor_session', _sessionId);
    }
    return _sessionId;
  }

  function _saveHistory() {
    try {
      sessionStorage.setItem('advisor_history', JSON.stringify(_history));
    } catch (e) { /* quota exceeded — ignore */ }
  }

  function _loadHistory() {
    try {
      const raw = sessionStorage.getItem('advisor_history');
      if (raw) _history = JSON.parse(raw);
    } catch (e) { _history = []; }
  }

  // -----------------------------------------------------------------------
  // Rendering
  // -----------------------------------------------------------------------
  function _renderMarkdown(text) {
    if (typeof marked !== 'undefined' && marked.parse) {
      return marked.parse(text, { breaks: true });
    }
    return text.replace(/\n/g, '<br>');
  }

  function _createMessageEl(role, content) {
    const wrap = document.createElement('div');
    wrap.className = 'advisor-msg advisor-msg--' + role;

    const bubble = document.createElement('div');
    bubble.className = 'advisor-bubble advisor-bubble--' + role;

    if (role === 'assistant') {
      bubble.innerHTML = _renderMarkdown(content);
    } else {
      bubble.textContent = content;
    }

    wrap.appendChild(bubble);
    return wrap;
  }

  function _appendMessage(role, content) {
    if (!_messagesEl) return;
    const el = _createMessageEl(role, content);
    _messagesEl.appendChild(el);
    _scrollToBottom(role === 'user');  // force scroll on user messages
    return el;
  }

  function _isNearBottom() {
    if (!_messagesEl) return true;
    const threshold = 120;
    return (_messagesEl.scrollHeight - _messagesEl.scrollTop - _messagesEl.clientHeight) < threshold;
  }

  function _scrollToBottom(force) {
    if (_messagesEl && (force || _isNearBottom())) {
      _messagesEl.scrollTop = _messagesEl.scrollHeight;
    }
  }

  function _showTypingIndicator() {
    const existing = _messagesEl?.querySelector('.advisor-typing');
    if (existing) return existing;

    const wrap = document.createElement('div');
    wrap.className = 'advisor-msg advisor-msg--assistant advisor-typing';
    wrap.innerHTML = `<div class="advisor-bubble advisor-bubble--assistant">
      <span class="advisor-dots"><span></span><span></span><span></span></span>
    </div>`;
    _messagesEl?.appendChild(wrap);
    _scrollToBottom();
    return wrap;
  }

  function _removeTypingIndicator() {
    _messagesEl?.querySelector('.advisor-typing')?.remove();
  }

  function _showToolIndicator(toolName) {
    const label = {
      'get_match_prediction': 'Analyzing match...',
      'get_team_detail': 'Looking up team...',
      'get_player_stats': 'Searching player...',
      'get_h2h': 'Checking H2H record...',
      'get_value_bets': 'Checking value bets...',
      'get_live_matches': 'Checking live scores...',
      'get_bankroll_status': 'Checking bankroll...',
      'get_match_context': 'Gathering match context...',
      'get_results': 'Checking recent results...',
      'get_match_scorers': 'Analyzing goalscorer candidates...',
      'get_match_players': 'Loading match player ratings...',
      'settle_bets': 'Settling bets...',
      'place_bet': 'Placing bet...',
      'manage_bets': 'Managing bets...',
      'build_parlay': 'Building parlay...',
      'query_history': 'Searching historical data...',
    }[toolName] || 'Fetching data...';

    _removeTypingIndicator();
    const wrap = document.createElement('div');
    wrap.className = 'advisor-msg advisor-msg--assistant advisor-typing';
    wrap.innerHTML = `<div class="advisor-bubble advisor-bubble--tool">
      <span class="advisor-tool-icon">&#x1f50d;</span> ${label}
    </div>`;
    _messagesEl?.appendChild(wrap);
    _scrollToBottom();
  }

  // -----------------------------------------------------------------------
  // Streaming
  // -----------------------------------------------------------------------
  async function _send(message) {
    if (_streaming || !message.trim()) return;

    _streaming = true;
    if (_sendBtn) _sendBtn.style.display = 'none';
    if (_stopBtn) _stopBtn.style.display = '';

    // Add user message
    _history.push({ role: 'user', content: message });
    _saveHistory();
    _appendMessage('user', message);
    if (_inputEl) { _inputEl.value = ''; _autoResize(_inputEl); }

    _showTypingIndicator();

    _abortController = new AbortController();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          session_id: _getSessionId(),
        }),
        signal: _abortController.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Request failed' }));
        _removeTypingIndicator();
        _appendMessage('assistant', `Error: ${err.error || resp.statusText}`);
        _finishStreaming();
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';
      let assistantEl = null;
      let bubbleEl = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let data;
          try { data = JSON.parse(raw); } catch { continue; }

          if (data.type === 'text') {
            _removeTypingIndicator();
            assistantText += data.content;

            if (!assistantEl) {
              assistantEl = _createMessageEl('assistant', '');
              bubbleEl = assistantEl.querySelector('.advisor-bubble');
              _messagesEl?.appendChild(assistantEl);
            }

            // Debounced markdown render
            if (_renderTimer) clearTimeout(_renderTimer);
            _renderTimer = setTimeout(() => {
              if (bubbleEl) bubbleEl.innerHTML = _renderMarkdown(assistantText);
              _scrollToBottom();
            }, 80);

          } else if (data.type === 'tool_use') {
            _showToolIndicator(data.tool);

          } else if (data.type === 'error') {
            _removeTypingIndicator();
            _appendMessage('assistant', `Error: ${data.content}`);

          } else if (data.type === 'done') {
            // Final render
            if (bubbleEl && assistantText) {
              if (_renderTimer) clearTimeout(_renderTimer);
              bubbleEl.innerHTML = _renderMarkdown(assistantText);
            }
            _removeTypingIndicator();
          }
        }
      }

      if (assistantText) {
        _history.push({ role: 'assistant', content: assistantText });
        _saveHistory();
      }

    } catch (err) {
      if (err.name !== 'AbortError') {
        _removeTypingIndicator();
        _appendMessage('assistant', `Connection error: ${err.message}`);
      }
    }

    _finishStreaming();
  }

  function _finishStreaming() {
    _streaming = false;
    _abortController = null;
    _removeTypingIndicator();
    if (_sendBtn) _sendBtn.style.display = '';
    if (_stopBtn) _stopBtn.style.display = 'none';
    _scrollToBottom();
  }

  function _stop() {
    if (_abortController) _abortController.abort();
    _finishStreaming();
  }

  // -----------------------------------------------------------------------
  // Auto-resize textarea
  // -----------------------------------------------------------------------
  function _autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  // -----------------------------------------------------------------------
  // Load greeting
  // -----------------------------------------------------------------------
  async function _loadGreeting() {
    try {
      const resp = await fetch('/api/chat/greeting');
      const data = await resp.json();
      if (data.content) {
        _appendMessage('assistant', data.content);
        _history.push({ role: 'assistant', content: data.content });
        _saveHistory();
      }
    } catch (e) {
      console.warn('Failed to load greeting:', e);
    }
  }

  // -----------------------------------------------------------------------
  // Restore session or show greeting
  // -----------------------------------------------------------------------
  function _initChat() {
    _loadHistory();
    if (_history.length > 0) {
      // Restore previous messages
      for (const msg of _history) {
        _appendMessage(msg.role, msg.content);
      }
    } else {
      _loadGreeting();
    }
  }

  // -----------------------------------------------------------------------
  // Input handlers
  // -----------------------------------------------------------------------
  function _bindInput() {
    if (_inputEl) {
      _inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          _send(_inputEl.value);
        }
      });
      _inputEl.addEventListener('input', () => _autoResize(_inputEl));
    }
    if (_sendBtn) {
      _sendBtn.addEventListener('click', () => {
        if (_inputEl) _send(_inputEl.value);
      });
    }
    if (_stopBtn) {
      _stopBtn.addEventListener('click', _stop);
    }
  }

  // -----------------------------------------------------------------------
  // Floating panel (all pages via base.html)
  // -----------------------------------------------------------------------
  function initPanel() {
    const panel = document.getElementById('advisor-panel');
    const fab = document.getElementById('advisor-fab');
    const closeBtn = document.getElementById('advisor-close');
    const expandBtn = document.getElementById('advisor-expand');

    if (!panel || !fab) return;

    _messagesEl = document.getElementById('advisor-messages');
    _inputEl = document.getElementById('advisor-input');
    _sendBtn = document.getElementById('advisor-send');
    _stopBtn = document.getElementById('advisor-stop-btn');

    fab.addEventListener('click', () => {
      panel.classList.add('advisor-panel--open');
      fab.style.display = 'none';
      if (!_history.length && _messagesEl && !_messagesEl.children.length) {
        _initChat();
      }
      _inputEl?.focus();
    });

    closeBtn?.addEventListener('click', () => {
      panel.classList.remove('advisor-panel--open');
      fab.style.display = '';
    });

    expandBtn?.addEventListener('click', () => {
      window.location.href = '/advisor';
    });

    _bindInput();
  }

  // -----------------------------------------------------------------------
  // Fullpage mode (/advisor)
  // -----------------------------------------------------------------------
  function initFullpage(messagesId, inputId, sendId, stopId) {
    _messagesEl = document.getElementById(messagesId);
    _inputEl = document.getElementById(inputId);
    _sendBtn = document.getElementById(sendId);
    _stopBtn = document.getElementById(stopId);

    _bindInput();
    _initChat();
  }

  async function clearChat() {
    _history = [];
    _saveHistory();
    if (_messagesEl) _messagesEl.innerHTML = '';
    // Clear server-side conversation
    try {
      await fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: _getSessionId() }),
      });
    } catch (e) { /* ignore */ }
    // Reload greeting
    _loadGreeting();
  }

  return { initPanel, initFullpage, clearChat };
})();
