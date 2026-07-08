"""
app.py — Resume RAG LLM — Streamlit Main Dashboard
Person 3: LLM QA Integration & Streamlit Interface

Commit 1: Design main Streamlit layout and session state
"""

import sys
from pathlib import Path

# Add src/ to path so modules can be imported cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page Config  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium dark-mode glassmorphism design
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-primary:   #0d0f1a;
    --bg-secondary: #131624;
    --bg-card:      rgba(255,255,255,0.04);
    --bg-card-hover:rgba(255,255,255,0.07);
    --border:       rgba(255,255,255,0.08);
    --border-glow:  rgba(99,179,237,0.35);
    --accent:       #63b3ed;
    --accent-2:     #9f7aea;
    --accent-grad:  linear-gradient(135deg, #63b3ed 0%, #9f7aea 100%);
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --success:      #48bb78;
    --warning:      #f6ad55;
    --danger:       #fc8181;
    --shadow-glow:  0 0 40px rgba(99,179,237,0.12);
    --radius-lg:    16px;
    --radius-md:    10px;
    --radius-sm:    6px;
    --transition:   all 0.25s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 4rem 2rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
}

/* ── Header ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.rag-header .logo {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: var(--accent-grad);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: var(--shadow-glow);
    flex-shrink: 0;
}
.rag-header .title-block h1 {
    font-size: 1.7rem;
    font-weight: 700;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.rag-header .title-block p {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin: 4px 0 0 0;
}

/* ── Chat Bubble — User ── */
.chat-bubble-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
    animation: fadeSlideIn 0.3s ease;
}
.chat-bubble-user .bubble {
    background: linear-gradient(135deg, #2b4a7a 0%, #3b3069 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    padding: 12px 18px;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text-primary);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Chat Bubble — Assistant ── */
.chat-bubble-ai {
    display: flex;
    justify-content: flex-start;
    gap: 12px;
    margin: 10px 0;
    animation: fadeSlideIn 0.3s ease;
}
.chat-bubble-ai .avatar {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--accent-grad);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
    box-shadow: var(--shadow-glow);
}
.chat-bubble-ai .bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px var(--radius-lg) var(--radius-lg) var(--radius-lg);
    padding: 14px 18px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
}
.chat-bubble-ai .bubble.blocked {
    border-color: rgba(252,129,129,0.4);
    background: rgba(252,129,129,0.06);
}
.chat-bubble-ai .bubble.loading {
    color: var(--text-muted);
    font-style: italic;
}

/* ── Alert Banners ── */
.alert-spelling {
    background: rgba(246,173,85,0.1);
    border: 1px solid rgba(246,173,85,0.4);
    border-left: 3px solid var(--warning);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    margin: 6px 0 10px 46px;
    font-size: 0.82rem;
    color: var(--warning);
    display: flex;
    align-items: center;
    gap: 8px;
    animation: fadeSlideIn 0.3s ease;
}
.alert-blocked {
    background: rgba(252,129,129,0.08);
    border: 1px solid rgba(252,129,129,0.35);
    border-left: 3px solid var(--danger);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    margin: 6px 0 10px 46px;
    font-size: 0.82rem;
    color: var(--danger);
    display: flex;
    align-items: center;
    gap: 8px;
    animation: fadeSlideIn 0.3s ease;
}

/* ── Candidate Name Badge ── */
.candidate-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 8px;
}

/* ── Stats Cards ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 120px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    backdrop-filter: blur(6px);
    transition: var(--transition);
}
.stat-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Chat Window Container ── */
.chat-window {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    min-height: 400px;
    max-height: 520px;
    overflow-y: auto;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    scroll-behavior: smooth;
}
.chat-window::-webkit-scrollbar { width: 4px; }
.chat-window::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 4px; }

/* ── Empty State ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted);
    text-align: center;
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
.empty-state p { font-size: 0.88rem; line-height: 1.6; max-width: 340px; }

/* ── Divider with label ── */
.chat-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0;
    color: var(--text-muted);
    font-size: 0.75rem;
}
.chat-divider::before,
.chat-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Input area styling ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: var(--transition) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent-grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 16px rgba(99,179,237,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,179,237,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar Sections ── */
.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    margin-bottom: 12px;
}
.sidebar-section h4 {
    color: var(--accent) !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 10px 0;
}
.sidebar-section .candidate-list-item {
    font-size: 0.82rem;
    color: var(--text-muted);
    padding: 3px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-section .candidate-list-item:last-child { border-bottom: none; }

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.thinking-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    margin: 0 2px;
    animation: pulse 1.2s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Scrollbar global ── */
* { scrollbar-width: thin; scrollbar-color: var(--border-glow) transparent; }

/* ── Utility ── */
.text-muted { color: var(--text-muted) !important; }
.text-accent { color: var(--accent) !important; }
.text-success { color: var(--success) !important; }
.text-warning { color: var(--warning) !important; }
.text-danger  { color: var(--danger) !important; }
.mono { font-family: 'JetBrains Mono', monospace !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _init_session_state():
    """Initialise all required Streamlit session-state keys on first load."""
    defaults = {
        "chat_history":     [],     # list of {role, content, metadata} dicts
        "engine":           None,   # lazy-loaded FilteredQueryEngine instance
        "engine_ready":     False,  # whether the engine loaded without errors
        "engine_error":     None,   # engine load error message if any
        "total_queries":    0,      # running count of queries
        "blocked_queries":  0,      # running count of blocked/rejected queries
        "corrected_queries":0,      # running count of spelling-corrected queries
        "last_candidate":   None,   # last successfully resolved candidate name
        "candidate_list":   [],     # list of known candidate names (from matcher)
        "theme_mode":       "dark", # reserved for future theme toggle
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# Lazy Engine Loader
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_engine():
    """
    Loads and caches the FilteredQueryEngine.
    Returns (engine, candidate_list, error_message).
    """
    try:
        from retriever import FilteredQueryEngine
        engine = FilteredQueryEngine()
        candidates = sorted(engine.matcher.candidates)
        return engine, candidates, None
    except Exception as exc:
        return None, [], str(exc)


def ensure_engine():
    """Ensures the engine is loaded into session state (called once per session)."""
    if not st.session_state.engine_ready and st.session_state.engine is None:
        with st.spinner("⚙️ Initialising RAG pipeline…"):
            engine, candidates, err = _load_engine()
        st.session_state.engine = engine
        st.session_state.candidate_list = candidates
        st.session_state.engine_error = err
        st.session_state.engine_ready = (engine is not None)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:16px 0 8px 0;">
              <div style="font-size:2.2rem;">🤖</div>
              <div style="font-weight:700;font-size:1rem;color:#63b3ed;">Resume RAG</div>
              <div style="font-size:0.75rem;color:#8892a4;">LLM Assistant · Team 3</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Status indicator ──
        if st.session_state.engine_ready:
            st.success("✅ Pipeline Ready", icon=None)
        elif st.session_state.engine_error:
            st.warning(f"⚠️ {st.session_state.engine_error[:120]}", icon=None)
        else:
            st.info("⏳ Not initialised yet")

        st.divider()

        # ── Known Candidates ──
        st.markdown("#### 👥 Known Candidates")
        candidates = st.session_state.candidate_list
        if candidates:
            for i, name in enumerate(candidates[:15]):
                st.markdown(
                    f'<div class="candidate-list-item">👤 {name}</div>',
                    unsafe_allow_html=True,
                )
            if len(candidates) > 15:
                st.markdown(
                    f'<div style="color:#8892a4;font-size:0.78rem;">+ {len(candidates)-15} more…</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#8892a4;font-size:0.82rem;">No candidates loaded.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Help / Usage ──
        with st.expander("💡 How to use", expanded=False):
            st.markdown(
                """
                **Supported queries:**
                - _"What are Ashok's Python skills?"_
                - _"Tell me about pawan's education"_  (spelling correction)
                - _"Summarize yasasvi's experience"_

                **Blocked queries (by design):**
                - Comparisons: _"Compare Ashok and Pawan"_
                - Rankings: _"Who is the best candidate?"_
                - Aggregations: _"List all Java developers"_
                """,
                unsafe_allow_html=False,
            )

        # ── Clear history ──
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.blocked_queries = 0
            st.session_state.corrected_queries = 0
            st.session_state.last_candidate = None
            st.rerun()

        st.divider()
        st.markdown(
            '<div style="font-size:0.72rem;color:#8892a4;text-align:center;">Resume RAG LLM · Person 3 · 2026</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Chat History Renderer
# ──────────────────────────────────────────────────────────────────────────────
def render_chat_message(msg: dict):
    """Renders a single chat message with appropriate styling."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    meta = msg.get("metadata", {})

    if role == "user":
        st.markdown(
            f'<div class="chat-bubble-user"><div class="bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        is_blocked = meta.get("is_blocked", False)
        bubble_class = "bubble blocked" if is_blocked else "bubble"
        st.markdown(
            f"""
            <div class="chat-bubble-ai">
              <div class="avatar">🤖</div>
              <div>
                {'<div class="candidate-badge">👤 ' + meta.get("candidate_name","") + '</div>' if meta.get("candidate_name") else ''}
                <div class="{bubble_class}">{content}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Spelling-correction alert
        if meta.get("spelling_suggestion"):
            st.markdown(
                f'<div class="alert-spelling">✏️ {meta["spelling_suggestion"]}</div>',
                unsafe_allow_html=True,
            )

        # Blocked-query alert
        if is_blocked and meta.get("blocked_reason"):
            st.markdown(
                f'<div class="alert-blocked">🚫 {meta["blocked_reason"]}</div>',
                unsafe_allow_html=True,
            )


def render_chat_window():
    """Renders the full chat history window."""
    history = st.session_state.chat_history

    if not history:
        st.markdown(
            """
            <div class="empty-state">
              <div class="icon">💬</div>
              <p>No conversation yet.<br>
              Type a question about a specific candidate to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for i, msg in enumerate(history):
        render_chat_message(msg)
        # Insert a light divider between QA pairs
        if msg.get("role") == "assistant" and i < len(history) - 1:
            st.markdown(
                '<div class="chat-divider">· · ·</div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Stats Row
# ──────────────────────────────────────────────────────────────────────────────
def render_stats():
    candidates_count = len(st.session_state.candidate_list) or "–"
    st.markdown(
        f"""
        <div class="stats-row">
          <div class="stat-card">
            <div class="stat-value">{candidates_count}</div>
            <div class="stat-label">Candidates indexed</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{st.session_state.total_queries}</div>
            <div class="stat-label">Queries sent</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{st.session_state.corrected_queries}</div>
            <div class="stat-label">Spelling corrections</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{st.session_state.blocked_queries}</div>
            <div class="stat-label">Blocked queries</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Query Handler  (Commit 2 logic integrated here)
# ──────────────────────────────────────────────────────────────────────────────
def handle_query(query: str):
    """
    Sends the user query through the FilteredQueryEngine and appends the
    result (including any spelling alerts or blocked-query alerts) to
    st.session_state.chat_history.
    """
    if not query.strip():
        return

    # Append user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "metadata": {},
    })
    st.session_state.total_queries += 1

    engine = st.session_state.engine

    if not engine:
        # Engine not available — show error message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⚠️ The RAG pipeline is not available. Please check that your `.env` file contains a valid `OPENROUTER_API_KEY` and that the Chroma DB index has been built (run `python src/indexer.py`).",
            "metadata": {"is_blocked": False},
        })
        return

    with st.spinner("Thinking…"):
        result = engine.query(query.strip())

    # Update running stats
    if result.get("is_blocked"):
        st.session_state.blocked_queries += 1
    if result.get("spelling_suggestion"):
        st.session_state.corrected_queries += 1
    if result.get("candidate_name"):
        st.session_state.last_candidate = result["candidate_name"]

    # Append AI response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("response", "No response returned."),
        "metadata": {
            "candidate_name":     result.get("candidate_name"),
            "spelling_suggestion":result.get("spelling_suggestion"),
            "is_blocked":         result.get("is_blocked", False),
            "blocked_reason":     result.get("blocked_reason"),
        },
    })


# ──────────────────────────────────────────────────────────────────────────────
# Quick-Questions Panel
# ──────────────────────────────────────────────────────────────────────────────
QUICK_QUESTIONS = [
    "What are the skills of Pawan?",
    "Tell me about Ashok Reddy's experience",
    "What is Yasasvi's educational background?",
    "Summarise Trinadh's projects",
    "Compare Ashok and Pawan",  # intentionally blocked — demo
]


def render_quick_questions():
    st.markdown(
        '<div style="font-size:0.8rem;color:#8892a4;margin-bottom:8px;">⚡ Quick Questions</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(QUICK_QUESTIONS))
    for col, q in zip(cols, QUICK_QUESTIONS):
        with col:
            if st.button(q[:30] + ("…" if len(q) > 30 else ""), key=f"quick_{q[:15]}", use_container_width=True):
                handle_query(q)
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main Layout
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ensure_engine()
    render_sidebar()

    # ── Header ──
    st.markdown(
        """
        <div class="rag-header">
          <div class="logo">🤖</div>
          <div class="title-block">
            <h1>Resume RAG Assistant</h1>
            <p>Ask detailed questions about individual candidates · Powered by LlamaIndex + Qwen via OpenRouter</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Stats ──
    render_stats()

    # ── Chat Window ──
    with st.container():
        st.markdown('<div class="chat-window">', unsafe_allow_html=True)
        render_chat_window()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Questions ──
    render_quick_questions()
    st.divider()

    # ── Input Form ──
    with st.form(key="query_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Ask about a candidate",
                placeholder='e.g. "What are Ashok\'s Python skills?" or "Summarise pawan\'s education"',
                label_visibility="collapsed",
                key="query_input",
            )
        with col_btn:
            submitted = st.form_submit_button("Send →", use_container_width=True)

        if submitted and user_input:
            handle_query(user_input)
            st.rerun()

    # ── Engine error banner ──
    if st.session_state.engine_error:
        st.error(
            f"**Pipeline Error:** {st.session_state.engine_error}\n\n"
            "Ensure your `.env` file is configured and dependencies are installed (`pip install -r requirements.txt`).",
            icon="🚨",
        )

    # ── Last resolved candidate chip (footer) ──
    if st.session_state.last_candidate:
        st.markdown(
            f'<div style="text-align:right;margin-top:-8px;">'
            f'<span class="candidate-badge">Last candidate: 👤 {st.session_state.last_candidate}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
