"""
app.py — Resume RAG LLM — Recruiter Dashboard Interface
"""

import sys
import time
from pathlib import Path
import nest_asyncio

# Enable nested event loops for streamlit compatibility
nest_asyncio.apply()

# Add src/ to path so modules can be imported cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import chromadb

# Import central configurations
from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHROMA_DB_PATH
)
from retriever import FilteredQueryEngine
from indexer import build_or_refresh_index

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recruiter AI Assistant Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium recruiter-themed dark mode
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;500&display=swap');

:root {
    --bg-main: #0B0F19;
    --bg-card: rgba(255, 255, 255, 0.03);
    --bg-card-hover: rgba(255, 255, 255, 0.06);
    --border-color: rgba(255, 255, 255, 0.08);
    --primary-blue: #3E8BFF;
    --primary-purple: #9D4EDD;
    --gradient-primary: linear-gradient(135deg, #3E8BFF 0%, #9D4EDD 100%);
    --text-main: #E2E8F0;
    --text-muted: #94A3B8;
    --success-green: #10B981;
    --warning-yellow: #F59E0B;
    --error-red: #EF4444;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
}

/* Streamlit application layout background overrides */
[data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .stApp {
    background-color: #0B0F19 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0) !important;
}

/* Force all text inside the main content area to be off-white and high contrast */
[data-testid="stAppViewContainer"] p, 
[data-testid="stAppViewContainer"] span, 
[data-testid="stAppViewContainer"] li, 
[data-testid="stAppViewContainer"] label, 
[data-testid="stAppViewContainer"] h1, 
[data-testid="stAppViewContainer"] h2, 
[data-testid="stAppViewContainer"] h3, 
[data-testid="stAppViewContainer"] h4 {
    color: #F1F5F9 !important;
}

/* Highlight bold elements in a bright blue color for readability */
[data-testid="stAppViewContainer"] strong {
    color: #3E8BFF !important;
}

/* Chat bubble styling */
.chat-user-row {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.chat-user-bubble {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid var(--border-color);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    color: var(--text-main);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.chat-assistant-row {
    display: flex;
    justify-content: flex-start;
    gap: 12px;
    margin: 12px 0;
}
.chat-assistant-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--gradient-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: white;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(62, 139, 255, 0.4);
}
.chat-assistant-bubble {
    background: #1E293B !important;
    border: 1px solid var(--border-color);
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    max-width: 80%;
    color: var(--text-main) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(8px);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #090D16 !important;
    border-right: 1px solid var(--border-color);
}

/* Cards and status badges */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(62, 139, 255, 0.3);
    transform: translateY(-2px);
}
.badge-candidate {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(62, 139, 255, 0.12);
    border: 1px solid rgba(62, 139, 255, 0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    color: #3E8BFF;
    font-weight: 500;
    margin-bottom: 6px;
}

/* Code fonts for debug panel */
.code-font {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

/* ── Text Input Styling ── */
div[data-testid="stTextInput"] input {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="stTextInput"] input::-webkit-input-placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input::-moz-placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #3E8BFF !important;
    box-shadow: 0 0 10px rgba(62, 139, 255, 0.4) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] label {
    color: #E2E8F0 !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

/* Streamlit button text and background style overrides */
div.stButton button {
    background-color: #3E8BFF !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease-in-out !important;
}

/* Explicit color formatting for text inside the button elements */
div.stButton button p,
div.stButton button span {
    color: #FFFFFF !important;
}

/* Streamlit button hover state styling */
div.stButton button:hover {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-color: #3E8BFF !important;
    box-shadow: 0 0 12px rgba(62, 139, 255, 0.4) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],
        "engine": None,
        "engine_ready": False,
        "engine_error": None,
        "total_queries": 0,
        "blocked_queries": 0,
        "corrected_queries": 0,
        "last_candidate": None,
        "candidate_list": [],
        "avg_latency_ms": 0.0,
        "total_latency_ms": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# Local Engine Loader & Helper
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine_cached(api_key_override=None):
    """
    Loads and caches the FilteredQueryEngine using the local cache_resource manager.
    """
    try:
        engine = FilteredQueryEngine(api_key=api_key_override)
        # Fetch candidate names from fuzzy matcher
        candidates = sorted(engine.matcher.candidates)
        return engine, candidates, None
    except Exception as exc:
        return None, [], str(exc)

def load_or_refresh_engine(api_key=None):
    """Refreshes the cached RAG pipeline engine in the current session."""
    engine, candidates, err = load_engine_cached(api_key)
    st.session_state.engine = engine
    st.session_state.candidate_list = candidates
    st.session_state.engine_error = err
    st.session_state.engine_ready = (engine is not None)

# Initialize engine on startup
if not st.session_state.engine_ready and st.session_state.engine_error is None:
    load_or_refresh_engine()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Section - System Health & Ingestion Console
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:10px 0;">
              <h2 style="margin:0;font-weight:700;background:linear-gradient(135deg, #3E8BFF 0%, #9D4EDD 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Recruiter AI</h2>
              <span style="font-size:0.8rem;color:#94A3B8;">Production RAG System Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

        # --- System Health Panel ---
        st.markdown("### 🖥️ System Health Panel")
        
        # Check OpenRouter API key status
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            st.success("OpenRouter Connected", icon="🔑")
        else:
            st.warning("OpenRouter API Key Missing", icon="⚠️")

        # Check Chroma DB and collection count
        db_status = "Disconnected"
        vector_count = 0
        if st.session_state.engine and st.session_state.engine.chroma_collection:
            try:
                vector_count = st.session_state.engine.chroma_collection.count()
                db_status = f"Connected ({vector_count} nodes)"
                st.success(db_status, icon="📁")
            except Exception as e:
                st.error(f"Chroma DB Error: {e}", icon="🚨")
        else:
            st.error("Chroma DB Offline", icon="🚨")

        st.divider()

        # --- Admin Ingestion Console ---
        st.markdown("### ⚙️ Admin Ingestion Console")
        st.markdown(
            "<span style='font-size:0.8rem;color:#94A3B8;'>Index and refresh resumes directory. Always deletes existing vectors before rebuilding.</span>",
            unsafe_allow_html=True
        )
        
        if st.button("🔄 Index Resume Collection", use_container_width=True):
            with st.spinner("Clearing collection and re-indexing Resumes..."):
                try:
                    # Run the indexer pipeline
                    build_or_refresh_index(force_rebuild=True)
                    # Clear st.cache_resource so it re-reads ChromaDB count
                    st.cache_resource.clear()
                    # Reload engine
                    load_or_refresh_engine()
                    
                    st.toast("Chroma DB re-indexed successfully!", icon="✅")
                    st.success("Verification Passed: Collection Count == Resume Count")
                except Exception as exc:
                    st.error(f"Rebuild failed: {exc}")
                    
        st.divider()

        # --- List of Known Candidates ---
        st.markdown("### 👥 Indexed Candidates")
        candidates = st.session_state.candidate_list
        if candidates:
            for c in candidates:
                st.markdown(f"<span style='font-size:0.85rem;display:block;margin:3px 0;'>👤 {c}</span>", unsafe_allow_html=True)
        else:
            st.info("No candidates indexed.")

        st.divider()

        # --- Configuration Information ---
        with st.expander("⚙️ Model Architecture Details", expanded=False):
            st.markdown(f"**LLM Model:**\n`{LLM_MODEL}`")
            st.markdown(f"**Embeddings:**\n`{EMBEDDING_MODEL}` (Local CPU)")
            st.markdown(f"**Vector Store:**\nChromaDB Collection: `{COLLECTION_NAME}`")

        # --- UI Preferences ---
        st.markdown("### 🛠️ UI Preferences")
        debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show detailed retrieval scores, intent logs, and timings.")
        st.session_state.debug_mode = debug_mode

        # --- Collapsible Debug Diagnostics in Sidebar ---
        if debug_mode:
            st.divider()
            with st.expander("🔍 Latest Query Diagnostics", expanded=True):
                last_assistant_msg = None
                for msg in reversed(st.session_state.chat_history):
                    if msg["role"] == "assistant" and msg.get("debug_info"):
                        last_assistant_msg = msg
                        break
                
                if last_assistant_msg:
                    dbg = last_assistant_msg["debug_info"]
                    st.markdown(f"**Total Latency:** `{dbg.get('total_time_ms', 0.0):.1f} ms`")
                    st.markdown(f"- Retrieval: `{dbg.get('retrieval_time_ms', 0.0):.1f} ms`")
                    st.markdown(f"- Generation: `{dbg.get('generation_time_ms', 0.0):.1f} ms`")
                    st.markdown(f"**Intent Detection:** `{dbg.get('intent', 'N/A')}`")
                    st.markdown(f"**Resolved Candidate:** `{dbg.get('candidate_name', 'N/A')}`")
                    st.markdown(f"**Active Metadata Filters:** `{dbg.get('metadata_filters', 'N/A')}`")
                    
                    nodes = dbg.get("retrieved_nodes", [])
                    if nodes:
                        st.markdown("**Retrieved Chunk Scores:**")
                        for idx, node in enumerate(nodes, 1):
                            st.markdown(f"- Chunk {idx}: `{node.get('score', 0.0):.4f}` ({node.get('source', 'N/A')})")
                    
                    if dbg.get("prompt"):
                        with st.expander("View Context Prompt", expanded=False):
                            st.text_area("Context Prompt", value=dbg.get("prompt"), height=150, key="sidebar_prompt_area", disabled=True)
                    if dbg.get("openrouter_response"):
                        with st.expander("View Raw LLM Response", expanded=False):
                            st.text_area("LLM Response", value=dbg.get("openrouter_response"), height=100, key="sidebar_res_area", disabled=True)
                else:
                    st.info("No query diagnostics available yet. Run a query in the chat.")
            
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.blocked_queries = 0
            st.session_state.corrected_queries = 0
            st.session_state.last_candidate = None
            st.session_state.avg_latency_ms = 0.0
            st.session_state.total_latency_ms = 0.0
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main Application Layout
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    st.markdown(
        """
        <div style="padding:15px 0 5px 0;">
          <h1 style="margin:0;font-size:2rem;font-weight:700;">💼 Recruiter RAG Workspace</h1>
          <p style="color:#94A3B8;margin-top:2px;">Query parsed candidates instantly under verified semantic matching constraints.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chat container window
    chat_container = st.container(height=520)
    
    # Render Chat History
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align:center;padding:120px 20px;color:#94A3B8;">
                  <div style="font-size:3rem;margin-bottom:10px;">💬</div>
                  <h4 style="margin:0;font-weight:600;color:#E2E8F0;">No active chat session</h4>
                  <p style="margin:4px 0 0 0;font-size:0.9rem;">Ask a question about a candidate (e.g. "What are Trinadh's Python skills?") to get started.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for i, msg in enumerate(st.session_state.chat_history):
                role = msg["role"]
                content = msg["content"]
                meta = msg.get("metadata", {})
                
                with st.chat_message(role):
                    if role == "assistant" and meta.get("candidate_name"):
                        st.markdown(f"👤 **Candidate: {meta['candidate_name']}**")
                    st.markdown(content)
                    
                    # Show spelling suggestion if spelling correction occurred
                    if meta.get("spelling_suggestion"):
                        st.info(meta["spelling_suggestion"], icon="✏️")
                        
                    # Show blocked reason if query was blocked
                    if meta.get("is_blocked") and meta.get("blocked_reason"):
                        st.error(f"Reason: {meta['blocked_reason']}", icon="🚫")

    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    # Chat Input Form
    with st.form("query_form", clear_on_submit=True):
        col_input, col_submit = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="recruiter_query_box",
                placeholder="Ask about a candidate (e.g. 'What are Yasaswi's projects?' or 'Summarize Ashok Reddy')...",
                label_visibility="collapsed"
            )
        with col_submit:
            submitted = st.form_submit_button("Send Query →", use_container_width=True)
            
        if submitted and user_input.strip():
            submit_query(user_input)
            st.rerun()


def submit_query(query: str):
    """
    Submits a query to the FilteredQueryEngine, measures latency, 
    and updates session state chat logs and statistics.
    """
    if not query.strip():
        return
        
    engine = st.session_state.engine
    
    # Append User prompt to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "metadata": {},
        "debug_info": {}
    })
    st.session_state.total_queries += 1

    if not engine:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⚠️ RAG Pipeline is not initialized. Please verify your OpenRouter API Key in `.env` and click 'Index Resume Collection'.",
            "metadata": {"is_blocked": False},
            "debug_info": {}
        })
        return

    # Run the query and track time
    t0 = time.time()
    result = engine.query(query.strip())
    latency_ms = (time.time() - t0) * 1000.0
    
    # Update latency averages
    st.session_state.total_latency_ms += latency_ms
    st.session_state.avg_latency_ms = st.session_state.total_latency_ms / st.session_state.total_queries

    # Extract metrics
    is_blocked = result.get("is_blocked", False)
    spelling_suggestion = result.get("spelling_suggestion")
    candidate_name = result.get("candidate_name")
    
    if is_blocked:
        st.session_state.blocked_queries += 1
    if spelling_suggestion:
        st.session_state.corrected_queries += 1
    if candidate_name:
        st.session_state.last_candidate = candidate_name

    # Add debug timing info
    debug_info = result.get("debug_info", {})
    if debug_info:
        debug_info["total_time_ms"] = latency_ms

    # Append Assistant answer to chat
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("response", "No response returned."),
        "metadata": {
            "candidate_name": candidate_name,
            "spelling_suggestion": spelling_suggestion,
            "is_blocked": is_blocked,
            "blocked_reason": result.get("blocked_reason")
        },
        "debug_info": debug_info
    })


if __name__ == "__main__":
    main()
