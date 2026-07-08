# Implementation Plan — Recruiter Dashboard UI Simplification & Readability Upgrade

This document outlines the proposed changes to the Streamlit frontend layout and CSS variables to improve text readability, simplify the chat workflow, and hide debug clutter from the main recruiter interface.

---

## Proposed Changes

### `app.py` UI Simplification & Accessibility Tuning

We will modify [app.py](file:///e:/Resume-Rag-LLM/app.py) as follows:

#### 1. Color Contrast & Accessibility Adjustments (CSS)
*   Update input box styles to use a dark-gray background (`#1E293B`), high-contrast white text (`#FFFFFF`), and a bright blue border focus glow to resolve the similar-range color reading issue.
*   Increase contrast of body and metadata label texts by shifting muted colors to brighter off-white and slate tones (e.g. `#CBD5E1` and `#E2E8F0`).
*   Ensure text color inside input placeholders has adequate contrast.

#### 2. Main Interface Simplification
*   **Remove Workspace Tabs**: Delete the "Candidate Resumes Inspector" and "System Performance Metrics" tabs. The main area will be dedicated solely to the chat interface.
*   **Remove Quick Questions**: Delete the "⚡ Quick Questions" buttons underneath the chat log to prevent clutter.
*   **Clean Chat Window**: Show only user queries and recruiter responses. 

#### 3. Redesigned Sidebar & Debug Panel
*   Keep the Admin controls (reindexing, candidates list, system health status) in the sidebar.
*   **Introduce 'Debug Mode' Checkbox**: Add a checkbox in the sidebar titled "🛠️ Enable Debug Mode".
*   If **Debug Mode** is enabled, show the timing statistics, local intents, resolved candidate name, similarity scores, and prompts as a clean collapsible code block at the bottom of the sidebar or under the chat query.
*   If **Debug Mode** is disabled, completely hide all retrieved chunks, similarity scores, prompt text areas, and timing statistics.

---

## Verification Plan

1. Launch Streamlit: `.venv\Scripts\streamlit run app.py`
2. Verify the main page is a clean single-pane chat interface (no tabs, no quick query buttons).
3. Test color contrast inside the chat query text box: verify background is dark gray and typed text is high-contrast white.
4. Run a query (e.g., "Summarize Trinadh's resume"):
   *   Verify no debug panels or text areas appear under the response in the chat flow.
   *   Toggle the **Debug Mode** checkbox in the sidebar: verify the diagnostic log (intent, metadata filters, timings) appears.
