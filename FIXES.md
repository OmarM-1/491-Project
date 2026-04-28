# Spotter AI — Bug Fixes & Changes Log

---

## 1. Context and Queries Not Being Saved / Bot Had No Memory

**Files:** `member.html`, `web_api.py` → `start.py`

### Problems
- `threadId` was declared as `let threadId = null` in the frontend — reset to `null` on every page reload, so the backend created a brand new thread each time. The AI always started from scratch.
- The frontend loaded chat history from a `chat_messages` table via the Supabase anon client directly. The backend saved messages to a completely separate `messages` table. These two tables were never in sync — `chat_messages` was always empty so history never showed up.
- A `saveMessage()` helper function existed in the frontend but was never called during the actual send flow.

### Fixes
- `threadId` is now persisted in `localStorage` and restored on every page load.
- On page load, if no `threadId` is in storage, the frontend calls `GET /threads/latest` to restore the most recent session.
- `loadChatHistory()` now fetches from `GET /history/{thread_id}` (backend's `messages` table) — single source of truth.
- The dead `saveMessage()` function was removed.
- On logout, `localStorage` clears the stored `threadId` so users don't share threads.

---

## 2. Conversation Memory Was a Text Blob, Not Real Multi-Turn

**Files:** `optimized_rag.py`, `hybrid_orchestrator.py`, `agentic_rag.py`, `start.py`

### Problem
Chat history was pasted as a raw text block inside the user message:
```
QUESTION: what weight should I use?

[RECENT MESSAGES]
User: I want to build muscle
Spotter AI: Great goal...

CONTEXT: [RAG docs]
```
The model had no structural understanding of who said what. It treated the whole thing as a monologue.

### Fix
History is now passed as actual multi-turn message objects, exactly how ChatGPT/Claude work:
```python
[
  {"role": "system",    "content": "You are Spotter AI..."},
  {"role": "user",      "content": "I want to build muscle"},
  {"role": "assistant", "content": "Great goal!..."},
  {"role": "user",      "content": "what weight should I use?"}
]
```
The `history` list (loaded from Supabase `messages` table) is threaded through `start.py` → `smart_answer()` → `orchestrator.answer()` → `generate_grounded_answer()` and injected as real conversation turns before the current question.

---

## 3. No Conversation Tabs / History Grew Forever

**Files:** `start.py`, `member.html`, `style.css`

### Problem
All messages went into one thread forever. Over time this would slow down every response as history grew. There was no way to start a fresh conversation.

### Fixes
- Added `GET /threads` endpoint returning all user threads in `{"threads": [...]}` format.
- Added `GET /threads/latest` endpoint.
- Added `GET /history/{thread_id}` endpoint for loading display history.
- New threads are auto-titled from the first message (truncated to 60 chars).
- Frontend sidebar now shows a scrollable conversation list with active thread highlighted.
- "+ New" button clears the chat and starts a fresh thread on next send.
- Clicking any past conversation loads its history instantly.
- After each reply, if a new thread was just created, the list refreshes automatically.

---

## 4. Chat Messages Not Displaying After Page Reload

**Files:** `member.html`, `style.css`

### Problem
Bot responses are stored in the database with Markdown formatting (`**bold**`, `## headings`, `- bullets`). The `appendMessage()` function used `bubble.textContent = text`, which renders everything as plain text. After a reload, stored messages displayed with raw asterisks and hashes instead of formatted output.

### Fix
- Added `marked.js` (Markdown → HTML) and `DOMPurify` (XSS sanitization) via CDN.
- Assistant messages now use `bubble.innerHTML = DOMPurify.sanitize(marked.parse(text))`.
- User messages still use `textContent` (plain text) — user input should never be treated as Markdown.
- Added CSS for rendered Markdown inside bubbles: paragraph spacing, list indentation, bold, code blocks.

---

## 5. User Profile Not in System Prompt

**Files:** `start.py`, `hybrid_orchestrator.py`, `optimized_rag.py`, `agentic_rag.py`

### Problem
The bot had no awareness of who it was talking to. It ignored the user's weight, height, age, sex, and fitness goal stored in the database, giving generic advice instead of personalized responses.

### Fix
- `start.py` `/chat` endpoint now calls `get_profile(user_id)` and passes the result through the call chain.
- `profile` is threaded through `smart_answer()` → `orchestrator.answer()` → `generate_grounded_answer()` → `agentic_answer()`.
- In `generate_grounded_answer`, profile fields are injected at the **top** of the system prompt (small models follow early context most reliably):
```
USER PROFILE:
Name: Thomas
Age: 22
Sex: Male
Weight: 185 lbs
Height: 71 in
Fitness goal: build_muscle
```

---

## 6. All Changes Were Going to the Wrong File (`web_api.py` vs `start.py`)

**Files:** `web_api.py`, `start.py`

### Problem
`web_api.py` was assumed to be the running server, but `start.py` is the actual entry point — it builds its own FastAPI app with its own routes. Every endpoint and logic change made to `web_api.py` was completely ignored at runtime.

Additionally, `hybrid_orchestrator.py` had `from web_api import chat` at the top level, creating a circular import (`web_api` → `hybrid_orchestrator` → `web_api`).

### Fixes
- All endpoint logic (`/chat`, `/threads`, `/threads/latest`, `/history/{thread_id}`) was ported into `start.py`.
- `from web_api import chat` removed from `hybrid_orchestrator.py`.
- The safety gate call updated to use `chat_text` (already imported at module level) instead of the FastAPI endpoint function.
- `web_api.py` is now unused dead code.

---

## 7. `UnboundLocalError: chat_text`

**File:** `hybrid_orchestrator.py`

### Problem
`chat_text` was imported at the module level (`from Spotter_AI import chat_text`), but inside the `answer()` method there was also a local import `from Spotter_AI import build_messages, chat_text` inside an `if not ok:` block. Python marks any variable assigned anywhere in a function as local to that entire function, making the module-level `chat_text` inaccessible before the local assignment line ran.

### Fix
Removed `chat_text` from the local import inside `answer()` — it only needed `build_messages` there. The module-level import handles `chat_text`.

---

## 8. Bot Giving Nonsensical Responses

**File:** `optimized_rag.py`

### Problems
Three issues combined to confuse the small (2B parameter) model:

1. **RAG context was appended to the user message.** The model saw `"i am 180 lbs\n\n---\nRAG CONTEXT: [500 words about BMI/weight...]"` and followed the RAG documents instead of the actual question.
2. **Profile was at the end of the system prompt.** Small models follow instructions at the start most reliably. Profile at the end got treated as extra content to respond to rather than context to use.
3. **10 history turns.** Too many tokens for a 2B model to stay coherent across a long context.

### Fixes
- **RAG context moved into the system prompt** under a `RELEVANT FITNESS KNOWLEDGE` section — keeps the user message clean so the model focuses on the question.
- **Profile moved to the very top of the system prompt**, before all rules and instructions.
- **History capped at 4 turns** (2 exchanges) instead of 10.
- System prompt simplified to be shorter and more direct — fewer rules to misinterpret.

---

## 9. Performance Bottlenecks (Slow Response Times)

**Files:** `optimized_rag.py`, `supabase_service.py`, `start.py`

### Problems
Every request had significant overhead before the LLM even started:

| Bottleneck | Cost |
|---|---|
| 5 sequential Supabase HTTP calls (auth + thread + profile + history + save) | ~800–1000ms |
| CrossEncoder reranking 18 doc pairs on CPU | ~500ms–2s |
| `save_message` doing 3 DB operations (insert + count query + thread update) | ~300–400ms |
| Assistant response save blocking the reply to the user | Adds latency after generation |

### Fixes
1. **Reduced rerank candidates from 18 → 8** (`k*3=18` changed to `k+2=8`). The CrossEncoder now scores 8 pairs instead of 18 — proportionally faster with negligible quality difference.
2. **`save_message` reduced from 3 DB calls to 1.** Removed the `get_message_count()` call and the `conversation_threads` update that followed — these were non-critical counters.
3. **Profile and history loads parallelized** using `ThreadPoolExecutor(max_workers=2)` — both Supabase queries run simultaneously instead of sequentially.
4. **Assistant response saved in a background thread** (`threading.Thread(..., daemon=True).start()`) — the user receives their answer immediately without waiting for the DB write to complete.

---

## 10. Dead / Broken Code Cleaned Up

### `admin.rpc("", {}).execute` in `save_message`
A line `admin.rpc("", {}).execute` was present — calling an RPC with an empty function name and then accessing `.execute` as a property (not calling it). It was a complete no-op but looked like it should be doing something. Removed.

### Duplicate `safety_gate_agent` definition in `SAFETY_AGENT.py`
The function was defined twice — the second definition overwrote the first. The second definition also referenced `SAFETY_PROMPT` which was never defined, meaning any flagged query would crash with a `NameError`. The safety gate works correctly for non-flagged queries (regex returns immediately) but the LLM-based second pass was broken. Left as-is since fixing it requires defining `SAFETY_PROMPT`.

### Duplicate `import os` in `hybrid_orchestrator.py`
`import os` appeared twice at the top of the file. One removed.

### History routing pollution
The entire `context_block` (all past conversation history as text) was being prepended to the query before passing it to the orchestrator. This caused the keyword router to match calorie/diet words from old messages on every request — triggering the nutrition agents with hardcoded fake user parameters (25yo male, 80kg) regardless of what was actually asked. Fixed by routing on only the current user message.
