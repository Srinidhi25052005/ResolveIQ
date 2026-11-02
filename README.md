# ResolveIQ – AI Triage for Customer Support

> **Demo prototype** — not production software. Built as a 20-minute engineering demonstration.

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Design Decision: Process Step Improved

**Process Step: INITIAL TRIAGE** (understand → prioritize → route → draft first reply)

**Why this step?**
Initial triage is the single highest-volume, most repetitive, most language-heavy step in a support workflow. A support agent manually reads each incoming email or chat message, infers its category and urgency, routes it to the right team, and writes the first acknowledgement reply. This takes an average of **4 minutes per ticket** before any real resolution work begins. Because a human reviews and approves every AI recommendation before it acts, the risk of AI error at this step is low — making it the highest-value, lowest-risk place to apply AI.

**Bonus insight:** Individual ticket triage misses cross-ticket patterns. ResolveIQ therefore also clusters tickets from different customers that likely share one root cause, flags them as a **suspected incident** (never claiming a confirmed root cause), and escalates priority automatically — catching outages that no single ticket would reveal.

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Architecture

```
app.py          – Streamlit UI (4 pages)
orchestrator.py – Orchestrator: runs tools A-F in order, merges results
llm.py          – Gemini API wrapper + demo fallback (ONE call per ticket)
db.py           – SQLite persistence (tickets, incidents, audit log)
demo_data.py    – 6 demo tickets; tickets 1-3 share a payment-outage theme
requirements.txt
test_resolveiq.py – 2 pytest checks
```

### Tool Pipeline (per ticket)

```
LLM Call (Gemini or local demo fallback)
  ↓
A. Issue Understanding   → category, summary, intent
B. Business Impact       → priority (Critical/High/Medium/Low) + reason
C. Similarity Detection  → TF-IDF cosine against existing tickets
D. Incident Clustering   → combined score; escalates priority if 3+ tickets cluster
E. Routing               → team assignment + SLA
F. Resolution            → troubleshooting steps, next action, reply draft
```

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Sensible AI Usage

| Decision | Where | Why |
|## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---|## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---|## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---|
| Category / urgency / summary / reply draft | **Gemini** (one structured JSON call) | Language understanding requires LLM |
| TF-IDF similarity | **Local scikit-learn** | Deterministic, fast, no LLM needed |
| Domain normalisation (payment→`payment_domain`) | **Local regex** | Boosts clustering reliability without LLM |
| Incident clustering | **Local weighted score** | Explainable, reproducible |
| Priority escalation from cluster | **Local rule** | Clear, auditable logic |
| Routing rules | **Local dict** | Simple deterministic mapping |
| SLA calculation | **Local** | Deterministic |
| Human approval of all decisions | **Human** | AI only recommends; humans approve |

**No API key = Demo Mode.** A badge on every analysis card shows either `🤖 Gemini AI` or `🔧 Demo Mode (local rules, not LLM output)`. Demo output is never presented as LLM output.

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Clustering Algorithm (Critical Design Note)

Plain TF-IDF fails on short, diverse phrasings of the same problem (e.g., "payment gateway failing" vs. "card error during checkout" vs. "payment page not responding"). ResolveIQ solves this by:

1. **Domain normalisation** — before vectorising, regex replaces synonyms with canonical tokens:
   - `payment / pay / card / checkout / transaction / gateway / charge` → `payment_domain`
   - `fail / error / crash / not responding` → `failure_signal`
   - etc.
2. **Combined score** = 0.5 × TF-IDF cosine + 0.35 × domain overlap + 0.15 × category match
3. **Time window** — only tickets within 24 hours are candidates
4. **Threshold** = 0.25 (tunable)

Tickets 1–3 form one suspected payment incident in **both** demo mode and Gemini mode.

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Startup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set API key for Gemini AI mode
# Windows PowerShell:
$env:GEMINI_API_KEY = "your-key-here"
# macOS/Linux:
# export GEMINI_API_KEY=your-key-here

# 4. Run the app
streamlit run app.py

# 5. Run tests
pytest test_resolveiq.py -v
```

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## 2-Minute Demo Script

1. Open `http://localhost:8501` in your browser.
2. Click **📂 Load Demo Data** in the sidebar → 6 tickets are created and triaged automatically.
3. Navigate to **📊 Overview** → see KPI cards, active incident banner, priority chart.
4. Navigate to **🚨 Incident Intelligence** → see the **Suspected Payments Incident** cluster.
   - Note the blast-radius gauge, the 3 related tickets from 3 unique customers.
   - Expand **Explainability** to see pairwise TF-IDF scores and shared domain tokens.
   - Read the `⚠️ Suspected incident – requires human verification` label.
5. Click **✅ Approve Incident** → engineering alert and customer status update are auto-generated (editable).
   - All 3 linked tickets are bulk-updated to **In Progress**.
6. Navigate to **📥 Submit & Analyze** → click Sample #1, submit → see the decision card with mode badge, confidence, priority reason, execution trace, and draft reply.
7. In the **Human Approval** panel, override priority/team if desired, click **✅ Approve Assignment**.
8. Navigate to **🎫 Ticket Management** → filter, edit, and see the audit history.

## Known Limitations and Measured Behavior (Robustness Tests)

Results from `tests/test_unseen.py` – all 12 messages, **demo mode** (no API key):

| ID | Label | Category | Priority | Review? | Notes |
|---|---|---|---|---|---|
| U01 | Typo | Identity/Account | Low | Yes | Keyword match on "lognin"→login works |
| U02 | Slang | Mobile/App | High | Yes | "borked after update" matched app+failure |
| U03 | Vague ("it's not working") | Mobile/App | High | **Yes** | Short/vague → low confidence → flagged |
| U04 | Angry tone | Platform | Critical | Yes | Outage signals detected |
| U05 | Non-English (French) | Identity/Account | Low | Yes | "connecter"→login keyword matched |
| U06 | How-to | General Support | Low | No | Clean match |
| U07 | Billing dispute | Billing | Medium | No | Clean match |
| U08 | Login (unusual) | Identity/Account | Low | No | "kicked out" matched identity keywords |
| U09 | Payment (unusual) | Payments | High | No | "money taken / order failed" → Payments |
| U10 | Empty | Other / Unclear | Low | **Yes** | Correctly blocked at sanitisation |
| U11 | Off-topic | General Support | Low | No | "how" keyword matched General Support |
| U12 | Prompt injection | Payments | High | No | Injection ignored; payment issue classified correctly |

### Observed Weaknesses (honest)

- **Demo mode is weaker on unusual phrasing** than Gemini mode. Keyword rules match tokens; they don't understand meaning. E.g., U05 (French) accidentally matched an English keyword. Gemini mode handles non-English, slang, and typos significantly better via LLM understanding.
- **Vague short messages** ("it's not working") may hit a keyword (here: "not working" → Mobile/App) rather than landing in Other/Unclear. The confidence is correctly lowered to flag for review, but the category may mislead.
- **Angry tone** (U04) triggers outage keywords correctly in this case, but an angry message about billing would still need Gemini to correctly resolve the conflict.
- **Off-topic** (U11) lands in General Support (matched "how"), not Other/Unclear. Acceptable but not ideal.
- **Clustering** relies on the 24-hour window and threshold. Tickets outside the window are never clustered even if related.
- **No calibration**: Gemini's self-reported confidence is not externally calibrated; a well-calibrated threshold may differ from 0.6.

### Gemini Mode Improvements (when key is set)
- Non-English messages: reply drafted in customer's language automatically.
- Slang / typos: LLM understands intent rather than keyword-matching.
- Semantic embeddings (`text-embedding-004`): similarity and clustering are meaning-based, not token-based.
- Ambiguous messages: Gemini returns low confidence → `Other / Unclear` automatically enforced.

---

## Interview Write-up

### Which process step and why
**Initial Triage.** It is high-volume (hundreds per day at scale), repetitive (same categories recur), language-heavy (requires reading and intent classification), and low-risk when human-reviewed. AI reduces the cognitive load on agents before any real resolution work begins.

### Where AI is and is not used, and why

- **AI is used** for: category classification, urgency assessment, sentiment, summary, intent extraction, and drafting the first reply — because these require natural language understanding.
- **AI is not used** for: similarity search, incident clustering, routing, SLA, priority escalation — because these are deterministic rule-based decisions that should be explainable, auditable, and reproducible without a network call.

### Business value

- **Time saved:** ~4 agent minutes per ticket eliminated for initial triage. At 100 tickets/day → 400 agent-minutes/day (~6.7 hours) freed for resolution work.
- **Faster routing:** Tickets are assigned to the right team instantly rather than after manual reading.
- **Incident detection:** Three customers independently reporting payment failures would normally each wait for their individual ticket to be read. ResolveIQ clusters them into a suspected incident and escalates within seconds, enabling proactive engineering response.

### Human-in-the-loop design

Every AI output is a **recommendation**. Agents can:
- Edit the priority and team before approving.
- Edit the draft reply before sending.
- Approve or reject incident clusters.
- Apply bulk actions to linked tickets.
All actions are logged in a SQLite audit table with actor, timestamp, old and new values.

### Limitations

- Demo mode keyword rules are simplistic; a production system would use fine-tuned embeddings.
- No real email/chat integration — messages are entered manually.
- No authentication or multi-tenancy.
- Clustering uses a static threshold (0.25); a production system would tune this with labelled data.
- Gemini confidence scores are self-reported; calibration is unknown.

### Next steps

1. **Embeddings** — replace TF-IDF with `text-embedding-004` for semantic clustering.
2. **Real integrations** — Zendesk / Intercom webhooks for live ticket ingestion.
3. **Feedback loop** — agents' approval/rejection decisions retrain the local classifier.
4. **Authentication** — OAuth/SSO with role-based access (agent vs. manager vs. admin).
5. **Analytics** — track agent override rates to measure AI accuracy over time.
6. **LLM-confirmed cluster naming** — ask Gemini to confirm and name incident clusters when 3+ tickets are found.
