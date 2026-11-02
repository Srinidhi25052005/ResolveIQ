"""
orchestrator.py – Runs tools A-F in order and returns a merged decision dict.
v2 additions: Gemini embeddings for similarity/clustering, first-reporter priority,
              Other/Unclear handling, sidebar-tunable threshold.
"""
import json, re, hashlib
from datetime import datetime
from typing import Any

import db
import llm as llm_module

# ── Domain normalisation for TF-IDF fallback clustering ──────────────────────
DOMAIN_PATTERNS = [
    (r"\b(payment|pay|card|checkout|transaction|gateway|charge|money|purchase)\b", "payment_domain"),
    (r"\b(fail|failing|failed|error|crash|crashes|not responding|broken|not working|kicked out)\b", "failure_signal"),
    (r"\b(login|password|sign in|signin|account|reset|access|cant get in)\b", "identity_domain"),
    (r"\b(subscription|billing|invoice|refund|charged)\b", "billing_domain"),
    (r"\b(app|application|mobile|ios|android|update|keeps closing)\b", "app_domain"),
    (r"\b(server|platform|outage|down|unavailable|site down|everyone|all customers)\b", "platform_domain"),
]

def normalize_text(text: str) -> str:
    """Replace domain terms with canonical tokens before TF-IDF."""
    t = text.lower()
    for pattern, token in DOMAIN_PATTERNS:
        t = re.sub(pattern, f" {token} ", t, flags=re.IGNORECASE)
    return t

# ── Routing rules ─────────────────────────────────────────────────────────────
ROUTING = {
    "Payments":         "Payments",
    "Billing":          "Billing",
    "Platform":         "Platform",
    "Mobile/App":       "Mobile/App",
    "Identity/Account": "Identity/Account",
    "General Support":  "General Support",
    "Other / Unclear":  "General Support",
}

# ── SLA minutes by priority ───────────────────────────────────────────────────
SLA = {"Critical": 15, "High": 60, "Medium": 240, "Low": 1440}

# ── Default clustering threshold (overridden by sidebar) ─────────────────────
DEFAULT_CLUSTER_THRESHOLD = 0.25
INCIDENT_WINDOW_HOURS     = 24


# ══════════════════════════════════════════════════════════════════════════════
# Outage-signal first-reporter rule (req. #4)
# ══════════════════════════════════════════════════════════════════════════════
_OUTAGE_SIGNALS = [
    "all customers", "everyone", "site down", "nobody can", "gateway failing",
    "widespread", "complete outage", "entire system", "all users", "no one can",
    "totally down", "system wide", "platform down", "cannot complete transactions",
]

def _has_outage_signal(message: str) -> bool:
    ml = message.lower()
    return any(s in ml for s in _OUTAGE_SIGNALS)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding-based similarity (Gemini) with TF-IDF fallback
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: list, b: list) -> float:
    """Pure-Python cosine similarity for two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)

def _get_or_create_embedding(ticket_id: str, message: str) -> list | None:
    """Return cached embedding or call API and cache it."""
    cached = db.get_embedding(ticket_id)
    if cached:
        return cached
    vec = llm_module.get_embedding(message)
    if vec:
        db.save_embedding(ticket_id, vec)
    return vec

def _similarity_embedding(ticket_id: str, message: str) -> tuple[dict, str]:
    """
    Embedding-based similarity. Returns (result_dict, method_label).
    Falls back to TF-IDF if embeddings unavailable.
    """
    # Get current ticket embedding
    curr_vec = _get_or_create_embedding(ticket_id, message)
    if curr_vec is None:
        result, _ = _similarity_tfidf(ticket_id, message)
        return result, "TF-IDF (local)"

    all_tickets = db.get_all_tickets()
    others = [t for t in all_tickets if t["id"] != ticket_id and t.get("message")]
    if not others:
        return {"similar_tickets": [], "top_score": 0.0}, "Embeddings (Gemini)"

    matches = []
    for t in others:
        other_vec = _get_or_create_embedding(t["id"], t["message"])
        if other_vec is None:
            continue
        score = _cosine(curr_vec, other_vec)
        if score > 0.01:
            matches.append({
                "ticket_id":       t["id"],
                "score":           round(score, 3),
                "category":        t.get("category", ""),
                "message_preview": t["message"][:80],
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    top3    = matches[:3]
    top_score = top3[0]["score"] if top3 else 0.0
    return {"similar_tickets": top3, "top_score": top_score}, "Embeddings (Gemini)"

def _similarity_tfidf(ticket_id: str, message: str) -> tuple[dict, str]:
    """TF-IDF cosine similarity. Returns (result_dict, method_label)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        all_tickets = db.get_all_tickets()
        others = [t for t in all_tickets if t["id"] != ticket_id and t.get("message")]
        if not others:
            return {"similar_tickets": [], "top_score": 0.0}, "TF-IDF (local)"

        corpus    = [normalize_text(t["message"]) for t in others] + [normalize_text(message)]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf     = vectorizer.fit_transform(corpus)
        scores    = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

        top_idx = np.argsort(scores)[::-1][:3]
        matches = []
        for i in top_idx:
            if scores[i] > 0.01:
                matches.append({
                    "ticket_id":       others[i]["id"],
                    "score":           round(float(scores[i]), 3),
                    "category":        others[i].get("category", ""),
                    "message_preview": others[i]["message"][:80],
                })
        top_score = round(float(scores[top_idx[0]]), 3) if len(top_idx) else 0.0
        return {"similar_tickets": matches, "top_score": top_score}, "TF-IDF (local)"
    except Exception as e:
        return {"similar_tickets": [], "top_score": 0.0, "error": str(e)}, "TF-IDF (local)"

def tool_c_similarity(ticket_id: str, message: str) -> tuple[dict, str]:
    """Route to embedding or TF-IDF similarity based on API availability."""
    if llm_module.is_gemini_ready():
        return _similarity_embedding(ticket_id, message)
    return _similarity_tfidf(ticket_id, message)


# ══════════════════════════════════════════════════════════════════════════════
# Clustering (embedding or TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════

def _domain_overlap(msg_a: str, msg_b: str) -> float:
    def domains(text):
        tokens = set()
        for _, token in DOMAIN_PATTERNS:
            if re.search(token, normalize_text(text)):
                tokens.add(token)
        return tokens
    da, db_ = domains(msg_a), domains(msg_b)
    if not da and not db_:
        return 0.0
    return len(da & db_) / max(len(da | db_), 1)

def tool_d_clustering(ticket_id: str, message: str, category: str,
                       created_at: str, threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> dict:
    """
    Groups tickets likely sharing one root cause.
    Uses Gemini embeddings when available, otherwise normalised TF-IDF + domain overlap.
    threshold is tunable from the UI sidebar.
    """
    try:
        all_tickets = db.get_all_tickets()
        now = datetime.fromisoformat(created_at)

        # Filter candidates: exclude self, apply time window
        candidates = []
        for t in all_tickets:
            if t["id"] == ticket_id or not t.get("message"):
                continue
            try:
                t_time = datetime.fromisoformat(t["created_at"])
                if abs((now - t_time).total_seconds()) > INCIDENT_WINDOW_HOURS * 3600:
                    continue
            except Exception:
                pass
            candidates.append(t)

        if not candidates:
            return {"incident_detected": False, "cluster": [], "cluster_method": "TF-IDF (local)"}

        use_embeddings = llm_module.is_gemini_ready()
        curr_vec = _get_or_create_embedding(ticket_id, message) if use_embeddings else None
        if use_embeddings and curr_vec is None:
            use_embeddings = False  # fall back gracefully

        # TF-IDF scores (used as part of combined score in non-embedding mode,
        # or as fallback completely)
        tfidf_scores = {}
        if not use_embeddings:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                corpus     = [normalize_text(t["message"]) for t in candidates] + [normalize_text(message)]
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                tfidf      = vectorizer.fit_transform(corpus)
                scores     = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
                tfidf_scores = {candidates[i]["id"]: float(scores[i]) for i in range(len(candidates))}
            except Exception:
                tfidf_scores = {}

        cluster_members = []
        for t in candidates:
            if use_embeddings:
                other_vec = _get_or_create_embedding(t["id"], t["message"])
                if other_vec is None:
                    continue
                emb_score    = _cosine(curr_vec, other_vec)
                domain_score = _domain_overlap(message, t["message"])
                same_cat     = 1.0 if t.get("category") == category else 0.0
                combined     = 0.6 * emb_score + 0.25 * domain_score + 0.15 * same_cat
            else:
                tfidf_score  = tfidf_scores.get(t["id"], 0.0)
                domain_score = _domain_overlap(message, t["message"])
                same_cat     = 1.0 if t.get("category") == category else 0.0
                combined     = 0.5 * tfidf_score + 0.35 * domain_score + 0.15 * same_cat

            if combined >= threshold:
                cluster_members.append({
                    "ticket_id":    t["id"],
                    "score":        round(combined, 3),
                    "domain_score": round(domain_score, 3),
                })

        incident_detected  = len(cluster_members) >= 2   # + current = 3 total
        cluster_method     = "Embeddings (Gemini)" if use_embeddings else "TF-IDF (local)"
        result = {
            "incident_detected": incident_detected,
            "cluster":           cluster_members,
            "cluster_size":      len(cluster_members) + 1,
            "cluster_method":    cluster_method,
        }
        if incident_detected:
            result["escalate_reason"] = (
                f"Cluster of {len(cluster_members)+1} tickets (via {cluster_method}) "
                "likely sharing one issue. Priority escalated to Critical. "
                "Suspected incident – requires human verification."
            )
        return result
    except Exception as e:
        return {"incident_detected": False, "cluster": [], "error": str(e),
                "cluster_method": "TF-IDF (local)"}


# ══════════════════════════════════════════════════════════════════════════════
# Other tools (unchanged logic, minor additions)
# ══════════════════════════════════════════════════════════════════════════════

def tool_a_understanding(message: str, customer_name: str, ai_result: dict) -> dict:
    return {
        "category": ai_result.get("category", "Other / Unclear"),
        "summary":  ai_result.get("summary", message[:120]),
        "intent":   ai_result.get("intent", "Unknown"),
    }

def tool_b_impact(ai_result: dict, category: str, message: str) -> dict:
    """
    Determine priority and reason.
    First-reporter rule: outage signals → at least High even without a cluster.
    """
    urgency = ai_result.get("urgency", "Medium")

    # First-reporter escalation (req. #4)
    first_reporter_reason = ""
    if _has_outage_signal(message) and urgency not in ("Critical", "High"):
        urgency = "High"
        first_reporter_reason = (
            " Elevated to High: message contains outage-signal language "
            "('all customers', 'site down', etc.) — first-reporter rule applied."
        )

    reasons = {
        "Critical": "Major outage or widespread disruption affecting multiple customers.",
        "High":     "Significant functionality failure impacting the customer's workflow.",
        "Medium":   "Individual technical or billing issue requiring attention.",
        "Low":      "General information request, no service impact.",
    }
    return {
        "priority":        urgency,
        "priority_reason": reasons.get(urgency, "Standard triage.") + first_reporter_reason,
        "sla_minutes":     SLA.get(urgency, 240),
    }

def tool_e_routing(category: str, priority: str) -> dict:
    team = ROUTING.get(category, "General Support")
    return {"team": team, "sla_minutes": SLA.get(priority, 240)}

def tool_f_resolution(ai_result: dict, customer_name: str) -> dict:
    steps = ai_result.get("troubleshooting_steps", [])
    if isinstance(steps, str):
        steps = [steps]
    reply = ai_result.get("reply_draft", "")
    if customer_name and "{name}" in reply:
        reply = reply.replace("{name}", customer_name)
    return {
        "troubleshooting_steps": steps,
        "next_action":           "Assign to team and await customer response.",
        "reply_draft":           reply,
    }

# ── Blast-radius scorer ───────────────────────────────────────────────────────
def blast_radius(ticket_ids: list[str]) -> float:
    if not ticket_ids:
        return 0.0
    tickets = [db.get_ticket(tid) for tid in ticket_ids if db.get_ticket(tid)]
    n_tickets        = len(tickets)
    unique_customers = len({t.get("customer_name", "?") for t in tickets})
    priorities       = [t.get("priority", "Low") for t in tickets]
    priority_weight  = max(
        {"Critical": 40, "High": 30, "Medium": 15, "Low": 5}.get(p, 5)
        for p in priorities
    )
    return round(min(100, priority_weight + n_tickets * 8 + unique_customers * 5), 1)

# ── Incident creation/update ──────────────────────────────────────────────────
def _make_incident_id(ticket_ids: list[str]) -> str:
    key = "-".join(sorted(ticket_ids))
    return "INC-" + hashlib.md5(key.encode()).hexdigest()[:8].upper()

def create_or_update_incident(ticket_id: str, cluster_info: dict,
                               category: str, priority: str) -> str | None:
    if not cluster_info.get("incident_detected"):
        return None

    related_ids = [c["ticket_id"] for c in cluster_info["cluster"]] + [ticket_id]
    inc_id      = _make_incident_id(related_ids)

    existing = db.get_all_incidents()
    for inc in existing:
        existing_ids = json.loads(inc.get("ticket_ids", "[]"))
        if set(existing_ids) & set(related_ids):
            merged = list(set(existing_ids) | set(related_ids))
            db.upsert_incident({
                "id":           inc["id"],
                "ticket_ids":   json.dumps(merged),
                "blast_radius": blast_radius(merged),
                "priority":     priority,
            })
            return inc["id"]

    db.upsert_incident({
        "id":           inc_id,
        "name":         f"Suspected {category} Incident",
        "summary":      (f"Multiple customers reporting similar {category} issues. "
                         "Suspected incident – requires human verification."),
        "ticket_ids":   json.dumps(related_ids),
        "priority":     priority,
        "team":         ROUTING.get(category, "General Support"),
        "blast_radius": blast_radius(related_ids),
        "status":       "Pending",
        "created_at":   datetime.now().isoformat(),
    })
    return inc_id


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_orchestrator(ticket_id: str, message: str, customer_name: str,
                      channel: str, cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> dict:
    """
    Runs tools A-F, merges results, saves to DB, returns full decision.
    cluster_threshold is forwarded from the sidebar slider.
    """
    trace      = []
    created_at = datetime.now().isoformat()

    def record(tool: str, output: Any, status: str = "ok"):
        trace.append({"tool": tool, "output": output, "status": status})

    # ── Input sanitisation ────────────────────────────────────────────────────
    message, san_err = llm_module.sanitise_message(message)
    if san_err and not message:
        # Completely empty – create minimal ticket and return
        ticket_data = {
            "id": ticket_id, "created_at": created_at,
            "customer_name": customer_name, "channel": channel,
            "message": "(empty)", "category": "Other / Unclear",
            "summary": "Empty or whitespace message.", "intent": "Unknown",
            "priority": "Low", "priority_reason": san_err, "team": "General Support",
            "sentiment": "Neutral", "confidence": 0.0, "needs_review": 1,
            "needs_review_reason": san_err,
            "troubleshooting": json.dumps(["Request clarification from the customer."]),
            "reply_draft": "Hi there,\n\nWe received your message but it appeared blank. Could you please resend your question or issue? We're happy to help!",
            "status": "Open", "ai_mode": "demo", "similarity_method": "TF-IDF (local)",
            "ai_decision": "{}", "execution_trace": "[]",
        }
        db.insert_ticket(ticket_data)
        return {**ticket_data, "ai_result": {}, "ai_mode": "demo",
                "similarity": {}, "cluster_info": {}, "incident_id": None,
                "trace": [], "needs_review": True, "confidence": 0.0}

    # ── Step 0: LLM call ──────────────────────────────────────────────────────
    ai_result, ai_mode = llm_module.call_gemini(message)
    record("LLM", {"mode": ai_mode, "category": ai_result.get("category"),
                   "urgency": ai_result.get("urgency"),
                   "confidence": ai_result.get("confidence"),
                   "detected_language": ai_result.get("detected_language", "en")})

    # ── Tool A: Understanding ─────────────────────────────────────────────────
    understanding = tool_a_understanding(message, customer_name, ai_result)
    record("A_Understanding", understanding)
    category = understanding["category"]

    # ── Tool B: Impact (includes first-reporter rule) ─────────────────────────
    impact = tool_b_impact(ai_result, category, message)
    record("B_Impact", impact)
    priority = impact["priority"]

    # ── Tool C: Similarity ────────────────────────────────────────────────────
    similarity, sim_method = tool_c_similarity(ticket_id, message)
    record("C_Similarity", {"method": sim_method, "top_score": similarity.get("top_score", 0)})

    # ── Tool D: Clustering ────────────────────────────────────────────────────
    cluster_info = tool_d_clustering(ticket_id, message, category, created_at, cluster_threshold)
    record("D_Clustering", {
        "incident_detected": cluster_info["incident_detected"],
        "cluster_size":      cluster_info.get("cluster_size", 1),
        "method":            cluster_info.get("cluster_method", "TF-IDF (local)"),
    })

    # Escalate priority when cluster ≥ 3 → Critical
    if cluster_info.get("incident_detected"):
        old_priority = priority
        if cluster_info.get("cluster_size", 1) >= 3:
            priority = "Critical"
        else:
            escalation_map = {"Low": "Medium", "Medium": "High", "High": "Critical"}
            priority = escalation_map.get(priority, priority)
        impact["priority"]        = priority
        impact["priority_reason"] = cluster_info.get("escalate_reason", impact["priority_reason"])
        record("D_Clustering_Escalation", {"old": old_priority, "new": priority}, "escalated")

    # ── Tool E: Routing ────────────────────────────────────────────────────────
    routing = tool_e_routing(category, priority)
    record("E_Routing", routing)
    team = routing["team"]

    # ── Tool F: Resolution ────────────────────────────────────────────────────
    resolution = tool_f_resolution(ai_result, customer_name)
    record("F_Resolution", {"steps": len(resolution["troubleshooting_steps"])})

    # ── Confidence & review flag ───────────────────────────────────────────────
    confidence   = float(ai_result.get("confidence", 0.72))
    is_unclear   = (category == "Other / Unclear")
    is_short     = len(message.strip()) < 15
    needs_review = confidence < 0.6 or cluster_info.get("incident_detected", False) or is_unclear or is_short

    review_reasons = []
    if confidence < 0.6:
        review_reasons.append(f"Low confidence ({confidence:.0%})")
    if is_unclear:
        review_reasons.append("Unclear category")
    if is_short:
        review_reasons.append("Very short message")
    if cluster_info.get("incident_detected"):
        review_reasons.append("Part of suspected incident cluster")
    needs_review_reason = "; ".join(review_reasons) if review_reasons else ""

    # ── Persist ticket ────────────────────────────────────────────────────────
    ticket_data = {
        "id":                 ticket_id,
        "created_at":         created_at,
        "customer_name":      customer_name,
        "channel":            channel,
        "message":            message,
        "category":           category,
        "summary":            understanding["summary"],
        "intent":             understanding["intent"],
        "priority":           priority,
        "priority_reason":    impact["priority_reason"],
        "team":               team,
        "sentiment":          ai_result.get("sentiment", "Neutral"),
        "confidence":         confidence,
        "needs_review":       int(needs_review),
        "needs_review_reason": needs_review_reason,
        "troubleshooting":    json.dumps(resolution["troubleshooting_steps"]),
        "reply_draft":        resolution["reply_draft"],
        "status":             "Open",
        "ai_mode":            ai_mode,
        "similarity_method":  sim_method,
        "ai_decision":        json.dumps(ai_result),
        "execution_trace":    json.dumps(trace),
    }
    db.insert_ticket(ticket_data)
    db.log_audit("ticket_created", "System", ticket_id=ticket_id,
                 new_val=f"Priority={priority}, Team={team}, Mode={ai_mode}")

    # ── Incident detection ─────────────────────────────────────────────────────
    incident_id = create_or_update_incident(ticket_id, cluster_info, category, priority)
    if incident_id:
        db.log_audit("incident_detected", "System",
                     ticket_id=ticket_id, incident_id=incident_id,
                     new_val=f"Cluster size={cluster_info.get('cluster_size', 1)}")

    return {
        **ticket_data,
        "ai_result":      ai_result,
        "ai_mode":        ai_mode,
        "sim_method":     sim_method,
        "similarity":     similarity,
        "cluster_info":   cluster_info,
        "incident_id":    incident_id,
        "trace":          trace,
        "needs_review":   needs_review,
        "needs_review_reason": needs_review_reason,
        "confidence":     confidence,
    }
