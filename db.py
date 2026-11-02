"""
db.py – SQLite persistence for ResolveIQ.
Tables: tickets, incidents, audit_log, embedding_cache, corrections
"""
import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "resolveiq.db")

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    customer_name   TEXT,
    channel         TEXT,
    message         TEXT,
    category        TEXT,
    summary         TEXT,
    intent          TEXT,
    priority        TEXT,
    priority_reason TEXT,
    team            TEXT,
    sentiment       TEXT,
    confidence      REAL,
    needs_review    INTEGER DEFAULT 0,
    needs_review_reason TEXT,
    troubleshooting TEXT,
    reply_draft     TEXT,
    status          TEXT DEFAULT 'Open',
    ai_mode         TEXT DEFAULT 'demo',
    similarity_method TEXT DEFAULT 'TF-IDF (local)',
    ai_decision     TEXT,
    execution_trace TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    summary         TEXT,
    ticket_ids      TEXT,
    priority        TEXT,
    team            TEXT,
    blast_radius    REAL,
    status          TEXT DEFAULT 'Pending',
    created_at      TEXT,
    approved_by     TEXT,
    approved_at     TEXT,
    engineering_alert TEXT,
    customer_update   TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    ticket_id   TEXT,
    incident_id TEXT,
    action      TEXT,
    actor       TEXT,
    old_val     TEXT,
    new_val     TEXT
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    ticket_id   TEXT PRIMARY KEY,
    embedding   TEXT NOT NULL,
    model       TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,
    ticket_id       TEXT,
    field           TEXT,
    ai_value        TEXT,
    agent_value     TEXT,
    actor           TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # ── migrations: add columns that didn't exist in v1 ──────────────────
        _add_column_if_missing(conn, "tickets", "needs_review_reason", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "tickets", "similarity_method",   "TEXT DEFAULT 'TF-IDF (local)'")

def _add_column_if_missing(conn, table: str, column: str, col_def: str):
    """ALTER TABLE only when the column doesn't already exist (SQLite safe)."""
    existing = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")

# ── Tickets ───────────────────────────────────────────────────────────────────
def generate_ticket_id():
    """RIQ-YYYYMMDD-NNNN format."""
    today = datetime.now().strftime("%Y%m%d")
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM tickets WHERE id LIKE ?", (f"RIQ-{today}-%",)
        ).fetchall()
    seq = len(rows) + 1
    return f"RIQ-{today}-{seq:04d}"

def insert_ticket(data: dict):
    sql = """INSERT INTO tickets
        (id, created_at, customer_name, channel, message, category, summary,
         intent, priority, priority_reason, team, sentiment, confidence,
         needs_review, needs_review_reason, troubleshooting, reply_draft,
         status, ai_mode, similarity_method, ai_decision, execution_trace)
        VALUES
        (:id, :created_at, :customer_name, :channel, :message, :category,
         :summary, :intent, :priority, :priority_reason, :team, :sentiment,
         :confidence, :needs_review, :needs_review_reason, :troubleshooting,
         :reply_draft, :status, :ai_mode, :similarity_method,
         :ai_decision, :execution_trace)"""
    with get_conn() as conn:
        conn.execute(sql, data)

def update_ticket(ticket_id: str, **kwargs):
    if not kwargs:
        return
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [ticket_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE tickets SET {sets} WHERE id=?", vals)

def get_ticket(ticket_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id=?", (ticket_id,)).fetchone()
    return dict(row) if row else None

def get_all_tickets():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

# ── Incidents ─────────────────────────────────────────────────────────────────
def upsert_incident(data: dict):
    existing = get_incident(data["id"])
    if existing:
        sets = ", ".join(f"{k}=?" for k in data if k != "id")
        vals = [data[k] for k in data if k != "id"] + [data["id"]]
        with get_conn() as conn:
            conn.execute(f"UPDATE incidents SET {sets} WHERE id=?", vals)
    else:
        cols = ", ".join(data.keys())
        phs  = ", ".join("?" * len(data))
        with get_conn() as conn:
            conn.execute(f"INSERT INTO incidents ({cols}) VALUES ({phs})", list(data.values()))

def get_incident(incident_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
    return dict(row) if row else None

def get_all_incidents():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM incidents ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_pending_incidents():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE status='Pending' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

# ── Audit log ─────────────────────────────────────────────────────────────────
def log_audit(action: str, actor: str = "System",
              ticket_id: str = "", incident_id: str = "",
              old_val: str = "", new_val: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, ticket_id, incident_id, action, actor, old_val, new_val) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), ticket_id, incident_id, action, actor, old_val, new_val)
        )

def get_audit_log(ticket_id: str = "", incident_id: str = ""):
    with get_conn() as conn:
        if ticket_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE ticket_id=? ORDER BY ts DESC", (ticket_id,)
            ).fetchall()
        elif incident_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE incident_id=? ORDER BY ts DESC", (incident_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 200").fetchall()
    return [dict(r) for r in rows]

def reset_db():
    """Drop and recreate all tables."""
    with get_conn() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS tickets;
            DROP TABLE IF EXISTS incidents;
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS embedding_cache;
            DROP TABLE IF EXISTS corrections;
        """)
    init_db()

# ── Embedding cache ───────────────────────────────────────────────────────────
def save_embedding(ticket_id: str, embedding: list, model: str = "text-embedding-004"):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (ticket_id, embedding, model, created_at) "
            "VALUES (?,?,?,?)",
            (ticket_id, json.dumps(embedding), model, datetime.now().isoformat())
        )

def get_embedding(ticket_id: str) -> list | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE ticket_id=?", (ticket_id,)
        ).fetchone()
    return json.loads(row["embedding"]) if row else None

def get_all_embeddings() -> list[dict]:
    """Returns [{"ticket_id": ..., "embedding": [...]}]"""
    with get_conn() as conn:
        rows = conn.execute("SELECT ticket_id, embedding FROM embedding_cache").fetchall()
    return [{"ticket_id": r["ticket_id"], "embedding": json.loads(r["embedding"])} for r in rows]

# ── Corrections / feedback loop ───────────────────────────────────────────────
def log_correction(ticket_id: str, field: str, ai_value: str,
                   agent_value: str, actor: str = "Agent"):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO corrections (ts, ticket_id, field, ai_value, agent_value, actor) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(), ticket_id, field, ai_value, agent_value, actor)
        )

def get_accuracy_metrics() -> dict:
    """
    Returns accept rate = tickets with no corrections / total tickets.
    Also cluster precision = approved / (approved+rejected) incidents.
    """
    with get_conn() as conn:
        total_tickets = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        corrected = conn.execute(
            "SELECT COUNT(DISTINCT ticket_id) FROM corrections"
        ).fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE status='Approved'"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE status='Rejected'"
        ).fetchone()[0]

    accepted_without_edit = max(0, total_tickets - corrected)
    accept_rate = round(100 * accepted_without_edit / max(total_tickets, 1), 1)
    total_clusters = approved + rejected
    cluster_precision = round(100 * approved / max(total_clusters, 1), 1)
    return {
        "total_tickets":      total_tickets,
        "corrected":          corrected,
        "accept_rate":        accept_rate,
        "approved_clusters":  approved,
        "rejected_clusters":  rejected,
        "cluster_precision":  cluster_precision,
    }
