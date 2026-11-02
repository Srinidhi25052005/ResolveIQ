"""
test_resolveiq.py – Quick pytest checks for ResolveIQ.
Run: pytest test_resolveiq.py -v
"""
import json, os, sys, pytest
sys.path.insert(0, os.path.dirname(__file__))

# ── Test 1: Tickets 1-3 form one suspected payment incident ──────────────────
def test_payment_cluster_forms_incident(tmp_path, monkeypatch):
    """
    Demo tickets 1-3 (payment gateway, card error, payment page not responding)
    MUST form a single suspected incident cluster even in demo mode.
    """
    test_db = str(tmp_path / "test.db")
    # Patch db path in db module AND in orchestrator's imported db
    import importlib
    import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", test_db)
    importlib.reload(db_mod)
    db_mod.DB_PATH = test_db
    db_mod.init_db()

    import orchestrator as orch
    importlib.reload(orch)

    messages = [
        ("Alice Thompson", "Our payment gateway is failing. Customers cannot complete transactions."),
        ("Bob Martinez",   "My card transaction is showing an error during checkout."),
        ("Carol Singh",    "The payment page is not responding."),
    ]

    incident_ids = set()
    ticket_ids   = []
    for name, msg in messages:
        tid = db_mod.generate_ticket_id()
        result = orch.run_orchestrator(
            ticket_id=tid, message=msg,
            customer_name=name, channel="Email"
        )
        ticket_ids.append(tid)
        if result.get("incident_id"):
            incident_ids.add(result["incident_id"])

    # All 3 tickets should share exactly ONE incident
    assert len(incident_ids) == 1, (
        f"Expected 1 incident, got {len(incident_ids)}: {incident_ids}"
    )

    # That incident should reference all 3 tickets
    inc = db_mod.get_incident(list(incident_ids)[0])
    assert inc is not None, "Incident record not found in DB"
    inc_tickets = json.loads(inc["ticket_ids"])
    for tid in ticket_ids:
        assert tid in inc_tickets, f"Ticket {tid} not in incident ticket list"

    print(f"\n✅ Cluster incident: {inc['id']} — tickets: {inc_tickets}")


# ── Test 2: LLM JSON response validation ──────────────────────────────────────
def test_json_validation():
    """
    The _validate function must:
    - accept a well-formed result
    - reject missing fields
    - reject invalid enum values
    """
    from llm import _validate, REQUIRED_FIELDS

    good = {
        "category":            "Payments",
        "summary":             "Payment gateway failing",
        "intent":              "Resolve payment outage",
        "urgency":             "Critical",
        "sentiment":           "Frustrated",
        "confidence":          0.92,
        "troubleshooting_steps": ["Check gateway", "Escalate to SRE"],
        "reply_draft":         "We are investigating the issue.",
    }
    ok, err = _validate(good)
    assert ok, f"Good payload failed validation: {err}"

    # Missing field
    bad_missing = dict(good)
    del bad_missing["category"]
    ok2, err2 = _validate(bad_missing)
    assert not ok2 and "Missing" in err2, "Should fail for missing category"

    # Bad enum
    bad_enum = dict(good)
    bad_enum["urgency"] = "SUPER_URGENT"
    ok3, err3 = _validate(bad_enum)
    assert not ok3 and "urgency" in err3.lower(), "Should fail for bad urgency enum"

    # Confidence out of range
    bad_conf = dict(good)
    bad_conf["confidence"] = 1.5
    ok4, err4 = _validate(bad_conf)
    assert not ok4 and "confidence" in err4.lower(), "Should fail for out-of-range confidence"

    print("\n✅ JSON validation tests passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
