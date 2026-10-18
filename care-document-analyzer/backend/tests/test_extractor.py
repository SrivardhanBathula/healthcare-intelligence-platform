from app.extractor import extract_document

def test_fake_fall_extracts_identifier_and_date():
    result = extract_document("2026-10-12 RES-1042 experienced an unwitnessed fall.")
    assert result.resident_id == "RES-1042"
    assert result.event_type == "fall"
    assert str(result.event_date) == "2026-10-12"
    assert "unwitnessed_event" in result.risk_flags

def test_missing_identifier_and_date_are_flagged():
    result = extract_document("General shift note, no identifiers included.")
    assert result.resident_id is None
    assert "resident_id_missing" in result.risk_flags
    assert "date_missing" in result.risk_flags
