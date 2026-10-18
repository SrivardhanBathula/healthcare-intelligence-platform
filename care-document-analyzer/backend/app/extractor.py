import re
from datetime import date
from .schemas import Extraction

def extract_document(text: str) -> Extraction:
    """Conservative demo extractor; use fabricated documents only."""
    low = text.lower()
    event = "unknown"
    for word, kind in [("fall", "fall"), ("medication", "medication"), ("dose", "medication"),
                       ("behavior", "behavior"), ("injury", "injury"), ("bruise", "injury")]:
        if word in low:
            event = kind
            break
    rid = re.search(r"\b(RES-[0-9]{3,6})\b", text, re.I)
    dt = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    meds = [m for m in ["acetaminophen", "ibuprofen", "metformin", "lisinopril", "aspirin"] if m in low]
    flags = []
    if "unwitnessed" in low:
        flags.append("unwitnessed_event")
    if "head impact" in low or "hit head" in low:
        flags.append("possible_head_impact")
    if "missed dose" in low:
        flags.append("missed_medication_dose")
    if not rid:
        flags.append("resident_id_missing")
    parsed = None
    if dt:
        try:
            parsed = date.fromisoformat(dt.group(1))
        except ValueError:
            flags.append("date_invalid")
    else:
        flags.append("date_missing")
    summary = re.sub(r"\s+", " ", text).strip()[:500]
    return Extraction(
        resident_id=rid.group(1).upper() if rid else None,
        event_type=event,
        event_date=parsed,
        medications_mentioned=meds,
        risk_flags=flags,
        summary=summary,
        needs_human_review=True,
    )
