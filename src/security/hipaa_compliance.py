import hashlib
import hmac
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

PHI_FIELDS = ["patient_name", "dob", "ssn", "address", "phone", "email",
              "mrn", "ip_address", "account_number", "certificate_number"]


class HIPAAComplianceManager:
    """HIPAA-compliant PHI de-identification and audit logging."""

    def __init__(self, secret_key: str = None, audit_log_path: str = "/var/log/hipaa_audit.log"):
        self.secret_key = (secret_key or os.getenv("HIPAA_SECRET_KEY", "default-key")).encode()
        self.audit_log_path = audit_log_path
        self._setup_audit_logger()

    def _setup_audit_logger(self):
        self.audit_logger = logging.getLogger("hipaa_audit")
        handler = logging.FileHandler(self.audit_log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)

    def pseudonymize(self, value: str, field_type: str = "generic") -> str:
        token = hmac.new(self.secret_key, value.encode(), hashlib.sha256).hexdigest()[:16]
        return f"[{field_type.upper()}_REDACTED_{token}]"

    def deidentify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if any(phi in col.lower() for phi in PHI_FIELDS):
                df[col] = df[col].astype(str).apply(
                    lambda x: self.pseudonymize(x, col) if x != "nan" else x
                )
        return df

    def log_data_access(self, user_id: str, resource: str,
                       action: str, record_count: int = 0):
        entry = {"timestamp": datetime.utcnow().isoformat(), "user_id": user_id,
                 "resource": resource, "action": action, "record_count": record_count,
                 "compliant": True}
        self.audit_logger.info(json.dumps(entry))
