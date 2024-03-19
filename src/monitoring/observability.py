from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time

PIPELINE_RECORDS = Counter("ehr_records_processed_total", "EHR records processed", ["pipeline", "status"])
NLP_LATENCY = Histogram("ner_inference_latency_seconds", "NER inference latency", ["model"])
ICU_RISK_GAUGE = Gauge("icu_high_risk_patients_current", "Current high-risk ICU patients")
DATA_QUALITY = Gauge("ehr_data_quality_score", "EHR data quality score", ["dataset"])
ETL_FAILURES = Counter("ehr_etl_failures_total", "ETL pipeline failures", ["stage"])


def track_ehr_processing(pipeline_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                count = len(result) if hasattr(result, "__len__") else 1
                PIPELINE_RECORDS.labels(pipeline=pipeline_name, status="success").inc(count)
                return result
            except Exception as e:
                ETL_FAILURES.labels(stage=pipeline_name).inc()
                raise
        return wrapper
    return decorator
