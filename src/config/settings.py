import os
from dotenv import load_dotenv
load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
HIPAA_AUDIT_LOG = os.getenv("HIPAA_AUDIT_LOG_PATH", "/var/log/hipaa_audit.log")
