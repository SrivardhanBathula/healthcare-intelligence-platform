# 🏥 AI-Driven Healthcare Intelligence & Clinical Analytics Platform

> **Enterprise-grade clinical AI system for EHR analytics, ICU risk forecasting, and clinical NLP — processing 500K+ unstructured notes with 25% improvement in clinical risk forecasting accuracy.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![HIPAA](https://img.shields.io/badge/Compliance-HIPAA-green)](https://www.hhs.gov/hipaa)
[![OMOP](https://img.shields.io/badge/Standard-OMOP_CDM-blue)](https://www.ohdsi.org/data-standardization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This platform was architected and deployed at **HCL Technologies** for a major healthcare enterprise to unify clinical data, predict patient risk, and extract insights from unstructured clinical text using state-of-the-art NLP and deep learning.

### 🏆 Key Results
| Metric | Improvement |
|---|---|
| ICU Risk Forecasting Accuracy | **+25%** |
| Operational Reporting Overhead | **-40%** |
| Data Architecture Bottleneck | **Eliminated** |
| Clinical Notes Processed | **500K+** |
| EHR Records Standardized | **OMOP CDM compliant** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                       │
│        EHR Systems │ Clinical Claims │ Lab Results           │
│        HL7 FHIR │ CSV/EDI │ DICOM (Imaging)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              STANDARDIZATION & ETL LAYER                     │
│    OMOP CDM │ ICD-10 Mapping │ SNOMED CT │ RxNorm           │
│    Apache NiFi │ PySpark │ Apache Airflow                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼───────┐ ┌────────▼───────┐ ┌───────▼────────┐
│  Predictive    │ │  Clinical NLP  │ │  Medical       │
│  Risk Models   │ │  Pipeline      │ │  Imaging AI    │
│  XGBoost +     │ │  BERT/spaCy    │ │  CNN/TF        │
│  Survival      │ │  NER/Text      │ │  Batch         │
│  Analysis      │ │  Mining        │ │  Inference     │
└────────┬───────┘ └────────┬───────┘ └───────┬────────┘
         └──────────────────┼──────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│               SERVING & OBSERVABILITY LAYER                  │
│       FastAPI │ Prometheus │ Grafana │ CloudWatch            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Components

### 1. Healthcare Data Standardization (`src/data_pipeline/`)
- OMOP CDM transformation for heterogeneous EHR datasets
- ICD-10, SNOMED CT, RxNorm terminology mapping
- Apache NiFi orchestration for real-time data ingestion
- Eliminates data architecture bottlenecks across distributed inference pipelines

### 2. Predictive Risk Models (`src/models/`)
- **XGBoost** + survival analysis for real-time ICU mortality and readmission forecasting
- **Scikit-Learn** ensemble for sepsis prediction and deterioration alerting
- Workload profiling-optimized training achieving +25% forecasting accuracy
- SHAP-based clinical explainability for clinician trust and regulatory compliance

### 3. Clinical NLP Pipeline (`src/nlp/`)
- **BERT / ClinicalBERT** fine-tuned for medical entity recognition
- **spaCy** custom pipelines for clinical Named Entity Recognition (NER)
- Processes 500K+ unstructured clinical notes for diagnostic insights
- Negation detection, temporal relation extraction, medication dosage parsing

### 4. Medical Imaging AI (`src/models/imaging_model.py`)
- CNN architectures (TensorFlow) for radiology image classification
- AI Accelerator Optimization for GPU batch inference throughput
- Latency minimization through System-Level Performance Tuning

### 5. Production Observability (`src/monitoring/`)
- Prometheus + Grafana telemetry dashboards replacing manual tracking
- 40% reduction in operational reporting overhead
- CloudWatch alerts and automated drift detection

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **ML Frameworks** | PyTorch, TensorFlow, XGBoost, Scikit-Learn, Lifelines |
| **Clinical NLP** | ClinicalBERT, Transformers, spaCy, scispaCy, MedSpaCy |
| **Data Standards** | OMOP CDM, ICD-10, SNOMED CT, RxNorm, HL7 FHIR |
| **Data Engineering** | PySpark, Apache NiFi, Apache Airflow, SQL, Databricks |
| **Serving** | FastAPI, Docker, Kubernetes, RESTful Microservices |
| **Observability** | Prometheus, Grafana, AWS CloudWatch, Evidently AI |
| **Compliance** | HIPAA, GDPR, CCPA, De-identification (Safe Harbor) |

---

## 📁 Project Structure

```
healthcare-intelligence-platform/
├── src/
│   ├── data_pipeline/
│   │   ├── omop_transformer.py         # OMOP CDM standardization
│   │   ├── ehr_ingestion.py            # Multi-source EHR ingestion
│   │   └── terminology_mapper.py       # ICD-10 / SNOMED / RxNorm
│   ├── models/
│   │   ├── icu_risk_model.py           # ICU forecasting (XGBoost + survival)
│   │   ├── sepsis_predictor.py         # Sepsis early warning model
│   │   └── imaging_model.py            # CNN medical imaging classifier
│   ├── nlp/
│   │   ├── clinical_ner.py             # ClinicalBERT NER pipeline
│   │   ├── note_processor.py           # Clinical note preprocessing
│   │   └── entity_extractor.py         # Medication, diagnosis, procedure NER
│   ├── api/
│   │   ├── main.py                     # FastAPI application
│   │   └── schemas.py                  # HIPAA-compliant request/response models
│   └── monitoring/
│       ├── telemetry.py                # Prometheus metrics exporter
│       └── drift_monitor.py            # Model drift detection
├── tests/
│   ├── test_icu_model.py
│   ├── test_nlp_pipeline.py
│   └── test_omop_transformer.py
├── configs/
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── notebooks/
│   ├── 01_EDA_EHR_Data.ipynb
│   ├── 02_ICU_Risk_Model_Training.ipynb
│   └── 03_Clinical_NLP_Demo.ipynb
├── docs/
│   └── architecture.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/SrivardhanBathula/healthcare-intelligence-platform.git
cd healthcare-intelligence-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download clinical NLP models
python -m spacy download en_core_sci_lg
python -c "from transformers import AutoModel; AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"

# Set environment variables
cp configs/.env.example configs/.env

# Run API
uvicorn src.api.main:app --reload --port 8001

# Run with Docker
docker-compose up --build
```

---

## 📊 Model Performance

| Model | Task | AUC-ROC | Accuracy | Notes |
|---|---|---|---|---|
| XGBoost + Cox | ICU Mortality (24h) | 0.93 | 89% | +25% vs baseline |
| XGBoost Ensemble | Sepsis Prediction | 0.91 | 87% | 6-hour early warning |
| ClinicalBERT NER | Diagnosis Extraction | — | 91% F1 | 500K+ notes |
| CNN (ResNet-50) | Chest X-Ray Classification | 0.94 | 90% | GPU-optimized |

---

## 🔒 HIPAA Compliance & Data Privacy
- Safe Harbor de-identification applied to all patient data
- PHI (Protected Health Information) never logged or exposed in API responses
- Audit logging for all data access per HIPAA requirements
- Role-based access control with MFA enforcement
- Data encrypted at rest (AES-256) and in transit (TLS 1.3)

---

## 👤 Author

**Srivardhan Bathula** — AI/ML Engineer  
📧 Srivardhan.Bathula1@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/srivardhan-bathula) | [GitHub](https://github.com/SrivardhanBathula)
