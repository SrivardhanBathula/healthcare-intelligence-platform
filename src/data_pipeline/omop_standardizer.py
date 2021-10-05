import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

ICD10_CONCEPT_MAP = {
    "I21": 312327, "I50": 316139, "J18": 255848,
    "E11": 201826, "N18": 46271022, "F32": 440383,
}

SNOMED_MAP = {
    "chest pain": 29857009, "dyspnea": 230145002,
    "hypertension": 38341003, "diabetes": 73211009,
}


class OMOPStandardizer:
    """Standardize heterogeneous EHR data to OMOP CDM v5.4."""

    def __init__(self):
        self.icd_map = ICD10_CONCEPT_MAP
        self.snomed_map = SNOMED_MAP

    def standardize_person(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "person_id": df["patient_id"],
            "gender_concept_id": df["gender"].map({"M": 8507, "F": 8532}).fillna(0).astype(int),
            "year_of_birth": pd.to_datetime(df["dob"]).dt.year,
            "race_concept_id": df.get("race", pd.Series([0]*len(df))).fillna(0).astype(int),
            "ethnicity_concept_id": df.get("ethnicity", pd.Series([0]*len(df))).fillna(0).astype(int),
        })

    def standardize_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["condition_concept_id"] = df["icd_code"].str[:3].map(self.icd_map).fillna(0).astype(int)
        return pd.DataFrame({
            "condition_occurrence_id": range(len(df)),
            "person_id": df["patient_id"],
            "condition_concept_id": df["condition_concept_id"],
            "condition_start_date": pd.to_datetime(df["diagnosis_date"]),
            "condition_type_concept_id": 32020,
            "visit_occurrence_id": df.get("visit_id", pd.Series([None]*len(df))),
        })

    def standardize_drug_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "drug_exposure_id": range(len(df)),
            "person_id": df["patient_id"],
            "drug_concept_id": df.get("rxnorm_code", pd.Series([0]*len(df))).fillna(0).astype(int),
            "drug_exposure_start_date": pd.to_datetime(df["prescription_date"]),
            "drug_type_concept_id": 32817,
            "quantity": df.get("quantity", pd.Series([None]*len(df))),
        })
