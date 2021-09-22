"""
OMOP CDM Transformer
Standardizes heterogeneous EHR and clinical claims data to OMOP Common Data Model.
Eliminates data architecture bottlenecks across distributed inference pipelines.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OMOPPerson:
    """OMOP CDM Person domain."""
    person_id: int
    gender_concept_id: int
    year_of_birth: int
    month_of_birth: Optional[int]
    day_of_birth: Optional[int]
    race_concept_id: int
    ethnicity_concept_id: int
    care_site_id: Optional[int]


@dataclass
class OMOPConditionOccurrence:
    """OMOP CDM Condition Occurrence domain."""
    condition_occurrence_id: int
    person_id: int
    condition_concept_id: int     # SNOMED CT mapped
    condition_start_date: str
    condition_end_date: Optional[str]
    condition_type_concept_id: int
    visit_occurrence_id: Optional[int]
    condition_source_value: str    # Original ICD-10 code
    condition_source_concept_id: int


class ICD10ToSNOMEDMapper:
    """
    Maps ICD-10 diagnosis codes to SNOMED CT concept IDs.
    Uses OMOP vocabulary tables for standardized mapping.
    """

    def __init__(self, vocabulary_path: str):
        self.vocab_df = pd.read_parquet(vocabulary_path)
        self._build_index()

    def _build_index(self):
        icd10_vocab = self.vocab_df[
            (self.vocab_df["vocabulary_id"] == "ICD10CM") &
            (self.vocab_df["invalid_reason"].isna())
        ]
        self.icd10_map = dict(zip(
            icd10_vocab["concept_code"],
            icd10_vocab["concept_id"]
        ))

        snomed_map = self.vocab_df[
            self.vocab_df["vocabulary_id"] == "SNOMED"
        ]
        self.snomed_map = dict(zip(
            snomed_map["concept_id"],
            snomed_map["concept_name"]
        ))

    def map_icd10_to_snomed(self, icd10_code: str) -> tuple[int, str]:
        """Return (snomed_concept_id, snomed_concept_name) for an ICD-10 code."""
        concept_id = self.icd10_map.get(icd10_code.replace(".", ""), 0)
        concept_name = self.snomed_map.get(concept_id, "Unknown")
        return concept_id, concept_name


class OMOPTransformer:
    """
    Transforms raw EHR data from various source schemas into OMOP CDM v5.4.
    Supports Epic, Cerner, and custom EDI claim formats.
    """

    GENDER_CONCEPT_MAP = {
        "M": 8507, "MALE": 8507,
        "F": 8532, "FEMALE": 8532,
        "U": 8551, "UNKNOWN": 8551,
    }

    RACE_CONCEPT_MAP = {
        "WHITE": 8527,
        "BLACK": 8516,
        "ASIAN": 8515,
        "HISPANIC": 38003563,
        "UNKNOWN": 8552,
    }

    def __init__(self, icd10_mapper: Optional[ICD10ToSNOMEDMapper] = None):
        self.icd10_mapper = icd10_mapper

    def transform_persons(self, raw_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw patient demographics to OMOP Person table.
        Input columns: patient_id, sex, dob, race, ethnicity, site_id
        """
        logger.info(f"Transforming {len(raw_patients):,} patient records to OMOP Person...")

        omop_persons = pd.DataFrame({
            "person_id": raw_patients["patient_id"].astype(int),
            "gender_concept_id": raw_patients["sex"].str.upper().map(self.GENDER_CONCEPT_MAP).fillna(8551),
            "year_of_birth": pd.to_datetime(raw_patients["dob"]).dt.year,
            "month_of_birth": pd.to_datetime(raw_patients["dob"]).dt.month,
            "day_of_birth": pd.to_datetime(raw_patients["dob"]).dt.day,
            "race_concept_id": raw_patients["race"].str.upper().map(self.RACE_CONCEPT_MAP).fillna(8552),
            "ethnicity_concept_id": raw_patients.get("ethnicity", pd.Series(["UNKNOWN"] * len(raw_patients)))
                                        .map({"HISPANIC": 38003563, "NON_HISPANIC": 38003564}).fillna(0),
            "care_site_id": raw_patients.get("site_id", None),
            "person_source_value": raw_patients["patient_id"].astype(str),
        })

        logger.info(f"Person transformation complete. {len(omop_persons):,} records.")
        return omop_persons

    def transform_conditions(self, raw_diagnoses: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw diagnosis records to OMOP Condition Occurrence table.
        Input columns: diag_id, patient_id, icd10_code, onset_date, resolution_date, visit_id
        """
        logger.info(f"Transforming {len(raw_diagnoses):,} diagnosis records...")

        if self.icd10_mapper:
            mapped = raw_diagnoses["icd10_code"].apply(self.icd10_mapper.map_icd10_to_snomed)
            raw_diagnoses["condition_concept_id"] = mapped.apply(lambda x: x[0])
        else:
            raw_diagnoses["condition_concept_id"] = 0

        omop_conditions = pd.DataFrame({
            "condition_occurrence_id": raw_diagnoses["diag_id"].astype(int),
            "person_id": raw_diagnoses["patient_id"].astype(int),
            "condition_concept_id": raw_diagnoses["condition_concept_id"],
            "condition_start_date": pd.to_datetime(raw_diagnoses["onset_date"]).dt.strftime("%Y-%m-%d"),
            "condition_end_date": pd.to_datetime(raw_diagnoses.get("resolution_date", None),
                                                  errors="coerce").dt.strftime("%Y-%m-%d"),
            "condition_type_concept_id": 32817,  # EHR Type Concept
            "visit_occurrence_id": raw_diagnoses.get("visit_id", None),
            "condition_source_value": raw_diagnoses["icd10_code"],
            "condition_source_concept_id": 0,
        })

        logger.info(f"Condition transformation complete. {len(omop_conditions):,} records.")
        return omop_conditions

    def validate_omop_schema(self, df: pd.DataFrame, domain: str) -> dict:
        """
        Validate transformed OMOP table against CDM v5.4 schema requirements.
        Returns dict of validation results and error counts.
        """
        required_fields = {
            "person": ["person_id", "gender_concept_id", "year_of_birth", "race_concept_id"],
            "condition_occurrence": ["condition_occurrence_id", "person_id",
                                     "condition_concept_id", "condition_start_date"],
        }

        results = {"domain": domain, "total_records": len(df), "errors": []}
        fields = required_fields.get(domain, [])

        for field in fields:
            if field not in df.columns:
                results["errors"].append(f"Missing required field: {field}")
            elif df[field].isna().sum() > 0:
                null_count = int(df[field].isna().sum())
                results["errors"].append(f"Null values in {field}: {null_count}")

        # Check for invalid concept IDs
        if "condition_concept_id" in df.columns:
            unmapped = int((df["condition_concept_id"] == 0).sum())
            if unmapped > 0:
                results["errors"].append(f"Unmapped concept IDs: {unmapped} ({unmapped/len(df):.1%})")

        results["is_valid"] = len(results["errors"]) == 0
        logger.info(f"OMOP validation [{domain}]: {'PASS' if results['is_valid'] else 'FAIL'} — "
                    f"{len(results['errors'])} issue(s)")
        return results

    def run_full_pipeline(
        self,
        raw_patients: pd.DataFrame,
        raw_diagnoses: pd.DataFrame,
        output_dir: str = "data/omop/",
    ) -> dict[str, pd.DataFrame]:
        """
        Run full OMOP transformation pipeline and write outputs.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        persons = self.transform_persons(raw_patients)
        conditions = self.transform_conditions(raw_diagnoses)

        # Validate
        person_validation = self.validate_omop_schema(persons, "person")
        condition_validation = self.validate_omop_schema(conditions, "condition_occurrence")

        # Write outputs
        persons.to_parquet(f"{output_dir}/person.parquet", index=False)
        conditions.to_parquet(f"{output_dir}/condition_occurrence.parquet", index=False)

        logger.info(f"OMOP pipeline complete. Outputs written to {output_dir}")
        return {
            "person": persons,
            "condition_occurrence": conditions,
            "validation": {
                "person": person_validation,
                "condition_occurrence": condition_validation,
            },
        }
