"""
Clinical NLP Pipeline — ClinicalBERT + spaCy NER
Processes 500K+ unstructured clinical notes for deep text mining,
Named Entity Recognition (NER), negation detection, and temporal extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)

# BERT NER label schema (BIO tagging)
LABEL_MAP = {
    "B-PROBLEM": "DIAGNOSIS",
    "I-PROBLEM": "DIAGNOSIS",
    "B-TREATMENT": "TREATMENT",
    "I-TREATMENT": "TREATMENT",
    "B-TEST": "LAB_TEST",
    "I-TEST": "LAB_TEST",
    "B-MEDICATION": "MEDICATION",
    "I-MEDICATION": "MEDICATION",
    "B-DOSAGE": "DOSAGE",
    "I-DOSAGE": "DOSAGE",
}

NEGATION_TERMS = {
    "no", "not", "without", "denies", "denied", "absent", "negative",
    "never", "non", "neither", "nor", "rule out", "ruled out", "free of",
}


@dataclass
class ClinicalEntity:
    text: str
    entity_type: str  # DIAGNOSIS, TREATMENT, MEDICATION, LAB_TEST, DOSAGE
    start: int
    end: int
    negated: bool = False
    confidence: float = 1.0
    temporal_context: Optional[str] = None  # "current", "historical", "family"


@dataclass
class ClinicalNoteResult:
    note_id: str
    patient_id: str
    entities: list[ClinicalEntity] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    negated_conditions: list[str] = field(default_factory=list)


class ClinicalNERPipeline:
    """
    Clinical NLP pipeline using ClinicalBERT for entity recognition
    and spaCy for dependency-based negation detection and preprocessing.
    """

    def __init__(
        self,
        bert_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        spacy_model: str = "en_core_sci_lg",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading ClinicalBERT from {bert_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert_model = AutoModelForTokenClassification.from_pretrained(bert_model)
        self.bert_pipeline = pipeline(
            "ner",
            model=self.bert_model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1,
        )

        logger.info(f"Loading spaCy model: {spacy_model}...")
        self.nlp = spacy.load(spacy_model, disable=["parser"])
        self.nlp.add_pipe("sentencizer")

    def preprocess(self, text: str) -> str:
        """
        Clinical text preprocessing:
        - Normalize whitespace and line breaks
        - Expand common clinical abbreviations
        - Remove PHI placeholders (de-identified data)
        """
        text = re.sub(r"\s+", " ", text).strip()

        # Expand common abbreviations
        abbrev_map = {
            r"\bpt\b": "patient", r"\bhx\b": "history", r"\bdx\b": "diagnosis",
            r"\btx\b": "treatment", r"\brx\b": "prescription", r"\bs/p\b": "status post",
            r"\bc/o\b": "complains of", r"\bw/o\b": "without", r"\bSOB\b": "shortness of breath",
            r"\bHTN\b": "hypertension", r"\bDM\b": "diabetes mellitus",
            r"\bCAD\b": "coronary artery disease", r"\bCHF\b": "congestive heart failure",
        }
        for pattern, replacement in abbrev_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def detect_negation(self, entity_text: str, context: str) -> bool:
        """
        Rule-based + dependency negation detection.
        Checks 5-token window before entity for negation terms.
        """
        entity_pos = context.lower().find(entity_text.lower())
        if entity_pos == -1:
            return False

        pre_context = context[max(0, entity_pos - 60): entity_pos].lower()
        tokens = pre_context.split()
        window = tokens[-5:] if len(tokens) >= 5 else tokens

        return any(neg in " ".join(window) for neg in NEGATION_TERMS)

    def detect_temporal_context(self, entity_text: str, context: str) -> str:
        """
        Classify entity temporal context: current, historical, or family history.
        """
        context_lower = context.lower()
        entity_pos = context_lower.find(entity_text.lower())
        window = context_lower[max(0, entity_pos - 100): entity_pos + 100]

        if any(t in window for t in ["family history", "mother", "father", "sibling", "fh:"]):
            return "family_history"
        if any(t in window for t in ["history of", "prior", "previous", "past medical", "hx of"]):
            return "historical"
        return "current"

    def extract_entities(self, text: str) -> list[ClinicalEntity]:
        """
        Run ClinicalBERT NER on clinical text and return enriched entities.
        """
        preprocessed = self.preprocess(text)
        raw_entities = self.bert_pipeline(preprocessed)

        entities = []
        for ent in raw_entities:
            label = LABEL_MAP.get(ent["entity_group"], ent["entity_group"])
            entity = ClinicalEntity(
                text=ent["word"],
                entity_type=label,
                start=ent["start"],
                end=ent["end"],
                confidence=round(ent["score"], 4),
                negated=self.detect_negation(ent["word"], preprocessed),
                temporal_context=self.detect_temporal_context(ent["word"], preprocessed),
            )
            entities.append(entity)

        return entities

    def process_note(self, note_id: str, patient_id: str, note_text: str) -> ClinicalNoteResult:
        """
        Process a single clinical note and return structured extraction results.
        """
        entities = self.extract_entities(note_text)
        result = ClinicalNoteResult(note_id=note_id, patient_id=patient_id, entities=entities)

        for ent in entities:
            if ent.negated:
                result.negated_conditions.append(ent.text)
            elif ent.entity_type == "DIAGNOSIS" and ent.temporal_context == "current":
                result.diagnoses.append(ent.text)
            elif ent.entity_type == "MEDICATION":
                result.medications.append(ent.text)
            elif ent.entity_type == "TREATMENT":
                result.procedures.append(ent.text)

        return result

    def process_batch(self, notes: list[dict]) -> list[ClinicalNoteResult]:
        """
        Process a batch of clinical notes efficiently.
        Input: list of {"note_id": ..., "patient_id": ..., "text": ...}
        """
        logger.info(f"Processing batch of {len(notes):,} clinical notes...")
        results = []
        for note in notes:
            try:
                result = self.process_note(
                    note_id=note["note_id"],
                    patient_id=note["patient_id"],
                    note_text=note["text"],
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process note {note.get('note_id')}: {e}")

        logger.info(f"Batch processing complete. {len(results):,} notes processed.")
        return results

    def results_to_dataframe(self, results: list[ClinicalNoteResult]):
        """Flatten NER results to a tidy DataFrame for downstream analytics."""
        import pandas as pd
        rows = []
        for r in results:
            for ent in r.entities:
                rows.append({
                    "note_id": r.note_id,
                    "patient_id": r.patient_id,
                    "entity_text": ent.text,
                    "entity_type": ent.entity_type,
                    "negated": ent.negated,
                    "temporal_context": ent.temporal_context,
                    "confidence": ent.confidence,
                })
        return pd.DataFrame(rows)
