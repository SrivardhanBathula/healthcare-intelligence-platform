import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

CLINICAL_ENTITIES = [
    "PROBLEM", "TEST", "TREATMENT", "MEDICATION", "DOSAGE",
    "ROUTE", "FREQUENCY", "DURATION", "ANATOMY", "SEVERITY",
    "TEMPORAL", "NEGATION", "PATIENT", "PHYSICIAN", "FACILITY",
    "LAB_VALUE", "VITAL_SIGN", "PROCEDURE"
]


class ClinicalNERPipeline:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 device: int = 0 if torch.cuda.is_available() else -1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                           aggregation_strategy="simple", device=device)
        try:
            self.sci_nlp = spacy.load("en_core_sci_lg")
        except OSError:
            self.sci_nlp = spacy.load("en_core_web_sm")
        self.entity_counts = {e: 0 for e in CLINICAL_ENTITIES}

    def extract(self, text: str) -> List[Dict[str, Any]]:
        hf_ents = self.ner(text[:512])
        spacy_doc = self.sci_nlp(text)
        entities = []
        for ent in hf_ents:
            label = ent["entity_group"].upper()
            entities.append({"text": ent["word"], "label": label,
                              "score": round(ent["score"], 4),
                              "start": ent["start"], "end": ent["end"]})
            if label in self.entity_counts:
                self.entity_counts[label] += 1
        for ent in spacy_doc.ents:
            if not any(e["start"] == ent.start_char for e in entities):
                entities.append({"text": ent.text, "label": ent.label_,
                                  "score": 1.0, "start": ent.start_char, "end": ent.end_char})
        return entities

    def batch_extract(self, notes: List[str], batch_size: int = 32) -> List[List[Dict]]:
        results = []
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i+batch_size]
            results.extend([self.extract(note) for note in batch])
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(notes)} notes")
        return results

    def get_precision_stats(self) -> Dict[str, int]:
        return self.entity_counts
