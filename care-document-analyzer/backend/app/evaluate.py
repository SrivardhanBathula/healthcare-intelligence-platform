"""Run from backend/: python -m app.evaluate"""
import json
from pathlib import Path
from .extractor import demo_extract

def main():
    examples=json.loads((Path(__file__).resolve().parent.parent/"data"/"labeled_examples.json").read_text())
    fields=["resident_id","event_type","event_date","medications_mentioned","risk_flags"]
    correct={field:0 for field in fields}; failures=[]
    for item in examples:
        predicted=demo_extract(item["text"]).model_dump(mode="json")
        for field in fields:
            actual=predicted.get(field); expected=item["expected"].get(field)
            if field=="event_date" and actual: actual=str(actual)
            if actual==expected: correct[field]+=1
            else: failures.append({"id":item["id"],"field":field,"expected":expected,"actual":actual})
    print(json.dumps({"n":len(examples),"accuracy_by_field":{f:round(correct[f]/len(examples),3) for f in fields},
                      "failures":failures,"note":"Tiny synthetic demo set; not a clinical performance claim."},indent=2))
if __name__=="__main__": main()
