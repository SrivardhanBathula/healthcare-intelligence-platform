import time
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select
from .database import SessionLocal, init_db, Document, ExtractionRecord
from .schemas import DocumentCreate, Extraction
from .extractor import extract_document
app=FastAPI(title="Care Document Analyzer",version="0.1.0",description="Synthetic-data portfolio demo; not for clinical use.")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
@app.on_event("startup")
def startup(): init_db()
def db_session():
    db=SessionLocal()
    try: yield db
    finally: db.close()
@app.get("/health")
def health(): return {"status":"ok","mode":"demo"}
@app.post("/api/documents")
def create(body:DocumentCreate,db:Session=Depends(db_session)):
    start=time.perf_counter()
    try: result, model_used=extract_document(body.content)
    except Exception as e: raise HTTPException(502,detail="Extraction failed") from e
    elapsed=round((time.perf_counter()-start)*1000)
    doc=Document(title=body.title.strip(),content=body.content); db.add(doc); db.flush()
    db.add(ExtractionRecord(document_id=doc.id,payload=result.model_dump(mode="json"),latency_ms=elapsed,model_used=model_used));
    try: db.commit()
    except Exception as e: db.rollback(); raise HTTPException(500,detail="Could not save document") from e
    db.refresh(doc)
    return {"id":doc.id,"title":doc.title,"extraction":result.model_dump(mode="json"),"created_at":doc.created_at.isoformat(),"latency_ms":elapsed,"model_used":model_used}
@app.get("/api/documents")
def list_docs(q:str|None=None,limit:int=Query(50,ge=1,le=100),db:Session=Depends(db_session)):
    stmt=select(Document,ExtractionRecord).join(ExtractionRecord,ExtractionRecord.document_id==Document.id).order_by(Document.created_at.desc()).limit(limit)
    rows=db.execute(stmt).all(); items=[]
    for doc,rec in rows:
        if q and q.lower() not in doc.content.lower() and q.lower() not in doc.title.lower(): continue
        items.append({"id":doc.id,"title":doc.title,"extraction":rec.payload,"created_at":doc.created_at.isoformat(),"latency_ms":rec.latency_ms,"model_used":rec.model_used})
    return {"items":items,"count":len(items)}
@app.get("/api/evaluation")
def evaluation(db:Session=Depends(db_session)):
    rows=db.execute(select(ExtractionRecord)).scalars().all()
    return {"documents_seen":len(rows),"average_latency_ms":round(sum(x.latency_ms for x in rows)/len(rows),2) if rows else None,"note":"Operational latency only; not accuracy. Run labeled evaluation before reporting accuracy."}
@app.patch("/api/documents/{doc_id}/extraction")
def correct(doc_id:int,payload:Extraction,db:Session=Depends(db_session)):
    row=db.execute(select(ExtractionRecord).where(ExtractionRecord.document_id==doc_id)).scalar_one_or_none()
    if row is None: raise HTTPException(404,detail="Document not found")
    row.payload=payload.model_dump(mode="json"); db.commit(); return {"id":doc_id,"extraction":row.payload,"saved":True}
