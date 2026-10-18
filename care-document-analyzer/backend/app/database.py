import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Text, DateTime, JSON, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
URL=os.getenv("DATABASE_URL","sqlite:///./care_analyzer.db")
engine=create_engine(URL,connect_args={"check_same_thread":False} if URL.startswith("sqlite") else {},pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine,expire_on_commit=False)
class Base(DeclarativeBase): pass
class Document(Base):
    __tablename__="documents"
    id:Mapped[int]=mapped_column(primary_key=True)
    title:Mapped[str]=mapped_column(String(200))
    content:Mapped[str]=mapped_column(Text)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))
class ExtractionRecord(Base):
    __tablename__="extractions"
    id:Mapped[int]=mapped_column(primary_key=True)
    document_id:Mapped[int]=mapped_column(ForeignKey("documents.id",ondelete="CASCADE"),unique=True)
    payload:Mapped[dict]=mapped_column(JSON)
    latency_ms:Mapped[int]=mapped_column(default=0)
    model_used:Mapped[str]=mapped_column(String(30),default="demo")
Index("ix_documents_created_at",Document.created_at)
def init_db(): Base.metadata.create_all(engine)
