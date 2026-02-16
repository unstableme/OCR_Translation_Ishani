from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from db.connection import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename = Column(String)
    stored_path = Column(String)
    upload_time = Column(DateTime, default=func.now())
    status = Column(String, default="pending")


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    extracted_text = Column(Text)
    confidence = Column(Float)
    created_at = Column(DateTime, default=func.now())
    status = Column(String, default="Extracted")


class Translation(Base):
    __tablename__ = "translations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    translated_text = Column(Text)
    model_used = Column(String)
    created_at = Column(DateTime, default=func.now())
    status = Column(String, default="Completed")
