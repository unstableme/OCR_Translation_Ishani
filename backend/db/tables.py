from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from db.connection import Base

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str | None] = mapped_column(String)
    stored_path: Mapped[str | None] = mapped_column(String)
    upload_time: Mapped[datetime | None] = mapped_column(DateTime, default=func.now())
    status: Mapped[str] = mapped_column(String, default="pending")


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"))
    extracted_text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=func.now())
    status: Mapped[str] = mapped_column(String, default="Extracted")


class Translation(Base):
    __tablename__ = "translations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"))
    translated_text: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=func.now())
    status: Mapped[str] = mapped_column(String, default="Completed")


class AudioTranscription(Base):
    __tablename__ = "audio_transcriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"))
    transcribed_text: Mapped[str | None] = mapped_column(Text)
    language_detected: Mapped[str | None] = mapped_column(String)
    audio_duration: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=func.now())
    status: Mapped[str] = mapped_column(String, default="Transcribed")
