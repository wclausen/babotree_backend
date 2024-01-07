import datetime
import enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, UUID, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import mapped_column

from app.database import Base


class Highlight(Base):
    __tablename__ = 'highlights'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    readwise_id = Column(Integer, unique=True)
    text = Column(String(length=2048))
    source_id = Column(UUID)
    source_location = Column(Integer)
    source_end_location = Column(Integer)
    source_location_type = Column(String(length=2048))
    note = Column(String(length=2048))
    color = Column(String(length=2048))
    highlighted_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime)
    external_id = Column(String(length=2048))
    url = Column(String(length=2048))
    is_favorite = Column(Boolean)
    is_discarded = Column(Boolean)
    readwise_url = Column(String(length=2048))

    def __repr__(self):
        return f"Highlight(id={self.id}, text={self.text})"

class HighlightSource(Base):
    __tablename__ = 'highlight_sources'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    readwise_id = Column(Integer, unique=True)
    title = Column(String(length=2048))
    readable_title = Column(String(length=2048))
    author = Column(String(length=2048))
    source = Column(String(length=2048))
    cover_image_url = Column(String(length=2048))
    unique_url = Column(String(length=2048))
    category = Column(String(length=2048))
    readwise_url = Column(String(length=2048))
    source_url = Column(String(length=2048))
    asin = Column(String(length=2048))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"HighlightSource(readwise_id={self.readwise_id}, title={self.title})"


class HighlightSourceOutline(Base):
    __tablename__ = 'highlight_sources_outlines'

    id = mapped_column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    source_id = mapped_column(UUID, ForeignKey('highlight_sources.id'), index=True)
    outline_md = Column(String(length=65536))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SourceType(enum.Enum):
    ALL_HIGHLIGHTS_FROM_SOURCE = 'FULL_HIGHLIGHTS_FROM_SOURCE'
    SOURCE_KEY_WORDS = 'SOURCE_KEY_WORDS'
    HIGHLIGHT_TEXT = 'HIGHLIGHT_TEXT'
    FLASHCARD_TEXT = 'FLASHCARD_TEXT'


class ContentEmbedding(Base):
    __tablename__ = 'content_embeddings'

    id = mapped_column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    # this source_id field will be a foreign key to the particular source table. The actual
    # source table will be determined by the source_type field
    source_id = mapped_column(UUID)
    source_type = Column(String(length=2048))
    embedding = mapped_column(Vector(1536), index=True)
    created_at = mapped_column(DateTime, default=datetime.datetime.utcnow)

class JobResults(Base):
    __tablename__ = 'job_results'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    job_id = Column(UUID)
    job_execution_start_time = Column(DateTime)
    job_execution_end_time = Column(DateTime)
    result = Column(String(length=2048))
    extra_data = Column(JSON)
    created_at = Column(DateTime)


class FlashcardGenerationType(enum.Enum):
    BASIC_FRONT_BACK_FLASHCARD = 'BASIC_FRONT_BACK_FLASHCARD'
    TERM_DEFINITION_FLASHCARD = 'TERM_DEFINITION_FLASHCARD'
    CLOZE_DELETION_FLASHCARD = 'CLOZE_DELETION_FLASHCARD'

class Flashcard(Base):
    __tablename__ = 'flashcards'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    topic = Column(String(length=2048))
    question = Column(String(length=2048))
    answer = Column(String(length=2048))
    generation_type = Column(String(length=512), nullable=False)
    highlight_source_id = Column(UUID, ForeignKey('highlight_sources.id'), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"Flashcard(id={self.id}, question={self.question}, answer={self.answer})"


class FlashcardHighlightSource(Base):
    __tablename__ = 'flashcard_highlight_sources'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    flashcard_id = Column(UUID, ForeignKey('flashcards.id'), index=True)
    highlight_id = Column(UUID, ForeignKey('highlights.id'), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
class KeyValue(Base):
    __tablename__ = 'key_value'

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    key = Column(String(length=2048))
    value = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
