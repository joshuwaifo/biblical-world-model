from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    abbreviation = Column(String, nullable=False)
    testament = Column(String, nullable=False)   # OT | NT | Deuterocanon | Ethiopian
    canonical_order = Column(Integer, nullable=False)
    chapters = relationship("Chapter", back_populates="book")


class Chapter(Base):
    __tablename__ = "chapters"
    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    number = Column(Integer, nullable=False)
    book = relationship("Book", back_populates="chapters")
    verses = relationship("Verse", back_populates="chapter")


class Verse(Base):
    __tablename__ = "verses"
    id = Column(Integer, primary_key=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=False)
    number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    chapter = relationship("Chapter", back_populates="verses")


# Proper-noun entities extracted by POS tagger — no semantic labels.
class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    canonical_form = Column(String, nullable=False, unique=True)
    first_mention_verse_id = Column(Integer, ForeignKey("verses.id"))
    mentions = relationship("EntityMention", back_populates="entity")


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    verse_id = Column(Integer, ForeignKey("verses.id"), nullable=False)
    entity = relationship("Entity", back_populates="mentions")


# Graph edges persisted so the graph can be reconstructed without re-computing.
class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id = Column(Integer, primary_key=True)
    src_type = Column(String, nullable=False)   # "verse" | "entity"
    src_id = Column(Integer, nullable=False)
    dst_type = Column(String, nullable=False)
    dst_id = Column(Integer, nullable=False)
    edge_type = Column(String, nullable=False)  # PRECEDES | IN_CHAPTER | MENTIONS | QUOTES | ECHOES | CO_OCCURS
    weight = Column(Float, default=1.0)


# GNN-trained embeddings stored for retrieval.
class NodeEmbedding(Base):
    __tablename__ = "node_embeddings"
    id = Column(Integer, primary_key=True)
    node_type = Column(String, nullable=False)
    node_id = Column(Integer, nullable=False)
    vector_json = Column(Text, nullable=False)   # JSON-serialised float list
