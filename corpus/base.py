"""
CorpusConfig — abstract interface for any information space.

A corpus is an ordered hierarchy of text units:

  Collection  (e.g. testament, volume, series, category)
    └─ Document  (e.g. book, paper, article, play)
         └─ Section  (e.g. chapter, scene, page, section)
              └─ Unit  (e.g. verse, paragraph, line, sentence)  ← atomic

The learning pipeline (Word2Vec → graph → GNN → FAISS) is the same for every
corpus.  Only this interface is corpus-specific:
  - how to load unit metadata from the DB
  - how to format / parse unit references
  - what the hierarchy levels are called
  - how to resolve "give me all units in document X"

To add a new corpus:
  1. Write an ingest script that populates the DB (using the generic schema or
     a corpus-specific one) and trains Word2Vec + builds the graph.
  2. Subclass CorpusConfig and implement the four abstract methods.
  3. Register the subclass in corpus/registry.py.
  4. Set CORPUS_ID in config.py (or environment variable).
"""

from abc import ABC, abstractmethod


class CorpusConfig(ABC):
    # ── Human-readable labels ──────────────────────────────────────────────
    corpus_id: str          # machine key, e.g. "bible", "shakespeare", "arxiv"
    corpus_name: str        # display name
    unit_label: str         # what atomic units are called  ("verse", "paragraph")
    section_label: str      # grouping above unit           ("chapter", "scene")
    document_label: str     # grouping above section        ("book", "play")
    collection_label: str   # top-level grouping            ("testament", "volume")

    # ── Metadata loader ────────────────────────────────────────────────────
    @abstractmethod
    def load_unit_metadata(self, db) -> dict[int, dict]:
        """
        Return a dict mapping unit_id (int) → metadata dict.

        Every metadata dict MUST contain:
          "id"           int    — same as the key
          "text"         str    — full unit text
          "ref"          str    — human-readable reference (e.g. "GEN 1:1")
          "document"     str    — document name/abbreviation
          "collection"   str    — collection name (may be empty string)
          "can_order"    int    — canonical ordering integer for the document

        Additional corpus-specific fields are allowed.
        """

    @abstractmethod
    def build_document_lookup(self, db) -> dict[str, str]:
        """
        Return a mapping from every reasonable user-supplied document name
        (full name, abbreviation, lowercase, etc.) to the canonical document
        identifier used in unit metadata["document"].

        Example for the Bible: {"genesis": "GEN", "gen": "GEN", ...}
        """

    @abstractmethod
    def units_in_documents(
        self,
        unit_meta: dict[int, dict],
        document_names: list[str],
    ) -> list[int]:
        """
        Return all unit_ids belonging to the given canonical document names.
        `document_names` are already resolved through build_document_lookup.
        """

    @abstractmethod
    def lookup_unit_by_ref(self, ref: str, unit_meta: dict[int, dict]) -> int:
        """
        Parse a reference string (corpus-specific format) and return the
        corresponding unit_id.  Raise ValueError if not found.
        """
