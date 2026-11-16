"""Scientific paper search system: Knowledge Graph + RAG.

Sub-packages:

- ``collection``: fetch paper metadata and PDFs from Semantic Scholar and arXiv.
- ``extraction``: extract entities (methods, datasets, topics) from paper text.
- ``graph``: Neo4j Knowledge Graph schema and loaders.
- ``indexing``: chunking, embeddings, FAISS and BM25 index building.
- ``retrieval``: search functions over the built indexes and the graph.
- ``rag``: (Phase 2) retrieval-augmented answer generation.
- ``app``: (Phase 2) API and chatbot UI.
"""

__version__ = "0.1.0"
