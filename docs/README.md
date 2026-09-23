# Documentation

Design and reference documents for the Scientific Paper Search System. Each one
is a Phase 1 deliverable listed in [../Roadmap.md](../Roadmap.md).

| Document | Contents | Roadmap task |
|---|---|---|
| `requirements.md` | Target users and their goals, functional requirements (search, filter, graph exploration, Q&A, summarization, method comparison, related-work recommendation), non-functional requirements (response time, answer faithfulness, citation accuracy, dataset size, cost limits), 5–8 use cases with example queries, and the scope limits. Covers Phase 2 features too, so the design supports them from the start. | 1.2 |
| `architecture.md` | The end-to-end pipeline diagram (collection → extraction → KG → indexing → retrieval → RAG → chatbot), each module's responsibility with its inputs and outputs, the interfaces between modules, the provenance-tracking design that lets answers cite their sources, and a record of the key design decisions and why they were made. | 1.3 |
| `decisions.md` | The open design decisions and their resolutions: reference snowballing, chunking strategy, the faithfulness review protocol, LLM choice, embedding model, and team conventions. Each entry keeps the rejected options and the reason for the choice. | 1.3 |
| `kg-schema.md` | The Knowledge Graph model: node types (`Paper`, `Author`, `Venue`, `Topic`, `Method`, `Dataset`), relationships (`AUTHORED`, `CITES`, `PUBLISHED_IN`, `HAS_TOPIC`, `USES_METHOD`, `USES_DATASET`), their properties, the unique keys used for deduplication (DOI, arXiv ID, Semantic Scholar ID), a schema diagram, and the example Cypher queries the system must answer. | 2.1 |
| `learning-path.md` | Self-study resources for the stack: arc42 documentation, the Semantic Scholar and arXiv APIs, PyMuPDF, Knowledge Graph concepts (Stanford CS520), Neo4j and Cypher (GraphAcademy), spaCy NER and LLM-based extraction, Sentence-Transformers and FAISS, and BM25 (`rank_bm25`). | 1.4 |

`requirements.md` and `decisions.md` are drafted (2026-09-23); the rest are written
during the remainder of September and October 2026.
