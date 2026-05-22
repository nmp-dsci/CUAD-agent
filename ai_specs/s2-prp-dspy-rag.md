# PRP: Build RAG agent DSPy CUAD 41-Agent Evaluation

## Goal
Build a RAG system on legal contract data to extract whole sentences for contract that match a legal clause. 


### Chunking strategy:
 * chunk on whole stenceses 
 * split on contract structure: 1., 1.1, (a), (i), headings, section titles, exhibit schedules, definitions.
 * Use small searchable chunks around 150-350 tokens where possible. These improve semantic matching for precise clause types.
 * Parent chunks for context: Store the full parent section around 500-1,200 tokens. Retrieve the small child chunk, then pass the parent section to the extraction model. This matches the parent-child retrieval idea: small chunks embed well, larger chunks preserve context 
 * Always store offsets: Every chunk should keep: document_id, contract_name, section_number, section_title, page, start_char, end_char, raw_text, normalized_text.

### Embedding / Search Strategy 
Use hybrid search, not dense embeddings alone.
 * Dense embeddings catch meaning: “may not transfer this agreement” can match “assignment restriction.”
 * Sparse/BM25 catches exact legal terms: “most favored nation”, “change of control”, “non-solicit”, “governing law”, “affiliate”, “survival.”

### a good setup 
 * Dense embedding: OpenAI text-embedding-3-large or text-embedding-3-small, or a legal-domain embedding model if you can benchmark it.
 * Sparse search: BM25 or SPLADE-style sparse vectors.
 * Reranker: cross-encoder, ColBERT/late-interaction, or provider reranker.
 * Final extractor: LLM returns structured JSON with exact quote, clause type, normalized answer, source span, and confidence.





