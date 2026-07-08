import argparse
import sys
from pathlib import Path
import chromadb
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# Import project config and parser
from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    RESUMES_DIR,
    COLLECTION_NAME,
    validate_config
)
from parser import parse_resume

def setup_llamaindex_settings(api_key: str = None):
    """Configures LlamaIndex to use Google Gemini LLM and Embeddings."""
    key_to_use = api_key or GEMINI_API_KEY
    
    # Configure Gemini LLM
    Settings.llm = Gemini(
        model=LLM_MODEL,
        api_key=key_to_use,
        temperature=0.1
    )
    
    # Configure Gemini Embeddings
    Settings.embed_model = GeminiEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=key_to_use
    )
    
    # Use our Large-Chunk / Whole-Document strategy
    Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=0)

def build_or_refresh_index(force_rebuild: bool = False):
    """
    Reads resumes from the Resumes directory, parses them,
    and indexes them in the local Chroma DB.
    """
    print("Validating configuration...")
    validate_config()
    
    print("Setting up LLM and Embedding models (Gemini)...")
    setup_llamaindex_settings()
    
    print(f"Initializing local Chroma DB client at {CHROMA_DB_PATH}...")
    db_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # If forcing rebuild, delete existing collection
    if force_rebuild:
        try:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            db_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
            
    chroma_collection = db_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Gather files
    resume_files = []
    for ext in ("*.pdf", "*.docx"):
        resume_files.extend(list(RESUMES_DIR.glob(ext)))
        
    print(f"Found {len(resume_files)} resumes in {RESUMES_DIR}.")
    
    # Parse and construct LlamaIndex Documents
    documents = []
    candidate_names = []
    
    for i, filepath in enumerate(resume_files, 1):
        print(f"[{i}/{len(resume_files)}] Parsing {filepath.name}...")
        try:
            candidate_name, text_content = parse_resume(filepath)
            
            # Create LlamaIndex Document with metadata
            doc = Document(
                text=text_content,
                metadata={
                    "candidate_name": candidate_name,
                    "source_file": filepath.name,
                },
                # Exclude from embeddings to avoid biasing retrieval based on filename metadata
                excluded_embed_metadata_keys=["source_file"],
                excluded_llm_metadata_keys=["source_file"],
            )
            documents.append(doc)
            candidate_names.append(candidate_name)
        except Exception as e:
            print(f"Failed to parse {filepath.name}: {e}")
            
    if not documents:
        print("No documents were successfully parsed. Exiting.")
        sys.exit(1)
        
    print(f"Indexing {len(documents)} documents into Chroma DB collection '{COLLECTION_NAME}'...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n--- Indexing Complete! ---")
    print(f"Unique candidates indexed: {sorted(list(set(candidate_names)))}")
    print(f"Total documents: {len(documents)}")
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume RAG LLM Indexer")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the Chroma DB collection by deleting the existing one first."
    )
    args = parser.parse_args()
    
    # Run the indexer
    build_or_refresh_index(force_rebuild=args.rebuild)
