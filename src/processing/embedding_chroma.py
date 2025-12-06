from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv('.env.local')

def load_all_json_insights(folder_path: str):
    import glob
    all_docs = []

    jq_schema='(.videos[0].insights.transcript // [] | map(.text) | join(" "))'
    
    # Use recursive glob to find JSON files in subdirectories too
    json_files = glob.glob(f"{folder_path}/**/*.json", recursive=True)
    print(f"Found {len(json_files)} JSON files to process\n")
    
    for file_path in json_files:
        try:
            print(f"Loading: {file_path}")
            loader = JSONLoader(file_path=file_path, jq_schema=jq_schema)
            docs = loader.load()
            
            for d in docs:
                d.metadata["file"] = file_path
                d.metadata["platform"] = "OnPrintShop"
                d.metadata["type"] = "video_tutorial"
            
            all_docs.extend(docs)
            print(f"  ✓ Loaded {len(docs)} document(s)\n")
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}\n")
            continue
        
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Show example chunk (safely check if chunks exist)
    if chunks:
        example_idx = min(10, len(chunks) - 1)  # Use index 10 or last chunk if fewer exist
        document = chunks[example_idx]
        print(f"\nExample chunk (index {example_idx}):")
        print(f"Content preview: {document.page_content[:200]}...")
        print(f"Metadata: {document.metadata}")
    else:
        print("No chunks created from documents.")

    return chunks

def enrich_chunk_with_metadata(chunk: Document) -> Document:
    """Inject metadata directly into page_content for better embeddings."""
    meta = chunk.metadata

    title = meta.get("name", "Unknown Title")
    file_name = os.path.basename(meta.get("file", "unknown"))
    type_ = meta.get("type", "video_tutorial")
    platform = meta.get("platform", "OnPrintShop")

    enriched_text = (
        f"[TITLE: {title}] "
        f"[FILE: {file_name}] "
        f"[TYPE: {type_}] "
        f"[PLATFORM: {platform}] "
        f"{chunk.page_content}"
    )

    # overwrite content for embedding
    chunk.page_content = enriched_text
    return chunk


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embed_documents_safe(docs: List[Document], embeddings, index_path: str, collection_name: str):
    """Retry wrapper so embedding doesn't fail if OpenAI times out."""
    return Chroma.from_documents(
        docs, 
        embeddings,
        persist_directory=index_path,
        collection_name=collection_name
    )


def create_chroma_index(chunks: List[Document], index_path: str = "chroma_index"):
    """
    Takes your list of chunked Documents, enriches them, embeds them,
    and saves a Chroma index to disk.

    Args:
        chunks: list of Document objects (your 9,740 chunks)
        index_path: folder to save Chroma files
    """

    print("🔍 Enriching chunks with metadata...")
    enriched_chunks = [enrich_chunk_with_metadata(c) for c in chunks]

    print(f"📦 Total chunks to embed: {len(enriched_chunks)}")
    print("⚙️ Initializing embeddings model (text-embedding-3-large)...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    print("🚀 Embedding and building Chroma index...")
    print(f"   This may take several minutes for {len(enriched_chunks)} chunks...")
    
    try:
        vectorstore = embed_documents_safe(
            enriched_chunks, 
            embeddings,
            index_path=index_path,
            collection_name="langchain_onprintshop_chroma"
        )
        
        # Verify documents were actually added
        doc_count = vectorstore._collection.count()
        print(f"💾 Chroma index saved to: {index_path}")
        print(f"📊 Documents in collection: {doc_count}")
        
        if doc_count == 0:
            print("⚠️  WARNING: Collection is empty! Embedding may have failed.")
        elif doc_count != len(enriched_chunks):
            print(f"⚠️  WARNING: Expected {len(enriched_chunks)} documents, but found {doc_count}")
        else:
            print("✅ Embedding complete! All documents saved.")
        
        return vectorstore
    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        import traceback
        traceback.print_exc()
        raise


docs = load_all_json_insights("/Users/macbook/Desktop/PycharmProjects/video_gpt/data/results")
chunks = split_text(docs)
vectorstore = create_chroma_index(
    chunks,
    index_path="chroma_index"
)