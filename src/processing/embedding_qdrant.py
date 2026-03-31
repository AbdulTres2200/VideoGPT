"""
Embedding pipeline using Qdrant Cloud vector database.
Handles both Azure Video Indexer format and new Google Drive format.
"""
import os
import json
import glob
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from dotenv import load_dotenv, find_dotenv

# Load environment variables
PROJECT_ROOT = Path(find_dotenv('.env.local')).parent
load_dotenv(PROJECT_ROOT / '.env.local')

# Qdrant configuration
QDRANT_ENDPOINT = os.getenv('QDRANT_ENDPOINT')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = "video_insights"

# Results directory
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'


def load_json_insights(folder_path: str) -> List[Document]:
    """
    Load JSON insight files and extract transcript text.
    Handles both Azure Video Indexer format and new Google Drive format.
    """
    all_docs = []
    json_files = glob.glob(f"{folder_path}/**/*.json", recursive=True)
    print(f"Found {len(json_files)} JSON files to process\n")

    for file_path in json_files:
        try:
            print(f"Loading: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract transcript based on format
            transcript = extract_transcript(data)

            if not transcript or not transcript.strip():
                print(f"  ⚠ No transcript found, skipping\n")
                continue

            # Extract metadata
            video_name = data.get('name', Path(file_path).stem)
            video_id = data.get('id', 'unknown')

            # Create document
            doc = Document(
                page_content=transcript,
                metadata={
                    "file": file_path,
                    "video_name": video_name,
                    "video_id": video_id,
                    "source": data.get('source', 'azure'),
                    "platform": "OnPrintShop",
                    "type": "video_tutorial"
                }
            )
            all_docs.append(doc)
            print(f"  ✓ Loaded transcript ({len(transcript)} chars)\n")

        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}\n")
            continue

    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def extract_transcript(data: dict) -> str:
    """
    Extract transcript from JSON data.
    Supports multiple formats.
    """
    # Format 1: New Google Drive format (direct transcript field)
    if 'transcript' in data and isinstance(data['transcript'], str):
        return data['transcript']

    # Format 2: Azure Video Indexer format
    if 'videos' in data:
        try:
            transcript_items = data['videos'][0]['insights'].get('transcript', [])
            return ' '.join([item.get('text', '') for item in transcript_items])
        except (KeyError, IndexError):
            pass

    # Format 3: Simple insights format
    if 'insights' in data and 'transcript' in data['insights']:
        transcript = data['insights']['transcript']
        if isinstance(transcript, str):
            return transcript
        elif isinstance(transcript, list):
            return ' '.join([item.get('text', '') for item in transcript])

    return ''


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def enrich_chunk_with_metadata(chunk: Document) -> Document:
    """Inject metadata into page_content for better embeddings."""
    meta = chunk.metadata

    video_name = meta.get("video_name", "Unknown")
    file_name = os.path.basename(meta.get("file", "unknown"))
    content_type = meta.get("type", "video_tutorial")
    platform = meta.get("platform", "OnPrintShop")

    enriched_text = (
        f"[VIDEO: {video_name}] "
        f"[FILE: {file_name}] "
        f"[TYPE: {content_type}] "
        f"[PLATFORM: {platform}] "
        f"{chunk.page_content}"
    )

    chunk.page_content = enriched_text
    return chunk


def create_qdrant_index(chunks: List[Document]) -> Qdrant:
    """
    Create Qdrant vector store from document chunks.
    """
    print("🔍 Enriching chunks with metadata...")
    enriched_chunks = [enrich_chunk_with_metadata(c) for c in chunks]

    print(f"📦 Total chunks to embed: {len(enriched_chunks)}")
    print("⚙️ Initializing embeddings model (text-embedding-3-large)...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    print(f"🚀 Embedding and uploading to Qdrant Cloud...")
    print(f"   Endpoint: {QDRANT_ENDPOINT}")
    print(f"   Collection: {COLLECTION_NAME}")

    try:
        vectorstore = Qdrant.from_documents(
            documents=enriched_chunks,
            embedding=embeddings,
            url=QDRANT_ENDPOINT,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=True  # Recreate collection if exists
        )

        print(f"✅ Successfully embedded {len(enriched_chunks)} chunks to Qdrant Cloud!")

        # Create payload indexes for efficient filtering
        print("🔧 Creating payload indexes...")
        create_payload_indexes()

        return vectorstore

    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client for direct operations."""
    return QdrantClient(
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY
    )


def create_payload_indexes(client: QdrantClient = None):
    """
    Create necessary payload indexes for efficient filtering.
    Call this after embedding to ensure indexes exist.
    """
    if client is None:
        client = get_qdrant_client()

    indexes_to_create = [
        'metadata.user_email',
        'metadata.file',
    ]

    for field_name in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✅ Created payload index for {field_name}")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"ℹ️  Index for {field_name} already exists")
            else:
                print(f"⚠️  Could not create index for {field_name}: {e}")


def get_vectorstore() -> Qdrant:
    """Get existing Qdrant vectorstore for queries."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return Qdrant(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )


def embed_single_file(file_path: str) -> bool:
    """
    Embed a single JSON insight file to Qdrant.
    Useful for adding new videos without re-embedding everything.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        transcript = extract_transcript(data)
        if not transcript:
            print(f"No transcript found in {file_path}")
            return False

        doc = Document(
            page_content=transcript,
            metadata={
                "file": file_path,
                "video_name": data.get('name', Path(file_path).stem),
                "video_id": data.get('id', 'unknown'),
                "source": data.get('source', 'google_drive'),
                "platform": "OnPrintShop",
                "type": "video_tutorial"
            }
        )

        chunks = split_documents([doc])
        enriched_chunks = [enrich_chunk_with_metadata(c) for c in chunks]

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Add to existing collection (don't recreate)
        vectorstore = Qdrant.from_documents(
            documents=enriched_chunks,
            embedding=embeddings,
            url=QDRANT_ENDPOINT,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=False
        )

        # Ensure payload indexes exist
        create_payload_indexes()

        print(f"✅ Embedded {len(enriched_chunks)} chunks from {file_path}")
        return True

    except Exception as e:
        print(f"❌ Error embedding {file_path}: {e}")
        return False


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Qdrant Cloud Embedding Pipeline")
    print("=" * 60)

    # Load all JSON insights
    docs = load_json_insights(str(RESULTS_DIR))

    if not docs:
        print("No documents to embed!")
        exit(1)

    # Split into chunks
    chunks = split_documents(docs)

    # Create Qdrant index
    vectorstore = create_qdrant_index(chunks)

    print("\n" + "=" * 60)
    print("Embedding complete!")
    print("=" * 60)
