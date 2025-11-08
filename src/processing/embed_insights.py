"""
Embed video insights JSON files for RAG pipeline.
Supports query-based filtering, batch processing, and resume capability.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv('.env.local')

# Configuration
RESULTS_DIR = 'data/results'
VECTOR_DB_DIR = 'data/vector_db'
EMBEDDING_PROGRESS_FILE = 'data/embedding_progress.json'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Processing parameters - Adjust these values
START_INDEX = 0  # Start from this file index (0-based)
FILE_COUNT = None  # Number of files to process (None = process all remaining)
QUERY_FILTER = None  # Filter files by name (e.g., "OnPrintShop" or None for all)


class InsightsEmbedder:
    """Extract and embed video insights for RAG with resume capability."""
    
    def __init__(self):
        """Initialize the embedder."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Initialize embeddings
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Load progress
        self.progress = self._load_progress()
        
        # Initialize or load vector store
        self.vectorstore = self._init_vectorstore()
    
    def _load_progress(self) -> Dict:
        """Load embedding progress from previous run."""
        if os.path.exists(EMBEDDING_PROGRESS_FILE):
            with open(EMBEDDING_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            'processed': [],
            'failed': [],
            'last_file_index': -1,
            'total_chunks': 0
        }
    
    def _save_progress(self):
        """Save current progress."""
        with open(EMBEDDING_PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _init_vectorstore(self) -> Chroma:
        """Initialize or load existing vector store."""
        if os.path.exists(VECTOR_DB_DIR):
            print(f"Loading existing vector database from: {VECTOR_DB_DIR}")
            return Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector database at: {VECTOR_DB_DIR}")
            # Create empty vector store
            return Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=self.embeddings
            )
    
    def list_json_files(self, query_filter: Optional[str] = None) -> List[Path]:
        """
        List all JSON files, optionally filtered by query.
        
        Args:
            query_filter: Optional string to filter filenames (case-insensitive)
            
        Returns:
            List of JSON file paths
        """
        results_path = Path(RESULTS_DIR)
        if not results_path.exists():
            raise ValueError(f"Results directory not found: {RESULTS_DIR}")
        
        all_files = sorted(results_path.glob('*.json'))
        
        if query_filter:
            query_lower = query_filter.lower()
            filtered_files = [
                f for f in all_files
                if query_lower in f.name.lower()
            ]
            return filtered_files
        
        return all_files
    
    def extract_text_from_insights(self, insights_data: Dict) -> Dict[str, str]:
        """Extract all text content from insights JSON."""
        extracted = {
            'video_id': insights_data.get('id', ''),
            'video_name': insights_data.get('name', ''),
            'transcript': '',
            'keywords': [],
            'labels': [],
            'summary': ''
        }
        
        # Extract transcript
        videos = insights_data.get('videos', [])
        if videos:
            insights = videos[0].get('insights', {})
            transcript = insights.get('transcript', [])
            if transcript:
                transcript_texts = [t.get('text', '') for t in transcript if t.get('text')]
                extracted['transcript'] = ' '.join(transcript_texts)
        
        # Extract keywords
        summarized_insights = insights_data.get('summarizedInsights', {})
        keywords = summarized_insights.get('keywords', [])
        extracted['keywords'] = [kw.get('name', '') for kw in keywords if kw.get('name')]
        
        # Extract labels
        labels = summarized_insights.get('labels', [])
        extracted['labels'] = [label.get('name', '') for label in labels if label.get('name')]
        
        # Extract summary if available
        summary = summarized_insights.get('summary', {})
        if summary:
            extracted['summary'] = summary.get('text', '')
        
        return extracted
    
    def create_documents(self, extracted_data: Dict, filename: str) -> List[Document]:
        """Create LangChain documents from extracted data."""
        documents = []
        video_id = extracted_data['video_id']
        video_name = extracted_data['video_name']
        
        base_metadata = {
            'video_id': video_id,
            'video_name': video_name,
            'source_file': filename,
            'content_type': 'metadata'
        }
        
        # Document 1: Full transcript (chunked)
        if extracted_data['transcript']:
            transcript_chunks = self.text_splitter.split_text(extracted_data['transcript'])
            for i, chunk in enumerate(transcript_chunks):
                metadata = {
                    **base_metadata,
                    'content_type': 'transcript',
                    'chunk_index': i,
                    'total_chunks': len(transcript_chunks)
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        # Document 2: Keywords summary
        if extracted_data['keywords']:
            keywords_text = f"Keywords: {', '.join(extracted_data['keywords'])}"
            metadata = {
                **base_metadata,
                'content_type': 'keywords'
            }
            documents.append(Document(page_content=keywords_text, metadata=metadata))
        
        # Document 3: Labels summary
        if extracted_data['labels']:
            labels_text = f"Labels: {', '.join(extracted_data['labels'])}"
            metadata = {
                **base_metadata,
                'content_type': 'labels'
            }
            documents.append(Document(page_content=labels_text, metadata=metadata))
        
        # Document 4: Summary if available
        if extracted_data['summary']:
            metadata = {
                **base_metadata,
                'content_type': 'summary'
            }
            documents.append(Document(page_content=extracted_data['summary'], metadata=metadata))
        
        return documents
    
    def process_json_file(self, json_path: Path, force_reprocess: bool = False) -> Optional[List[Document]]:
        """Process a single JSON insights file."""
        # Check if already processed (unless force_reprocess is True)
        if not force_reprocess and any(p.get('filename') == json_path.name for p in self.progress['processed']):
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                insights_data = json.load(f)
            
            extracted = self.extract_text_from_insights(insights_data)
            documents = self.create_documents(extracted, json_path.name)
            
            return documents
        except Exception as e:
            print(f"  ✗ Error processing {json_path.name}: {e}")
            self.progress['failed'].append({
                'filename': json_path.name,
                'error': str(e)
            })
            self._save_progress()
            return None
    
    def embed_batch(self, start_index: int = 0, count: Optional[int] = None, query_filter: Optional[str] = None):
        """
        Process and embed a batch of JSON files.
        
        Args:
            start_index: Index to start from (0-based)
            count: Number of files to process (None = process all remaining)
            query_filter: Optional string to filter filenames
        """
        print("=" * 70)
        print("VIDEO INSIGHTS EMBEDDING")
        print("=" * 70)
        
        # List files
        all_files = self.list_json_files(query_filter=query_filter)
        total_files = len(all_files)
        
        if query_filter:
            print(f"Query filter: '{query_filter}'")
        print(f"Total files found: {total_files}")
        print(f"Start index: {start_index}")
        print(f"Files to process: {count if count else 'all remaining'}")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
        print("=" * 70)
        print()
        
        # Determine files to process
        end_index = start_index + count if count else total_files
        files_to_process = all_files[start_index:end_index]
        
        print(f"Processing files {start_index} to {end_index-1} ({len(files_to_process)} files)")
        print()
        
        processed_count = 0
        total_chunks_added = 0
        
        for i, json_file in enumerate(files_to_process):
            current_index = start_index + i
            print(f"[{current_index + 1}/{total_files}] {json_file.name}")
            
            documents = self.process_json_file(json_file)
            
            if documents is None:
                print(f"  ⏭️  Skipped (already processed or error)")
                continue
            
            if not documents:
                print(f"  ⚠️  No content extracted")
                continue
            
            # Add documents to vector store
            try:
                # Get current document count before adding
                try:
                    current_count = self.vectorstore._collection.count()
                except:
                    current_count = 0
                
                # Add documents (this APPENDS, does not overwrite)
                self.vectorstore.add_documents(documents)
                
                # Verify documents were added
                try:
                    new_count = self.vectorstore._collection.count()
                    added_count = new_count - current_count
                    if added_count != len(documents):
                        print(f"  ⚠️  Warning: Expected {len(documents)} documents, but {added_count} were added")
                except:
                    pass  # If we can't verify, continue anyway
                
                # Update progress
                self.progress['processed'].append({
                    'filename': json_file.name,
                    'chunks': len(documents),
                    'index': current_index
                })
                self.progress['last_file_index'] = current_index
                self.progress['total_chunks'] += len(documents)
                self._save_progress()
                
                processed_count += 1
                total_chunks_added += len(documents)
                print(f"  ✓ Embedded {len(documents)} chunks (Total in DB: {new_count if 'new_count' in locals() else 'unknown'})")
            except Exception as e:
                print(f"  ✗ Error embedding: {e}")
                self.progress['failed'].append({
                    'filename': json_file.name,
                    'error': str(e)
                })
                self._save_progress()
        
        # Final summary
        print()
        print("=" * 70)
        print("BATCH EMBEDDING SUMMARY")
        print("=" * 70)
        print(f"Files processed: {processed_count}/{len(files_to_process)}")
        print(f"Chunks added: {total_chunks_added}")
        print(f"Total chunks in database: {self.progress['total_chunks']}")
        print(f"Last processed index: {self.progress['last_file_index']}")
        print(f"Failed: {len(self.progress['failed'])}")
        print(f"Vector database: {VECTOR_DB_DIR}")
        print("=" * 70)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search the vector database."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        
        return formatted_results


def main():
    """Main function."""
    embedder = InsightsEmbedder()
    
    # Show current progress
    if embedder.progress['processed']:
        print(f"Already processed: {len(embedder.progress['processed'])} files")
        print(f"Total chunks: {embedder.progress['total_chunks']}")
        print(f"Last index: {embedder.progress['last_file_index']}")
        print()
    
    # Start embedding
    embedder.embed_batch(
        start_index=START_INDEX,
        count=FILE_COUNT,
        query_filter=QUERY_FILTER
    )
    
    # Optional: Test search
    test_query = input("\nEnter a test query (or press Enter to skip): ")
    if test_query:
        results = embedder.search(test_query, k=3)
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Video: {result['metadata'].get('video_name', 'N/A')}")
            print(f"   Type: {result['metadata'].get('content_type', 'N/A')}")
            print(f"   Content: {result['content'][:200]}...")


if __name__ == "__main__":
    main()

