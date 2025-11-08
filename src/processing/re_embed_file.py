"""
Re-embed a specific file that was marked as processed but isn't in the database.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from embed_insights import InsightsEmbedder
import json

# File to re-embed
TARGET_FILE = "Block PDF Path Creation Illustrator  OnPrintShop_insights.json"

def re_embed_file():
    """Re-embed a specific file."""
    embedder = InsightsEmbedder()
    
    # Find the file
    results_dir = Path('data/results')
    file_path = results_dir / TARGET_FILE
    
    if not file_path.exists():
        print(f"❌ File not found: {TARGET_FILE}")
        return
    
    print(f"Re-embedding: {TARGET_FILE}")
    print("=" * 70)
    
    # Remove from progress if it exists
    if any(p.get('filename') == TARGET_FILE for p in embedder.progress['processed']):
        print("Removing from progress file...")
        embedder.progress['processed'] = [
            p for p in embedder.progress['processed'] 
            if p.get('filename') != TARGET_FILE
        ]
        # Update total chunks
        embedder.progress['total_chunks'] = sum(
            p.get('chunks', 0) for p in embedder.progress['processed']
        )
        embedder._save_progress()
        print("✓ Removed from progress")
    
    # Process and embed
    print(f"\nProcessing file...")
    documents = embedder.process_json_file(file_path, force_reprocess=True)
    
    if not documents:
        print("❌ No documents extracted")
        return
    
    print(f"Extracted {len(documents)} documents")
    
    # Add to vector store
    try:
        current_count = embedder.vectorstore._collection.count()
        print(f"Current database count: {current_count}")
        
        embedder.vectorstore.add_documents(documents)
        
        new_count = embedder.vectorstore._collection.count()
        added_count = new_count - current_count
        
        print(f"New database count: {new_count}")
        print(f"Added {added_count} documents")
        
        if added_count != len(documents):
            print(f"⚠️  Warning: Expected {len(documents)}, but {added_count} were added")
        
        # Update progress
        embedder.progress['processed'].append({
            'filename': TARGET_FILE,
            'chunks': len(documents),
            'index': len(embedder.progress['processed'])
        })
        embedder.progress['total_chunks'] += len(documents)
        embedder._save_progress()
        
        print(f"\n✓ Successfully re-embedded {TARGET_FILE}")
        print(f"  Chunks: {len(documents)}")
        print(f"  Total chunks in DB: {new_count}")
        
    except Exception as e:
        print(f"❌ Error embedding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    re_embed_file()

