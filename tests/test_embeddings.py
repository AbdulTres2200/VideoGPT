"""
Test script to query the embedded video insights.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

load_dotenv('.env.local')

# Configuration
VECTOR_DB_DIR = 'data/vector_db'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    """Load the existing vector store."""
    if not os.path.exists(VECTOR_DB_DIR):
        raise ValueError(f"Vector database not found: {VECTOR_DB_DIR}")
    
    print(f"Loading vector database from: {VECTOR_DB_DIR}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore


def search_and_display(vectorstore, query: str, k: int = 5):
    """
    Search the vector database and display results.
    
    Args:
        vectorstore: Chroma vector store
        query: Search query
        k: Number of results to return
    """
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}\n")
    
    # Perform similarity search
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{'─'*70}")
        print(f"Result {i} (Similarity Score: {score:.4f})")
        print(f"{'─'*70}")
        print(f"Video: {doc.metadata.get('video_name', 'N/A')}")
        print(f"Video ID: {doc.metadata.get('video_id', 'N/A')}")
        print(f"Content Type: {doc.metadata.get('content_type', 'N/A')}")
        print(f"Source File: {doc.metadata.get('source_file', 'N/A')}")
        
        if doc.metadata.get('chunk_index') is not None:
            print(f"Chunk: {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}")
        
        print(f"\nContent:")
        print(f"{doc.page_content[:500]}...")
        if len(doc.page_content) > 500:
            print(f"[... {len(doc.page_content) - 500} more characters]")
        print()


def interactive_search():
    """Interactive search interface."""
    print("="*70)
    print("VIDEO INSIGHTS EMBEDDING TEST")
    print("="*70)
    
    vectorstore = load_vectorstore()
    
    # Get collection info
    collection = vectorstore._collection
    count = collection.count()
    print(f"✓ Loaded vector database with {count} document chunks")
    print()
    
    print("Enter search queries (type 'quit' or 'exit' to stop)")
    print("="*70)
    
    while True:
        query = input("\n🔍 Search query: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        try:
            k = input("Number of results (default 5): ").strip()
            k = int(k) if k else 5
        except ValueError:
            k = 5
        
        search_and_display(vectorstore, query, k)


def test_sample_queries():
    """Test with sample queries."""
    print("="*70)
    print("VIDEO INSIGHTS EMBEDDING TEST - Sample Queries")
    print("="*70)
    
    vectorstore = load_vectorstore()
    
    # Get collection info
    collection = vectorstore._collection
    count = collection.count()
    print(f"✓ Loaded vector database with {count} document chunks\n")
    
    # Sample queries
    sample_queries = [
        "How to create a template?",
        "What is corporate auto-profiling?",
        "Designer studio features",
        "Product options and pricing",
        "How to manage orders?",
    ]
    
    for query in sample_queries:
        search_and_display(vectorstore, query, k=3)
        input("\nPress Enter to continue to next query...")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        test_sample_queries()
    else:
        interactive_search()


if __name__ == "__main__":
    main()

