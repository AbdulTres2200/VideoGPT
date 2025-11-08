"""
Quick test script for RAG system.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

load_dotenv('.env.local')

def check_setup():
    """Check if everything is set up correctly."""
    print("=" * 70)
    print("RAG SYSTEM SETUP CHECK")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✓ OPENAI_API_KEY is set")
        print(f"  Key starts with: {api_key[:7]}...")
    else:
        print("✗ OPENAI_API_KEY is NOT set")
        print("  Add it to .env.local: OPENAI_API_KEY=sk-...")
        return False
    
    # Check vector DB
    if os.path.exists('data/vector_db'):
        print("✓ Vector database exists")
    else:
        print("✗ Vector database not found")
        print("  Run: python src/processing/embed_insights.py")
        return False
    
    # Check if packages are installed
    try:
        import langchain_openai
        print("✓ langchain-openai installed")
    except ImportError:
        print("✗ langchain-openai not installed")
        print("  Run: pip install -r requirements.txt")
        return False
    
    try:
        import chromadb
        print("✓ chromadb installed")
    except ImportError:
        print("✗ chromadb not installed")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Setup looks good! You can run: python src/core/rag_query.py")
    print("=" * 70)
    return True


if __name__ == "__main__":
    if check_setup():
        print("\nTesting RAG system...")
        try:
            from core.rag_query import VideoRAGQuery
            rag = VideoRAGQuery()
            
            # Test query
            test_question = "What is OnPrintShop?"
            print(f"\nTest question: {test_question}\n")
            response = rag.query(test_question)
            print(rag.format_response(response))
        except Exception as e:
            print(f"\n✗ Error testing RAG: {e}")
            print("\nMake sure:")
            print("1. OPENAI_API_KEY is set in .env.local")
            print("2. You have internet connection")
            print("3. You have OpenAI API credits")
    else:
        print("\nPlease fix the issues above before testing.")

