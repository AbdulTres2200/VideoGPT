"""
Test script for RAG system with Qdrant Cloud.
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

    all_good = True

    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✓ OPENAI_API_KEY is set")
        print(f"  Key starts with: {api_key[:7]}...")
    else:
        print("✗ OPENAI_API_KEY is NOT set")
        print("  Add it to .env.local: OPENAI_API_KEY=sk-...")
        all_good = False

    # Check Qdrant Cloud credentials
    qdrant_endpoint = os.getenv('QDRANT_ENDPOINT')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if qdrant_endpoint:
        print("✓ QDRANT_ENDPOINT is set")
        print(f"  Endpoint: {qdrant_endpoint[:40]}...")
    else:
        print("✗ QDRANT_ENDPOINT is NOT set")
        print("  Add it to .env.local: QDRANT_ENDPOINT=https://...")
        all_good = False

    if qdrant_api_key:
        print("✓ QDRANT_API_KEY is set")
        print(f"  Key starts with: {qdrant_api_key[:8]}...")
    else:
        print("✗ QDRANT_API_KEY is NOT set")
        print("  Add it to .env.local: QDRANT_API_KEY=...")
        all_good = False

    # Check data/results directory
    results_dir = Path(__file__).parent.parent / 'data' / 'results'
    if results_dir.exists():
        json_files = list(results_dir.glob('*.json'))
        print(f"✓ data/results exists ({len(json_files)} JSON files)")
    else:
        print("✗ data/results directory not found")
        all_good = False

    # Check required packages
    print("\nChecking required packages...")

    packages = [
        ('langchain_openai', 'langchain-openai'),
        ('langchain_community', 'langchain-community'),
        ('qdrant_client', 'qdrant-client'),
        ('sentence_transformers', 'sentence-transformers'),
    ]

    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} not installed")
            print(f"  Run: pip install {package_name}")
            all_good = False

    print("\n" + "=" * 70)
    if all_good:
        print("✓ Setup looks good!")
    else:
        print("✗ Please fix the issues above before testing.")
    print("=" * 70)

    return all_good


def test_qdrant_connection():
    """Test connection to Qdrant Cloud."""
    print("\n" + "=" * 70)
    print("TESTING QDRANT CONNECTION")
    print("=" * 70)

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.getenv('QDRANT_ENDPOINT'),
            api_key=os.getenv('QDRANT_API_KEY')
        )

        # Get collection info
        collection_info = client.get_collection("video_insights")
        points_count = collection_info.points_count

        print(f"✓ Connected to Qdrant Cloud")
        print(f"  Collection: video_insights")
        print(f"  Documents: {points_count}")

        if points_count == 0:
            print("\n⚠ Warning: Collection is empty!")
            print("  Run: python src/processing/embedding_qdrant.py")
            return False

        return True

    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        return False


def test_rag_query():
    """Test RAG query system."""
    print("\n" + "=" * 70)
    print("TESTING RAG QUERY")
    print("=" * 70)

    try:
        from core.RAG import VideoRAGQuery

        print("Initializing RAG system...")
        rag = VideoRAGQuery()
        print("✓ RAG system initialized")

        # Test query based on available data
        test_question = "What is the contractor onboarding process?"
        print(f"\nTest question: {test_question}\n")

        response = rag.query(test_question, return_sources=True)

        print("-" * 70)
        print("ANSWER:")
        print("-" * 70)
        print(response.get('answer', 'No answer generated'))

        if response.get('sources'):
            print("\n" + "-" * 70)
            print("SOURCES:")
            print("-" * 70)
            for src in response['sources']:
                print(f"  - {src.get('video_name', 'Unknown')} ({src.get('source_file', 'Unknown')})")

        print("\n✓ RAG query test passed!")
        return True

    except Exception as e:
        import traceback
        print(f"✗ RAG query failed: {e}")
        traceback.print_exc()
        return False


def test_streaming_query():
    """Test streaming RAG query."""
    print("\n" + "=" * 70)
    print("TESTING STREAMING QUERY")
    print("=" * 70)

    try:
        from core.RAG import VideoRAGQuery

        rag = VideoRAGQuery()
        test_question = "How does the HR review workflow work?"

        print(f"Test question: {test_question}\n")
        print("Streaming response:")
        print("-" * 70)

        full_response = ""
        sources = []

        for chunk in rag.query_stream(test_question):
            if chunk['type'] == 'content':
                print(chunk['content'], end='', flush=True)
                full_response += chunk['content']
            elif chunk['type'] == 'sources':
                sources = chunk['sources']
            elif chunk['type'] == 'done':
                print("\n")

        print("-" * 70)
        print(f"Total response length: {len(full_response)} characters")
        print(f"Sources: {len(sources)}")

        print("\n✓ Streaming query test passed!")
        return True

    except Exception as e:
        import traceback
        print(f"✗ Streaming query failed: {e}")
        traceback.print_exc()
        return False


def test_conversation_history():
    """Test conversation history functionality."""
    print("\n" + "=" * 70)
    print("TESTING CONVERSATION HISTORY")
    print("=" * 70)

    try:
        from core.RAG import VideoRAGQuery

        rag = VideoRAGQuery()

        # First question
        q1 = "What is the onboarding process?"
        print(f"Q1: {q1}")
        r1 = rag.query(q1)
        print(f"A1: {r1['answer'][:200]}...\n")

        # Build conversation history
        history = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": r1['answer']}
        ]

        # Follow-up question
        q2 = "What happens after that?"
        print(f"Q2 (follow-up): {q2}")
        r2 = rag.query(q2, conversation_history=history)
        print(f"A2: {r2['answer'][:200]}...")

        print("\n✓ Conversation history test passed!")
        return True

    except Exception as e:
        import traceback
        print(f"✗ Conversation history test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test RAG system')
    parser.add_argument('--setup', action='store_true', help='Only check setup')
    parser.add_argument('--connection', action='store_true', help='Test Qdrant connection')
    parser.add_argument('--query', action='store_true', help='Test RAG query')
    parser.add_argument('--stream', action='store_true', help='Test streaming query')
    parser.add_argument('--history', action='store_true', help='Test conversation history')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    # If no args, run setup check and basic query test
    if not any(vars(args).values()):
        args.setup = True
        args.query = True

    results = {}

    if args.setup or args.all:
        results['setup'] = check_setup()

    if args.connection or args.all:
        results['connection'] = test_qdrant_connection()

    if args.query or args.all:
        results['query'] = test_rag_query()

    if args.stream or args.all:
        results['stream'] = test_streaming_query()

    if args.history or args.all:
        results['history'] = test_conversation_history()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("✓ All tests passed!" if all_passed else "✗ Some tests failed"))

    sys.exit(0 if all_passed else 1)
