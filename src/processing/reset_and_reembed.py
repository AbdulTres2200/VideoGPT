"""
Reset ChromaDB and re-embed all files.
Use this when the database is corrupted or empty.
"""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env.local')

VECTOR_DB_DIR = Path('data/vector_db')
EMBEDDING_PROGRESS_FILE = Path('data/embedding_progress.json')

def reset_database():
    """Reset the ChromaDB database."""
    print("=" * 70)
    print("RESETTING CHROMADB DATABASE")
    print("=" * 70)
    print()
    
    if VECTOR_DB_DIR.exists():
        print(f"Removing vector database directory: {VECTOR_DB_DIR}")
        shutil.rmtree(VECTOR_DB_DIR)
        print("✓ Database directory removed")
    else:
        print("✓ Database directory doesn't exist (already clean)")
    
    print()
    print("Database reset complete!")
    print()

def reset_progress():
    """Optionally reset embedding progress."""
    if EMBEDDING_PROGRESS_FILE.exists():
        response = input(f"Delete embedding progress file? (y/n): ").strip().lower()
        if response == 'y':
            EMBEDDING_PROGRESS_FILE.unlink()
            print("✓ Progress file deleted")
        else:
            print("  Keeping progress file (will resume from last position)")
    else:
        print("  No progress file found")

def main():
    """Main function."""
    print()
    print("This script will:")
    print("  1. Delete the ChromaDB database directory")
    print("  2. Optionally delete the embedding progress file")
    print("  3. You can then run embed_insights.py to re-embed all files")
    print()
    
    response = input("Continue with reset? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    reset_database()
    reset_progress()
    
    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("To re-embed all files, run:")
    print("  python src/processing/embed_insights.py")
    print()
    print("This will process all JSON files in data/results/ and create")
    print("fresh embeddings in the new database.")
    print()

if __name__ == "__main__":
    main()

