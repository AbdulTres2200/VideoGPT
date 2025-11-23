"""
RAG Query System - Ask questions about your video content.
Uses OpenAI GPT-4o-mini with your embedded video insights.
"""
import os
import json
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv('.env.local')

# Configuration
VECTOR_DB_DIR = 'data/vector_db'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI Configuration
# Use gpt-4o-mini for faster responses (2-3x faster than gpt-4o)
# Switch to "gpt-4o" for better quality if needed
OPENAI_MODEL = "gpt-4o-mini"  # Faster model for better response times
TEMPERATURE = 0.0  # Lower temperature for more consistent, factual answers

# Re-ranking Configuration
USE_RERANKING = True  # Enable re-ranking for better relevance
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast and effective re-ranker
FINAL_CHUNKS = 5  # Increased from 3 to 5 for better context and more comprehensive answers


class VideoRAGQuery:
    """RAG system for querying video insights."""
    
    def _load_video_filename_map(self) -> Dict[str, str]:
        """Load mapping from JSON insights filename to original video filename with extension."""
        mapping = {}
        batch_progress_path = Path('data/batch_progress.json')
        
        if batch_progress_path.exists():
            try:
                with open(batch_progress_path, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                # Extract mapping from processed videos
                for item in progress_data.get('processed', []):
                    summary = item.get('summary', {})
                    insights_file = summary.get('insights_file')
                    original_name = summary.get('original_name')
                    
                    if insights_file and original_name:
                        mapping[insights_file] = original_name
            except Exception as e:
                print(f"   Warning: Could not load video filename mapping: {e}")
        
        return mapping
    
    def _get_original_video_filename(self, json_filename: str) -> str:
        """Get original video or text filename with extension from JSON insights filename."""
        # First try to find in mapping
        if json_filename in self.video_filename_map:
            return self.video_filename_map[json_filename]
        
        # Fallback: extract base name and try common video and text extensions
        # JSON files are named like "Video Name_insights.json"
        base_name = json_filename.replace('_insights.json', '')
        
        # Try to find the original file with common extensions (video and text)
        common_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.txt']
        results_dir = Path('data/results')
        
        if results_dir.exists():
            for ext in common_extensions:
                potential_file = results_dir / f"{base_name}{ext}"
                if potential_file.exists():
                    return f"{base_name}{ext}"
        
        # If not found, default to .mp4 (most common for videos)
        return f"{base_name}.mp4"
    
    def __init__(self):
        """Initialize the RAG system."""
        # Load video filename mapping from batch_progress.json if available
        self.video_filename_map = self._load_video_filename_map()
        
        # Create vector database directory if it doesn't exist
        if not os.path.exists(VECTOR_DB_DIR):
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            print(f"Created vector database directory: {VECTOR_DB_DIR}")
        
        print(f"Loading vector database from: {VECTOR_DB_DIR}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store with robust error handling and recovery
        collection_name = "langchain"
        
        try:
            # First, verify database integrity
            import chromadb
            # Note: ChromaDB supports concurrent access, but if embedding is running,
            # the collection count might change. The server will use whatever is available.
            client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            
            # Check if collection exists and get its document count
            collection_count = 0
            collection_exists = False
            try:
                existing_collections = client.list_collections()
                print(f"   DEBUG: Found {len(existing_collections)} collection(s) in database")
                if existing_collections:
                    print(f"   DEBUG: Collection names: {[c.name for c in existing_collections]}")
                
                collection_exists = any(col.name == collection_name for col in existing_collections)
                
                if collection_exists:
                    # Get the actual collection to check document count
                    try:
                        existing_col = client.get_collection(collection_name)
                        collection_count = existing_col.count()
                        print(f"   Found existing collection '{collection_name}' with {collection_count} documents")
                        print(f"   DEBUG: Collection ID: {existing_col.id}")
                    except Exception as e:
                        print(f"   Warning: Could not access collection count: {e}")
                else:
                    print(f"   ⚠️  Collection '{collection_name}' not found in list")
                    # List all collections for debugging
                    if existing_collections:
                        print(f"   Available collections: {[c.name for c in existing_collections]}")
                        # Check if any collection has documents
                        for col in existing_collections:
                            try:
                                col_count = col.count()
                                print(f"   DEBUG: Collection '{col.name}' has {col_count} documents")
                            except:
                                pass
            except Exception as e:
                print(f"   Warning: Could not check collections: {e}")
                print(f"   Attempting to continue...")
            
            # Initialize vector store - Chroma will use existing collection if it exists
            # IMPORTANT: If collection exists, Chroma should automatically use it
            # We need to ensure we're using the same collection that was created during embedding
            if collection_exists and collection_count > 0:
                print(f"   Using existing collection '{collection_name}' with {collection_count} documents")
            
            # Initialize Chroma with explicit collection handling to prevent recreation
            # CRITICAL: If collection exists with data, we MUST reuse it, not create a new one
            if collection_exists and collection_count > 0:
                # Get the existing collection first to ensure we use it
                try:
                    existing_col = client.get_collection(collection_name)
                    # Verify it has documents before proceeding
                    if existing_col.count() > 0:
                        print(f"   ✓ Verified existing collection has {existing_col.count()} documents")
                        # Initialize Chroma with the existing collection
                        self.vectorstore = Chroma(
                            persist_directory=VECTOR_DB_DIR,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        # Double-check we're using the right collection
                        actual_count = self.vectorstore._collection.count()
                        if actual_count == 0:
                            print(f"   ⚠️  WARNING: LangChain wrapper shows 0, but direct access shows {existing_col.count()}")
                            print(f"   Data is safe in database - this is a wrapper connection issue")
                            print(f"   Try restarting server or check ChromaDB connection")
                        else:
                            print(f"   ✓ Successfully connected to existing collection with {actual_count} documents")
                    else:
                        print(f"   ⚠️  Collection exists but is empty - initializing new collection")
                        self.vectorstore = Chroma(
                            persist_directory=VECTOR_DB_DIR,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                except Exception as col_err:
                    print(f"   ⚠️  Error accessing existing collection: {col_err}")
                    print(f"   Initializing new collection (data may be in database but not accessible)")
                    self.vectorstore = Chroma(
                        persist_directory=VECTOR_DB_DIR,
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
            else:
                # No existing collection - create new one
                print(f"   Creating new collection '{collection_name}' (no existing collection found)")
                self.vectorstore = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                # Verify the new collection was created
                try:
                    new_count = self.vectorstore._collection.count()
                    print(f"   ✓ New collection created with {new_count} documents")
                except Exception as e:
                    print(f"   ⚠️  Could not verify new collection: {e}")
            
            # Immediately verify we got the right collection
            if collection_exists and collection_count > 0:
                # Force a fresh count to ensure we're using the existing collection
                try:
                    actual_count = self.vectorstore._collection.count()
                    if actual_count != collection_count:
                        print(f"   ⚠️  WARNING: Collection count mismatch!")
                        print(f"      Expected: {collection_count}, Got: {actual_count}")
                        print(f"      Attempting to force reload...")
                        # Try to get collection directly and verify
                        direct_col = client.get_collection(collection_name)
                        direct_count = direct_col.count()
                        print(f"      Direct access shows: {direct_count} documents")
                        if direct_count > 0 and actual_count == 0:
                            print(f"      ⚠️  LangChain wrapper shows 0 but direct access shows {direct_count}")
                            print(f"      This may be a LangChain Chroma wrapper issue")
                except Exception as verify_err:
                    print(f"   Could not verify collection: {verify_err}")
            
            # Force verify collection is accessible and get actual count
            try:
                collection = self.vectorstore._collection
                count = collection.count()
                
                # If we detected an existing collection but count is 0, something is wrong
                if collection_exists and collection_count > 0 and count == 0:
                    print(f"   ⚠️  WARNING: Collection exists with {collection_count} docs, but vectorstore shows {count}")
                    print(f"   This indicates LangChain Chroma wrapper is not accessing the existing collection")
                    print(f"   Attempting to use direct chromadb collection...")
                    # Try to get the collection directly and use it
                    try:
                        direct_col = client.get_collection(collection_name)
                        direct_count = direct_col.count()
                        print(f"   Direct collection access shows {direct_count} documents")
                        
                        # The issue is that LangChain Chroma might be creating a new collection
                        # Let's try to force it to use the existing one by getting the collection ID
                        # and re-initializing with explicit collection metadata
                        print(f"   Re-initializing vectorstore to use existing collection...")
                        # Delete the vectorstore and recreate to force it to use existing collection
                        del self.vectorstore
                        self.vectorstore = Chroma(
                            persist_directory=VECTOR_DB_DIR,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        # Try again
                        count = self.vectorstore._collection.count()
                        if count == 0 and direct_count > 0:
                            print(f"   ⚠️  Still showing 0 - this is a LangChain Chroma wrapper bug")
                            print(f"   The collection exists ({direct_count} docs) but wrapper can't access it")
                            print(f"   You may need to restart the server after embedding completes")
                    except Exception as e2:
                        print(f"   Could not access collection directly: {e2}")
                elif collection_exists and collection_count > 0 and count == collection_count:
                    print(f"   ✓ Collection '{collection_name}' loaded correctly with {count} documents")
                elif count > 0:
                    print(f"   ✓ Collection '{collection_name}' loaded with {count} documents")
                else:
                    # Collection might be empty or embedding is still in progress
                    if collection_exists and collection_count > 0:
                        print(f"   ⚠️  Collection exists with {collection_count} docs, but wrapper shows {count}")
                        print(f"   This may be a timing issue - documents will be available once embedding completes")
                        print(f"   The server will work with whatever documents are available")
                    else:
                        print(f"   ✓ Collection '{collection_name}' loaded with {count} documents")
                        if count == 0:
                            print(f"   Note: If embedding is running, documents will appear as they're added")
            except Exception as e:
                # If collection access fails, log but don't delete - might be a temporary issue
                print(f"   ⚠️  Collection access error: {e}")
                print(f"   Warning: Could not verify collection count, but continuing...")
                # Try direct access as fallback
                try:
                    if collection_exists:
                        direct_col = client.get_collection(collection_name)
                        direct_count = direct_col.count()
                        print(f"   Direct access shows {direct_count} documents in collection")
                except:
                    print(f"   ⚠️  Still cannot access collection, but will attempt to use it")
                
        except Exception as e:
            # Last resort: log error but don't delete database - data might still be recoverable
            print(f"   ✗ Database error: {e}")
            print(f"   ⚠️  WARNING: Not deleting database - embeddings may still be recoverable")
            print(f"   Attempting to continue with existing database...")
            # Try to create vector store anyway - ChromaDB might handle it
            try:
                self.vectorstore = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                count = self.vectorstore._collection.count()
                print(f"   ✓ Database accessible with {count} documents")
                # Final verification - check if we're using the right collection
                if collection_exists and collection_count > 0 and count == 0:
                    print(f"   ⚠️  CRITICAL: Expected {collection_count} documents but got {count}!")
                    print(f"   This suggests the collection was recreated or data was lost.")
                    print(f"   Check if embedding process is still running or if database was reset.")
            except Exception as e2:
                print(f"   ✗ Could not initialize database: {e2}")
                print(f"   You may need to manually check the database or re-embed files")
                raise
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env.local\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        # Initialize LLM
        print(f"Initializing {OPENAI_MODEL}...")
        self.llm = ChatOpenAI(
            model_name=OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=api_key,
            streaming=True  # Enable streaming
        )
        
        # Initialize faster LLM for query correction (use faster model for speed)
        # Use gpt-4o-mini for correction - it's faster and cheaper, and correction doesn't need the best model
        correction_model = "gpt-4o-mini" if OPENAI_MODEL == "gpt-4o" else OPENAI_MODEL
        self.correction_llm = ChatOpenAI(
            model_name=correction_model,
            temperature=0.0,  # Low temperature for consistent corrections
            api_key=api_key,
            streaming=False,  # Non-streaming for quick response
            timeout=5  # 5 second timeout for correction
        )
        
        # Create retriever - optimize k for speed vs quality balance
        # Increased default k for better retrieval quality
        initial_k = 15  # Increased from 12 to retrieve more candidates for better results
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": initial_k}  # Retrieve chunks - can be increased for vague questions
        )
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if USE_RERANKING:
            try:
                print(f"Loading re-ranker model: {RERANKING_MODEL}...")
                self.reranker = CrossEncoder(RERANKING_MODEL)
                print("✓ Re-ranker loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load re-ranker: {e}")
                print("   Continuing without re-ranking...")
                self.reranker = None
        
        # Create custom prompt template - prioritize showing what IS available
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context (from video transcripts, insights, and metadata).

Your goal is to **extract, organize, and present** ALL relevant information from the context to answer the user's question completely and helpfully.
Typo in a question should not affect the answer. You should fix the typo internally and answer the question based on the corrected question.
---

### 🧩 ANSWERING PRINCIPLES

1. **CRITICAL: Extract and Use Exact Feature Names**
   - **MANDATORY**: Before writing your answer, scan the context for ALL capitalized feature names, menu items, button labels, and technical terms.
   - **Examples of feature names to look for**: "Option Rules," "Additional Option Price," "4over Integration," "Master Templates," "Duplicate Templates," "Resize Option," "Combined Options Price," "Quantity Based Additional Options Price," "Designer Rules," etc.
   - **ALWAYS include these exact names** in your answer when discussing related features.
   - **NEVER paraphrase or simplify** feature names - use them verbatim with exact capitalization (e.g., "Option Rules" NOT "option rules" or "the option rules feature").
   - If the context mentions a feature name, **you MUST include it** in your answer when discussing that feature.

2. **Use Examples from Retrieved Content**
   - **Always ground your answer** in specific examples, quotes, or details from the retrieved context.
   - When explaining features or processes, **cite concrete examples** from the context rather than making generic statements.
   - **Quote specific feature names** directly from the context when available.
   - If you see phrases like "Option Rules," "Additional Option Price," "Combined Options Price," etc. in the context, **you must use these exact terms** in your answer.

3. **Reference Feature Names Exactly as Found**
   - **Preserve exact feature names, menu paths, and terminology** exactly as they appear in the context.
   - Do not paraphrase or simplify feature names - use them verbatim (e.g., "Option Rules," not "option rules" or "the option rules feature").
   - When listing features, use the **exact capitalization and wording** from the documents.
   - **Scan for capitalized terms** - these are often feature names that must be preserved exactly.

4. **Avoid Generic Statements - Ground in Retrieved Text**
   - **Never make generic claims** without backing them up with specific details from the context.
   - Instead of saying "The system has various features," say "The system includes features such as [exact feature names from context]."
   - **Quote or paraphrase specific examples** from the retrieved content to support your points.
   - Every major claim should be traceable to the provided context.
   - **When describing a feature, first identify its exact name from the context, then use that name throughout your answer.**

5. **Extract and Present All Relevant Information**
   - **Actively search** the context for any information related to the question, even if not explicitly stated.
   - If the context describes a process (e.g., "go to My Account → My Orders"), you can reasonably infer it works on mobile if the platform is web-based and mobile-responsive.
   - **Never start with** "The context does not mention..." or "However, it does not specify..." - instead, extract what IS available and present it helpfully.
   - If a process is described for "desktop" or "web browser", you can infer it likely works on mobile browsers unless explicitly stated otherwise.

6. **Present All Relevant Details Clearly**
   - Include **every related detail** (features, options, steps, settings, etc.) from the context.
   - Use clear structure with:
     - **Numbered steps** for processes
     - **Bullet points** for lists or options
     - **Bold headings** for major sections
     - **Emojis** for visual organization (📱 for mobile, 🛠️ for admin, ✅ for confirmation, etc.)
   - Preserve exact **feature names**, **menu paths**, and **option labels** from the context.

7. **Structure by Question Type**
   - **Feature Explanation:** Describe purpose, use cases, menu path, and available options.  
   - **How-To / Process:** Provide complete step-by-step instructions, in order, formatted clearly.  
   - **Configuration / Settings:** List all available fields, choices, and their effects.  
   - **Comparisons:** Use structured tables or bullet comparisons.

8. **Use Authentic Wording**
   - Paraphrase accurately or quote small portions from the transcript.
   - Maintain OnPrintShop terminology exactly as it appears in the context (e.g., "Duplicate Templates," "Master Templates," "Resize Option").

9. **Reasonable Inferences**
   - If the context describes a web-based process accessible through a browser, you can state it works on mobile browsers (unless explicitly stated otherwise).
   - If the context mentions "My Account" and "My Orders" sections, you can describe how to access them on mobile.
   - **Only infer** what is reasonable based on what IS in the context - don't make up features or capabilities.

10. **When Information Is Truly Missing**
   - **Extract and present** any related information that might answer the question, even if not explicitly stated.
   - If specific details are missing, provide what IS available and explain the general process or concept.
   - **ABSOLUTELY FORBIDDEN - NEVER USE THESE PHRASES**:
     - "The context does not specify..."
     - "The context does not mention..."
     - "The context does not include..."
     - "The context does not provide..."
     - "The context does not indicate..."
     - "The context does not contain..."
     - "The context does not state..."
     - "The context does not explain..."
     - "The context does not describe..."
     - "The context does not detail..."
     - Any variation of "The context does not [verb]..."
   - **CRITICAL**: If you catch yourself about to say "The context does not...", STOP and instead:
     - Extract what IS available from the context
     - Explain the general process or framework
     - Say "Based on the available information..." or "The process typically involves..."
     - Say "The context explains..." or "According to the context..."
     - If timing isn't specified, explain the process steps and note that "processing time may vary" or "depends on admin workflow"
     - If exact details aren't given, provide the general framework from context and explain the typical process
     - Focus on what CAN be answered from the context, not what cannot

11. **Do NOT Include**
   - External recommendations (support links, documentation references, etc.)
   - Information completely fabricated or not grounded in context
   - Personal opinions or filler phrases

---

### 🧭 FORMAT SUMMARY

**Answer Format:**
1. Start with a direct, helpful answer to the question.
2. **FIRST: Scan the context for exact feature names** (capitalized terms like "Option Rules," "Additional Option Price," etc.) and **MUST include them** in your answer.
3. **Use specific examples and exact feature names** from the retrieved content - never paraphrase feature names.
4. Provide step-by-step instructions when available, using exact menu paths and button names from context.
5. Organize by sections (using numbered or bulleted lists).
6. Use clear formatting with headings, emojis, and structure.
7. Include all relevant menu paths, terminology, and instructions from context.
8. **Ground every claim in the retrieved text** - avoid generic statements.
9. **When listing features, always use the exact capitalized names** found in the context (e.g., "Option Rules," "Combined Options Price," "Quantity Based Additional Options Price").

Context:
{context}"""),
            ("human", "{question}")
        ])
        
        # Format documents function with feature name highlighting
        def format_docs(docs):
            """Format documents and highlight feature names for better extraction."""
            formatted = []
            for doc in docs:
                content = doc.page_content
                # Add a separator with metadata info to help identify source
                source_info = f"[Source: {doc.metadata.get('video_name', 'Unknown')} - {doc.metadata.get('content_type', 'Unknown')}]"
                formatted.append(f"{source_info}\n{content}")
            return "\n\n---\n\n".join(formatted)
        
        # Store format function for re-ranking
        self.format_docs = format_docs
        
        # Create RAG chain using LCEL (for fallback, but we'll use direct LLM call with re-ranking)
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Store retriever for source documents
        self._retriever = self.retriever
        
        # Get collection info
        try:
            collection = self.vectorstore._collection
            count = collection.count()
        except:
            # Fallback: try a test search
            try:
                test_results = self.retriever.invoke("test query")
                count = len(test_results) if test_results else "unknown"
            except:
                count = "unknown"
        
        print(f"✓ Loaded vector database")
        if count != "unknown":
            print(f"✓ Document chunks available: {count}")
            if count == 0:
                print(f"  Note: If embedding is currently running, documents will be available as they're added")
                print(f"  The server will work with whatever documents are in the database")
        print(f"✓ Ready to answer questions!\n")
    
    def _hybrid_search(self, query: str, keywords: List[str], k: int) -> List:
        """
        Perform hybrid search combining semantic similarity with keyword matching.
        
        Args:
            query: The search query
            keywords: List of keywords extracted from the query
            k: Number of documents to retrieve
            
        Returns:
            List of documents ranked by combined semantic + keyword relevance
        """
        # First, perform semantic search
        semantic_docs = self._retriever.invoke(query)
        
        # If no keywords, just return semantic results
        if not keywords:
            return semantic_docs[:k]
        
        # Score documents by keyword matches
        scored_docs = []
        seen_docs = set()
        
        # First pass: score semantic results with keyword boost
        for doc in semantic_docs:
            doc_id = id(doc)  # Use object id as unique identifier
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            content_lower = doc.page_content.lower()
            metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
            
            # Count keyword matches in content and metadata
            keyword_score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_score += 2  # Content matches are more important
                if keyword in metadata_text:
                    keyword_score += 1  # Metadata matches are less important
            
            # Combine semantic position (earlier = better) with keyword score
            # Documents that appear early in semantic search AND have keywords get highest priority
            combined_score = (len(semantic_docs) - semantic_docs.index(doc)) + (keyword_score * 3)
            scored_docs.append((combined_score, doc))
        
        # If we have fewer than k documents, try to find more with keyword matching
        if len(scored_docs) < k:
            # Perform additional keyword-based search
            # Search for documents containing any of the keywords
            try:
                # Use Chroma's where filter for keyword search if available
                # For now, we'll do a broader semantic search and filter by keywords
                broader_docs = self.vectorstore.similarity_search(query, k=k*2)
                for doc in broader_docs:
                    doc_id = id(doc)
                    if doc_id in seen_docs:
                        continue
                    seen_docs.add(doc_id)
                    
                    content_lower = doc.page_content.lower()
                    metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
                    
                    keyword_score = 0
                    for keyword in keywords:
                        if keyword in content_lower:
                            keyword_score += 2
                        if keyword in metadata_text:
                            keyword_score += 1
                    
                    if keyword_score > 0:
                        # Lower priority than semantic results, but still include
                        combined_score = keyword_score * 2
                        scored_docs.append((combined_score, doc))
            except Exception as e:
                print(f"   Warning: Keyword search expansion failed: {e}")
        
        # Sort by combined score (highest first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k documents
        result_docs = [doc for _, doc in scored_docs[:k]]
        
        # If we still don't have enough, fill with remaining semantic results
        if len(result_docs) < k:
            for doc in semantic_docs:
                if doc not in result_docs:
                    result_docs.append(doc)
                    if len(result_docs) >= k:
                        break
        
        return result_docs[:k]
    
    def _correct_query_with_openai(self, question: str) -> str:
        """
        Use OpenAI to correct typos and improve query clarity.
        This leverages the same LLM we're already using for answers.
        
        Args:
            question: Original question (may contain typos)
            
        Returns:
            Corrected query (or original if no corrections needed)
        """
        try:
            # Use a simple, fast correction prompt
            correction_prompt = f"""Correct any spelling mistakes or typos in this question, but keep the meaning and intent exactly the same. Only correct obvious typos (like "cancle" -> "cancel"). If the question is already correct, return it unchanged.

Question: {question}

Corrected question:"""
            
            # Use the non-streaming LLM instance for quick correction
            if not hasattr(self, 'correction_llm'):
                # Fallback if correction_llm not initialized
                return question
            
            corrected = self.correction_llm.invoke(correction_prompt).content.strip()
            
            # Only use correction if it's different and reasonable
            if corrected and corrected.lower() != question.lower():
                # Check if the correction is similar enough (not a complete rewrite)
                # Simple check: if lengths are very different, it might be wrong
                if abs(len(corrected) - len(question)) < len(question) * 0.5:
                    return corrected
            
            return question
            
        except Exception as e:
            # If correction fails, just return original question
            # Semantic search should still work reasonably well
            print(f"   Warning: Query correction failed: {e}")
            return question
    
    def _enhance_context_with_feature_names(self, context: str) -> str:
        """
        Enhance context by adding a feature name extraction guide at the top.
        This helps the LLM identify and use exact feature names.
        
        Args:
            context: The formatted context string
            
        Returns:
            Enhanced context with feature name extraction instructions
        """
        # Common feature name patterns to look for (based on OnPrintShop)
        feature_patterns = [
            "Option Rules", "Additional Option", "Additional Option Price", 
            "Combined Options Price", "Quantity Based", "Master Options",
            "Duplicate Templates", "Resize Option", "4over Integration",
            "Designer Rules", "Option Label", "Price Calculator"
        ]
        
        # Extract unique capitalized phrases that might be feature names
        import re
        # Find capitalized phrases (2+ words, each starting with capital)
        capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', context)
        # Filter to likely feature names (2-4 words, not common sentences)
        likely_features = [p for p in set(capitalized_phrases) 
                          if 2 <= len(p.split()) <= 4 
                          and p not in ['My Account', 'My Orders', 'The System', 'The Product']]
        
        # Combine with known patterns
        all_features = list(set(feature_patterns + likely_features[:10]))  # Limit to top 10
        
        if all_features:
            feature_guide = f"""
=== ⚠️ CRITICAL: EXACT FEATURE NAMES FOUND IN CONTEXT ⚠️ ===
The following exact feature names appear in the context below. You MUST use these exact names (with exact capitalization) in your answer when discussing related topics:

{', '.join(sorted(all_features))}

**IMPORTANT RULES:**
1. When you mention any of these features, use the EXACT name as shown above
2. Do NOT paraphrase, simplify, or change capitalization
3. If the context discusses a feature, find its exact name from this list and use it
4. Example: If discussing option pricing, use "Additional Option Price" or "Combined Options Price" (exact names from list above), NOT "additional options" or "option pricing feature"

===
"""
            return feature_guide + "\n\n" + context
        
        return context
    
    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sources based on video_id, video_name, and source_file.
        For hybrid mode: keeps both mp4 and txt sources even if they share the same video_name.
        Only removes true duplicates (same video_id, video_name, AND source_file).
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Deduplicated list of sources
        """
        seen = set()
        deduplicated = []
        
        for source in sources:
            # Create a unique key from video_id, video_name, and source_file
            # This ensures mp4 and txt files with the same video_name are kept separately
            source_file = source.get("source_file", "Unknown")
            video_key = (
                source.get("video_id", "Unknown"), 
                source.get("video_name", "Unknown"),
                source_file
            )
            
            if video_key not in seen:
                seen.add(video_key)
                deduplicated.append(source)
        
        return deduplicated
    
    def _should_show_sources(self, question: str, answer: str, sources: List[Dict]) -> bool:
        """
        Determine if sources should be shown.
        Sources are shown if they exist - they were retrieved based on embedding similarity.
        No hardcoded filtering - let embeddings determine relevance.
        
        Args:
            question: User's question
            answer: Generated answer
            sources: List of source dictionaries
            
        Returns:
            True if sources should be shown, False otherwise
        """
        if not sources or len(sources) == 0:
            return False
        
        # Only filter out truly generic system questions (not content questions)
        # These are questions about the assistant itself, not about the content
        system_question_keywords = [
            "what is your job",
            "who are you",
            "what are you",
            "introduce yourself",
            "what is your purpose",
            "what is your role",
            "what do you do"
        ]
        
        question_lower = question.lower().strip()
        
        # Only hide sources for system questions about the assistant itself
        for keyword in system_question_keywords:
            if keyword in question_lower:
                return False
        
        # For all content questions, show sources if they were retrieved
        # Sources were retrieved based on embedding similarity, so they're relevant
        return True
    
    def query_stream(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None):
        """
        Stream query response from the RAG system.
        
        Args:
            question: User's question
            conversation_history: Optional list of previous messages in format [{"role": "user"/"assistant", "content": "..."}]
            
        Yields:
            Dictionary chunks with answer text and optional sources
        """
        print(f"🔍 Searching for relevant content...")
        
        # Start timing
        start_time = time.time()
        timing_info = {
            "query_enhancement": 0,
            "correction": 0,
            "retrieval": 0,
            "keyword_boosting": 0,
            "reranking": 0,
            "context_preparation": 0,
            "llm_generation": 0,
            "total": 0
        }
        
        # Enhance query with conversation context for better retrieval
        enhance_start = time.time()
        enhanced_question = question
        if conversation_history:
            # Get the last user questions to add context
            recent_user_questions = [msg.get("content", "") for msg in conversation_history[-4:] if msg.get("role") == "user"]
            
            if recent_user_questions:
                # Detect vague questions that need strong context
                question_lower = question.lower().strip()
                question_words = question.split()
                
                # Very vague questions: short, pronouns, or generic question words
                is_very_vague = (
                    len(question_words) <= 4 or 
                    any(word in question_lower for word in ["it", "that", "this", "they", "those", "one", "can i", "what", "how", "when", "where", "who", "which"]) or
                    question_lower.startswith(("what ", "how ", "when ", "where ", "who ", "which ", "can ", "will ", "does "))
                )
                
                if is_very_vague and recent_user_questions:
                    # For very vague questions, strongly prioritize the main topic
                    # Get the first question in the conversation (main topic)
                    main_topic = recent_user_questions[0] if recent_user_questions else ""
                    # Extract key terms from main topic (first 10 words) for better matching
                    main_topic_words = main_topic.split()[:10]
                    main_topic_key = " ".join(main_topic_words)
                    # Also get recent context
                    context_summary = " ".join(recent_user_questions[-2:])  # Last 2 user questions
                    # Put main topic key terms multiple times for maximum emphasis
                    enhanced_question = f"{main_topic_key} {main_topic_key} {main_topic_key} {context_summary} {question}".strip()
                    print(f"   Enhanced very vague query with strong context (3x emphasis): '{main_topic_key}'")
                    # Skip correction for very vague questions to preserve repetition
                    skip_correction = True
                elif len(question_words) <= 6:
                    # Moderately vague questions
                    main_topic = recent_user_questions[0] if recent_user_questions else ""
                    context_summary = " ".join(recent_user_questions[-2:])
                    enhanced_question = f"{main_topic} {context_summary} {question}".strip()
                    print(f"   Enhanced vague query with context: '{main_topic}'")
                    skip_correction = False
                else:
                    # For specific questions, combine context with question
                    context_summary = " ".join(recent_user_questions[-2:])  # Last 2 user questions
                    enhanced_question = f"{context_summary} {question}".strip()
                    print(f"   Enhanced query with conversation context")
                    skip_correction = False
            else:
                skip_correction = False
        else:
            skip_correction = False
        
        # Correct query typos using OpenAI (skip for very vague questions and short questions to save time)
        # Skip correction for: very vague questions, short questions (< 15 words), or if question looks correct
        # Increased threshold to 15 words to skip more corrections and speed up responses
        should_correct = not skip_correction and len(question.split()) >= 15
        if skip_correction or not should_correct:
            corrected_question = enhanced_question
            if skip_correction:
                print(f"   Skipping correction to preserve context emphasis for vague question")
            else:
                print(f"   Skipping correction for short/question (faster response)")
        else:
            corrected_question = self._correct_query_with_openai(enhanced_question)
            if corrected_question != enhanced_question:
                print(f"   Query corrected: '{enhanced_question}' -> '{corrected_question}'")
        
        # Hybrid search: Combine semantic search with keyword matching
        # Extract keywords from question for hybrid search
        question_words = re.findall(r'\b\w+\b', corrected_question.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'what', 'how', 'when', 'where', 'who', 'which', 'why'}
        keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        # Get source documents using hybrid search
        # For very vague questions, retrieve more documents to have better candidates
        # For specific questions, use fewer documents for faster retrieval
        if skip_correction and conversation_history:
            # Retrieve more documents for vague questions to increase chances of finding topic-relevant docs
            original_k = self.retriever.search_kwargs.get("k", 15)
            self.retriever.search_kwargs["k"] = 30  # Increased from 25 for better coverage
            source_docs = self._hybrid_search(corrected_question, keywords, k=30)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
        elif len(question.split()) > 12:
            # For long, specific questions, reduce retrieval for speed
            original_k = self.retriever.search_kwargs.get("k", 15)
            self.retriever.search_kwargs["k"] = 10  # Fewer docs for specific questions
            source_docs = self._hybrid_search(corrected_question, keywords, k=10)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
        else:
            source_docs = self._hybrid_search(corrected_question, keywords, k=15)
        
        if not source_docs:
            yield {
                "type": "error",
                "content": "No relevant content found in the video database for this question."
            }
            return
        
        # Debug: Show what was retrieved
        print(f"   Found {len(source_docs)} relevant chunks")
        
        # For very vague questions, boost documents that match conversation topic keywords
        # Only do this for vague questions to save time on specific questions
        if skip_correction and conversation_history and len(source_docs) > 10:
            # Get recent user questions again for keyword extraction
            recent_user_questions = [msg.get("content", "") for msg in conversation_history[-4:] if msg.get("role") == "user"]
            if recent_user_questions:
                main_topic = recent_user_questions[0] if recent_user_questions else ""
                # Extract key terms from main topic (remove common words) - limit to top 5 keywords for speed
                topic_keywords = [word.lower() for word in main_topic.split() 
                                if word.lower() not in ["tell", "me", "about", "the", "a", "an", "is", "are", "what", "how", "when", "where", "who", "which"] 
                                and len(word) > 3][:5]  # Limit to 5 keywords for faster processing
                
                if topic_keywords:
                    # Boost documents that contain topic keywords - only check first 20 docs for speed
                    docs_to_check = source_docs[:20] if len(source_docs) > 20 else source_docs
                    scored_docs = []
                    for doc in docs_to_check:
                        content_lower = doc.page_content.lower()
                        score_boost = sum(1 for keyword in topic_keywords if keyword in content_lower)
                        # Also check metadata
                        metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
                        score_boost += sum(0.5 for keyword in topic_keywords if keyword in metadata_text)
                        scored_docs.append((score_boost, doc))
                    
                    # Sort by boost score (higher = more relevant to conversation topic)
                    scored_docs.sort(key=lambda x: x[0], reverse=True)
                    # Keep boosted docs at top, append rest
                    boosted_docs = [doc for _, doc in scored_docs if _ > 0]
                    remaining_docs = [doc for doc in source_docs if doc not in docs_to_check]
                    source_docs = boosted_docs + [doc for _, doc in scored_docs if _ == 0] + remaining_docs
                    print(f"   Boosted documents matching conversation topic keywords: {topic_keywords[:5]}")
        
        # Store ALL retrieved documents before re-ranking (to find video sources)
        all_retrieved = source_docs.copy()
        
        # Don't filter before re-ranking - use all sources for context
        # We'll filter sources for display later, but use all available info for answering
        # Re-rank documents if re-ranker is available
        # Skip re-ranking more aggressively to save time (they're already well-matched)
        # Only re-rank for vague questions or when we have many documents
        should_rerank = (
            self.reranker and 
            len(source_docs) > FINAL_CHUNKS and
            (skip_correction or len(source_docs) > 20) and  # Only re-rank vague questions or when many docs
            not (len(question.split()) > 15 and not skip_correction)  # Skip for medium+ specific questions
        )
        if should_rerank:
            print(f"   Re-ranking to get top {FINAL_CHUNKS} most relevant chunks...")
            rerank_query = enhanced_question if 'enhanced_question' in locals() else corrected_question
            pairs = [[rerank_query, doc.page_content] for doc in source_docs]
            try:
                import numpy as np
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                # Convert to list and handle NaN
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                # Handle NaN scores - replace with -inf so they sort last
                scores = [float(s) if not (isinstance(s, float) and (s != s or s == float('inf') or s == float('-inf'))) else float('-inf') for s in scores]
                scored_docs = list(zip(scores, source_docs))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                source_docs = [doc for _, doc in scored_docs[:FINAL_CHUNKS]]
                print(f"   Selected top {len(source_docs)} chunks after re-ranking")
            except Exception as e:
                print(f"   ⚠️  Re-ranking failed: {e}, using original order")
        
        # Hybrid approach: Use both video and text sources for context
        # IMPORTANT: Search through ALL retrieved documents, not just re-ranked top 5
        context_start = time.time()
        # Separate video and text sources from ALL retrieved documents
        video_docs = []
        text_docs = []
        for doc in all_retrieved:
            source_file = doc.metadata.get("source_file", "Unknown")
            mapped_filename = self._get_original_video_filename(source_file)
            is_text_file = mapped_filename.endswith('.txt')
            if is_text_file:
                text_docs.append(doc)
            else:
                video_docs.append(doc)
        
        print(f"   Found {len(video_docs)} video and {len(text_docs)} text chunks in retrieved documents")
        
        # Hybrid approach: Mix video and text chunks together for context
        # This ensures both mp4 and txt files are used in answers
        context_docs = []
        
        if video_docs and text_docs:
            # Mix both types: take roughly equal amounts from each, prioritizing relevance (already re-ranked)
            video_count = min(len(video_docs), FINAL_CHUNKS // 2 + 1)  # At least half, rounded up
            text_count = min(len(text_docs), FINAL_CHUNKS - video_count)  # Fill remaining space
            
            context_docs = video_docs[:video_count] + text_docs[:text_count]
            
            # If we still have space, add more from whichever has more relevant chunks
            if len(context_docs) < FINAL_CHUNKS:
                remaining = FINAL_CHUNKS - len(context_docs)
                if len(video_docs) > video_count:
                    context_docs.extend(video_docs[video_count:video_count + remaining])
                elif len(text_docs) > text_count:
                    context_docs.extend(text_docs[text_count:text_count + remaining])
            
            print(f"   Using {len([d for d in context_docs if not self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} video and {len([d for d in context_docs if self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} text chunks for context (hybrid)")
        elif video_docs:
            # Only video sources available
            context_docs = video_docs[:FINAL_CHUNKS]
            print(f"   Using {len(context_docs)} video chunks for context (no text sources found)")
        elif text_docs:
            # Only text sources available
            context_docs = text_docs[:FINAL_CHUNKS]
            print(f"   Using {len(context_docs)} text chunks for context (no video sources found)")
        else:
            # Fallback: use re-ranked or original top documents
            context_docs = source_docs[:FINAL_CHUNKS]
            print(f"   No video or text sources found, using top {len(context_docs)} documents")
        
        # Format context - hybrid mix of video and text sources
        # Pre-process context to highlight feature names for better extraction
        context_text = self._enhance_context_with_feature_names(self.format_docs(context_docs))
        
        # Prepare sources from context_docs (which already prioritizes video sources)
        # Include content preview to help identify which file contributed which information
        sources = []
        for doc in context_docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            mapped_filename = self._get_original_video_filename(source_file)
            is_text_file = mapped_filename.endswith('.txt')
            
            # Get a meaningful preview that shows what this chunk contains
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            # Extract first sentence or key phrase for better identification
            first_sentence = content_preview.split('.')[0] if '.' in content_preview else content_preview.split('\n')[0]
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:150] + "..."
            
            sources.append({
                "video_name": doc.metadata.get("video_name", "Unknown"),
                "video_id": doc.metadata.get("video_id", "Unknown"),
                "content_type": doc.metadata.get("content_type", "Unknown"),
                "source_file": source_file,
                "is_text_file": is_text_file,
                "content_preview": content_preview,
                "key_phrase": first_sentence  # Key phrase to identify what this source contributes
            })
        
        # Hybrid approach: show both video and text sources together
        video_sources = [s for s in sources if not s.get("is_text_file", False)]
        text_sources = [s for s in sources if s.get("is_text_file", False)]
        
        # Keep both video and text sources for hybrid display
        # This allows showing both mp4 and txt references together
        print(f"   Displaying {len(video_sources)} video and {len(text_sources)} text sources (hybrid mode)")
        
        # Deduplicate sources (keeps both video and text if they're from different files)
        sources = self._deduplicate_sources(sources)
        timing_info["context_preparation"] = time.time() - context_start
        
        # Build messages with conversation history
        llm_start = time.time()
        # First, format the original prompt template to get the system message
        formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
        system_message_content = formatted_system_messages[0].content if formatted_system_messages else ""
        
        # Build messages list
        messages = []
        
        # Add system message with context
        messages.append(("system", system_message_content))
        
        # Add conversation history (last 6 messages to reduce tokens and speed up)
        if conversation_history:
            # Keep only recent history to avoid token limits and reduce processing time
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(("human", content))
                elif role == "assistant":
                    messages.append(("ai", content))
        
        # Add current question
        messages.append(("human", question))
        
        # Create prompt from messages
        from langchain_core.prompts import ChatPromptTemplate
        conversation_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Stream the response
        full_answer = ""
        formatted_messages = conversation_prompt.format_messages()
        for chunk in self.llm.stream(formatted_messages):
            if chunk.content:
                full_answer += chunk.content
                yield {
                    "type": "content",
                    "content": chunk.content
                }
        timing_info["llm_generation"] = time.time() - llm_start
        
        # Only send sources if they're relevant
        if self._should_show_sources(question, full_answer, sources):
            yield {
                "type": "sources",
                "sources": sources
            }
        
        # Calculate total time
        timing_info["total"] = time.time() - start_time
        
        # Print concise timing summary
        print(f"⏱️  Total: {timing_info['total']:.2f}s (Retrieval: {timing_info['retrieval']:.2f}s, LLM: {timing_info['llm_generation']:.2f}s)")
        
        # Send timing info with done message
        yield {
            "type": "done",
            "timing": timing_info
        }
    
    def query(self, question: str, return_sources: bool = True, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            return_sources: Whether to return source documents
            conversation_history: Optional list of previous messages in format [{"role": "user"/"assistant", "content": "..."}]
            
        Returns:
            Dictionary with answer and optional sources
        """
        print(f"🔍 Searching for relevant content...")
        
        # Enhance query with conversation context for better retrieval
        enhanced_question = question
        if conversation_history:
            # Get the last user questions to add context
            recent_user_questions = [msg.get("content", "") for msg in conversation_history[-4:] if msg.get("role") == "user"]
            
            if recent_user_questions:
                # Detect vague questions that need strong context
                question_lower = question.lower().strip()
                question_words = question.split()
                
                # Very vague questions: short, pronouns, or generic question words
                is_very_vague = (
                    len(question_words) <= 4 or 
                    any(word in question_lower for word in ["it", "that", "this", "they", "those", "one", "can i", "what", "how", "when", "where", "who", "which"]) or
                    question_lower.startswith(("what ", "how ", "when ", "where ", "who ", "which ", "can ", "will ", "does "))
                )
                
                if is_very_vague and recent_user_questions:
                    # For very vague questions, strongly prioritize the main topic
                    # Get the first question in the conversation (main topic)
                    main_topic = recent_user_questions[0] if recent_user_questions else ""
                    # Extract key terms from main topic (first 10 words) for better matching
                    main_topic_words = main_topic.split()[:10]
                    main_topic_key = " ".join(main_topic_words)
                    # Also get recent context
                    context_summary = " ".join(recent_user_questions[-2:])  # Last 2 user questions
                    # Put main topic key terms multiple times for maximum emphasis
                    enhanced_question = f"{main_topic_key} {main_topic_key} {main_topic_key} {context_summary} {question}".strip()
                    print(f"   Enhanced very vague query with strong context (3x emphasis): '{main_topic_key}'")
                    # Skip correction for very vague questions to preserve repetition
                    skip_correction = True
                elif len(question_words) <= 6:
                    # Moderately vague questions
                    main_topic = recent_user_questions[0] if recent_user_questions else ""
                    context_summary = " ".join(recent_user_questions[-2:])
                    enhanced_question = f"{main_topic} {context_summary} {question}".strip()
                    print(f"   Enhanced vague query with context: '{main_topic}'")
                    skip_correction = False
                else:
                    # For specific questions, combine context with question
                    context_summary = " ".join(recent_user_questions[-2:])  # Last 2 user questions
                    enhanced_question = f"{context_summary} {question}".strip()
                    print(f"   Enhanced query with conversation context")
                    skip_correction = False
            else:
                skip_correction = False
        else:
            skip_correction = False
        
        # Correct query typos using OpenAI (skip for very vague questions and short questions to save time)
        # Skip correction for: very vague questions, short questions (< 15 words), or if question looks correct
        # Increased threshold to 15 words to skip more corrections and speed up responses
        should_correct = not skip_correction and len(question.split()) >= 15
        if skip_correction or not should_correct:
            corrected_question = enhanced_question
            if skip_correction:
                print(f"   Skipping correction to preserve context emphasis for vague question")
            else:
                print(f"   Skipping correction for short/question (faster response)")
        else:
            corrected_question = self._correct_query_with_openai(enhanced_question)
            if corrected_question != enhanced_question:
                print(f"   Query corrected: '{enhanced_question}' -> '{corrected_question}'")
        
        # Hybrid search: Combine semantic search with keyword matching
        # Extract keywords from question for hybrid search
        question_words = re.findall(r'\b\w+\b', corrected_question.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'what', 'how', 'when', 'where', 'who', 'which', 'why'}
        keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        # Get source documents using hybrid search
        # For very vague questions, retrieve more documents to have better candidates
        # For specific questions, use fewer documents for faster retrieval
        if skip_correction and conversation_history:
            # Retrieve more documents for vague questions to increase chances of finding topic-relevant docs
            original_k = self.retriever.search_kwargs.get("k", 15)
            self.retriever.search_kwargs["k"] = 30  # Increased from 25 for better coverage
            source_docs = self._hybrid_search(corrected_question, keywords, k=30)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
        elif len(question.split()) > 12:
            # For long, specific questions, reduce retrieval for speed
            original_k = self.retriever.search_kwargs.get("k", 15)
            self.retriever.search_kwargs["k"] = 10  # Fewer docs for specific questions
            source_docs = self._hybrid_search(corrected_question, keywords, k=10)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
        else:
            source_docs = self._hybrid_search(corrected_question, keywords, k=15)
        
        if not source_docs:
            return {
                "answer": "No relevant content found in the video database for this question.",
                "question": question,
                "sources": []
            }
        
        # Debug: Show what was retrieved
        print(f"   Found {len(source_docs)} relevant chunks")
        
        # For very vague questions, boost documents that match conversation topic keywords
        # Only do this for vague questions to save time on specific questions
        if skip_correction and conversation_history and len(source_docs) > 10:
            # Get recent user questions again for keyword extraction
            recent_user_questions = [msg.get("content", "") for msg in conversation_history[-4:] if msg.get("role") == "user"]
            if recent_user_questions:
                main_topic = recent_user_questions[0] if recent_user_questions else ""
                # Extract key terms from main topic (remove common words) - limit to top 5 keywords for speed
                topic_keywords = [word.lower() for word in main_topic.split() 
                                if word.lower() not in ["tell", "me", "about", "the", "a", "an", "is", "are", "what", "how", "when", "where", "who", "which"] 
                                and len(word) > 3][:5]  # Limit to 5 keywords for faster processing
                
                if topic_keywords:
                    # Boost documents that contain topic keywords - only check first 20 docs for speed
                    docs_to_check = source_docs[:20] if len(source_docs) > 20 else source_docs
                    scored_docs = []
                    for doc in docs_to_check:
                        content_lower = doc.page_content.lower()
                        score_boost = sum(1 for keyword in topic_keywords if keyword in content_lower)
                        # Also check metadata
                        metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
                        score_boost += sum(0.5 for keyword in topic_keywords if keyword in metadata_text)
                        scored_docs.append((score_boost, doc))
                    
                    # Sort by boost score (higher = more relevant to conversation topic)
                    scored_docs.sort(key=lambda x: x[0], reverse=True)
                    # Keep boosted docs at top, append rest
                    boosted_docs = [doc for _, doc in scored_docs if _ > 0]
                    remaining_docs = [doc for doc in source_docs if doc not in docs_to_check]
                    source_docs = boosted_docs + [doc for _, doc in scored_docs if _ == 0] + remaining_docs
                    print(f"   Boosted documents matching conversation topic keywords: {topic_keywords[:5]}")
        
        # Store ALL retrieved documents before re-ranking (to find video sources)
        all_retrieved = source_docs.copy()
        
        # Don't filter before re-ranking - use all sources for context
        # We'll filter sources for display later, but use all available info for answering
        # Re-rank documents if re-ranker is available
        # Skip re-ranking more aggressively to save time (they're already well-matched)
        # Only re-rank for vague questions or when we have many documents
        should_rerank = (
            self.reranker and 
            len(source_docs) > FINAL_CHUNKS and
            (skip_correction or len(source_docs) > 20) and  # Only re-rank vague questions or when many docs
            not (len(question.split()) > 15 and not skip_correction)  # Skip for medium+ specific questions
        )
        if should_rerank:
            print(f"   Re-ranking to get top {FINAL_CHUNKS} most relevant chunks...")
            rerank_query = enhanced_question if 'enhanced_question' in locals() else corrected_question
            pairs = [[rerank_query, doc.page_content] for doc in source_docs]
            try:
                import numpy as np
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                # Convert to list and handle NaN
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                # Handle NaN scores - replace with -inf so they sort last
                scores = [float(s) if not (isinstance(s, float) and (s != s or s == float('inf') or s == float('-inf'))) else float('-inf') for s in scores]
                scored_docs = list(zip(scores, source_docs))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                source_docs = [doc for _, doc in scored_docs[:FINAL_CHUNKS]]
                print(f"   Selected top {len(source_docs)} chunks after re-ranking")
            except Exception as e:
                print(f"   ⚠️  Re-ranking failed: {e}, using original order")
        
        # Hybrid approach: Use both video and text sources for context
        # IMPORTANT: Search through ALL retrieved documents, not just re-ranked top 5
        context_start = time.time()
        # Separate video and text sources from ALL retrieved documents
        video_docs = []
        text_docs = []
        for doc in all_retrieved:
            source_file = doc.metadata.get("source_file", "Unknown")
            mapped_filename = self._get_original_video_filename(source_file)
            is_text_file = mapped_filename.endswith('.txt')
            if is_text_file:
                text_docs.append(doc)
            else:
                video_docs.append(doc)
        
        print(f"   Found {len(video_docs)} video and {len(text_docs)} text chunks in retrieved documents")
        
        # Hybrid approach: Mix video and text chunks together for context
        # This ensures both mp4 and txt files are used in answers
        context_docs = []
        
        if video_docs and text_docs:
            # Mix both types: take roughly equal amounts from each, prioritizing relevance (already re-ranked)
            video_count = min(len(video_docs), FINAL_CHUNKS // 2 + 1)  # At least half, rounded up
            text_count = min(len(text_docs), FINAL_CHUNKS - video_count)  # Fill remaining space
            
            context_docs = video_docs[:video_count] + text_docs[:text_count]
            
            # If we still have space, add more from whichever has more relevant chunks
            if len(context_docs) < FINAL_CHUNKS:
                remaining = FINAL_CHUNKS - len(context_docs)
                if len(video_docs) > video_count:
                    context_docs.extend(video_docs[video_count:video_count + remaining])
                elif len(text_docs) > text_count:
                    context_docs.extend(text_docs[text_count:text_count + remaining])
            
            print(f"   Using {len([d for d in context_docs if not self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} video and {len([d for d in context_docs if self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} text chunks for context (hybrid)")
        elif video_docs:
            # Only video sources available
            context_docs = video_docs[:FINAL_CHUNKS]
            print(f"   Using {len(context_docs)} video chunks for context (no text sources found)")
        elif text_docs:
            # Only text sources available
            context_docs = text_docs[:FINAL_CHUNKS]
            print(f"   Using {len(context_docs)} text chunks for context (no video sources found)")
        else:
            # Fallback: use re-ranked or original top documents
            context_docs = source_docs[:FINAL_CHUNKS]
            print(f"   No video or text sources found, using top {len(context_docs)} documents")
        
        # Format context - hybrid mix of video and text sources
        # Pre-process context to highlight feature names for better extraction
        context_text = self._enhance_context_with_feature_names(self.format_docs(context_docs))
        
        # Build messages with conversation history
        # First, format the original prompt template to get the system message
        formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
        system_message_content = formatted_system_messages[0].content if formatted_system_messages else ""
        
        # Build messages list
        messages = []
        
        # Add system message with context
        messages.append(("system", system_message_content))
        
        # Add conversation history (last 6 messages to reduce tokens and speed up)
        if conversation_history:
            # Keep only recent history to avoid token limits and reduce processing time
            recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(("human", content))
                elif role == "assistant":
                    messages.append(("ai", content))
        
        # Add current question
        messages.append(("human", question))
        
        # Create prompt from messages
        from langchain_core.prompts import ChatPromptTemplate
        conversation_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Get answer using the selected context with conversation history
        answer = self.llm.invoke(conversation_prompt.format_messages()).content
        
        response = {
            "answer": answer,
            "question": question
        }
        
        if return_sources:
            # Prepare sources from context_docs (which already prioritizes video sources)
            # Include content preview to help identify which file contributed which information
            sources = []
            for doc in context_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                mapped_filename = self._get_original_video_filename(source_file)
                is_text_file = mapped_filename.endswith('.txt')
                
                # Get a meaningful preview that shows what this chunk contains
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                # Extract first sentence or key phrase for better identification
                first_sentence = content_preview.split('.')[0] if '.' in content_preview else content_preview.split('\n')[0]
                if len(first_sentence) > 150:
                    first_sentence = first_sentence[:150] + "..."
                
                sources.append({
                    "video_name": doc.metadata.get("video_name", "Unknown"),
                    "video_id": doc.metadata.get("video_id", "Unknown"),
                    "content_type": doc.metadata.get("content_type", "Unknown"),
                    "source_file": source_file,
                    "is_text_file": is_text_file,
                    "content_preview": content_preview,
                    "key_phrase": first_sentence  # Key phrase to identify what this source contributes
                })
            
            # Hybrid approach: show both video and text sources together
            video_sources = [s for s in sources if not s.get("is_text_file", False)]
            text_sources = [s for s in sources if s.get("is_text_file", False)]
            
            # Keep both video and text sources for hybrid display
            # This allows showing both mp4 and txt references together
            print(f"   Displaying {len(video_sources)} video and {len(text_sources)} text sources (hybrid mode)")
            
            # Deduplicate sources (keeps both video and text if they're from different files)
            sources = self._deduplicate_sources(sources)
            
            # Only include sources if they're relevant
            if self._should_show_sources(question, answer, sources):
                response["sources"] = sources
            else:
                response["sources"] = []
        
        return response
    
    def format_response(self, response: Dict) -> str:
        """Format the response for display."""
        output = []
        output.append("=" * 70)
        output.append("ANSWER")
        output.append("=" * 70)
        output.append(response["answer"])
        output.append("")
        
        sources = response.get("sources", [])
        if sources:
            output.append("=" * 70)
            output.append(f"REFERENCES ({len(sources)} sources)")
            output.append("=" * 70)
            
            # Group sources by video to avoid duplicates, but preserve key phrases
            video_sources = {}
            for source in sources:
                video_name = source['video_name']
                if video_name not in video_sources:
                    video_sources[video_name] = {
                        'video_name': video_name,
                        'content_types': set(),
                        'source_file': source['source_file'],
                        'key_phrase': source.get('key_phrase', '')  # Store key phrase for display
                    }
                video_sources[video_name]['content_types'].add(source['content_type'])
                # Update key phrase if this source has a better one (longer/more descriptive)
                if source.get('key_phrase') and len(source.get('key_phrase', '')) > len(video_sources[video_name].get('key_phrase', '')):
                    video_sources[video_name]['key_phrase'] = source.get('key_phrase', '')
            
            # Display grouped sources with key phrases to show what each file contributed
            for i, (video_name, info) in enumerate(video_sources.items(), 1):
                content_types = sorted(info['content_types'])
                json_filename = info['source_file']
                # Get original video filename with extension
                original_video_file = self._get_original_video_filename(json_filename)
                # Show full video name with original file extension
                output.append(f"\n[{i}] {video_name}")
                output.append(f"    └─ File: {original_video_file}")
                # Only show content types if there are multiple or if it's not just transcript
                if len(content_types) > 1 or (len(content_types) == 1 and content_types[0] != 'transcript'):
                    # Format content types more clearly
                    type_labels = {
                        'transcript': 'Transcript',
                        'keywords': 'Keywords',
                        'labels': 'Labels',
                        'summary': 'Summary'
                    }
                    formatted_types = [type_labels.get(ct, ct.title()) for ct in content_types]
                    output.append(f"    └─ Content: {', '.join(formatted_types)}")
                
                # Show key phrase from this source to indicate what it contributed
                if 'key_phrase' in info and info['key_phrase']:
                    output.append(f"    └─ Key Info: {info['key_phrase']}")
        else:
            output.append("=" * 70)
            output.append("REFERENCES")
            output.append("=" * 70)
            output.append("No sources retrieved")
        
        output.append("=" * 70)
        return "\n".join(output)


def interactive_query():
    """Interactive query interface."""
    print("=" * 70)
    print("VIDEO INSIGHTS RAG QUERY SYSTEM")
    print("=" * 70)
    print(f"Model: {OPENAI_MODEL}")
    print("Ask questions about your video content!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 70)
    print()
    
    rag = VideoRAGQuery()
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        try:
            response = rag.query(question)
            print("\n" + rag.format_response(response) + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def single_query(question: str):
    """Query once and return result."""
    rag = VideoRAGQuery()
    response = rag.query(question)
    print(rag.format_response(response))


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        # Single query from command line
        question = " ".join(sys.argv[1:])
        single_query(question)
    else:
        # Interactive mode
        interactive_query()


if __name__ == "__main__":
    main()

