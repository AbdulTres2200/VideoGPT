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
OPENAI_MODEL = "gpt-4o"  # Faster model for better response times
TEMPERATURE = 0.0  # Lower temperature for more consistent, factual answers

# Re-ranking Configuration
USE_RERANKING = True  # Enable re-ranking for better relevance
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast and effective re-ranker
FINAL_CHUNKS = 15  # Increased to ensure comprehensive, detailed answers with all relevant information (including specific field details)


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
            
            # CRITICAL: Always try to get existing collection first, even if list_collections() didn't find it
            # ChromaDB sometimes doesn't list collections properly, but get_collection() works
            collection_found_with_docs = False
            existing_col_id = None
            try:
                # Try to get collection directly - this works even if list_collections() fails
                existing_col = client.get_collection(collection_name)
                direct_count = existing_col.count()
                existing_col_id = existing_col.id
                
                if direct_count > 0:
                    print(f"   ✓ Found existing collection '{collection_name}' with {direct_count} documents (direct access)")
                    print(f"   DEBUG: Collection ID: {existing_col_id}")
                    collection_found_with_docs = True
                else:
                    print(f"   Collection '{collection_name}' exists but is empty")
                    # If collection is empty, we might want to delete it to avoid confusion
                    # But for now, we'll keep it and let embedding add to it
            except Exception as get_err:
                # Collection doesn't exist - that's okay, we'll create it
                print(f"   Collection '{collection_name}' does not exist (will create if needed)")
            
            # Initialize Chroma - CRITICAL: Use existing collection if it has documents
            # If we found a collection with documents, we MUST use it, not create a new one
            self.vectorstore = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # CRITICAL FIX: If we found an existing collection with documents, verify we're using it
            # If LangChain created a new empty collection, delete it and force use of existing one
            if collection_found_with_docs and existing_col_id is not None:
                try:
                    actual_col_id = self.vectorstore._collection.id
                    actual_count = self.vectorstore._collection.count()
                    
                    # Check if LangChain created a different (empty) collection
                    if str(actual_col_id) != str(existing_col_id):
                        print(f"   ⚠️  WARNING: LangChain created a new collection instead of using existing one!")
                        print(f"      Existing collection ID: {existing_col_id} ({direct_count} docs)")
                        print(f"      New collection ID: {actual_col_id} ({actual_count} docs)")
                        
                        # Delete the new empty collection
                        if actual_count == 0:
                            print(f"      Deleting empty collection that was just created...")
                            try:
                                client.delete_collection(actual_col_id)
                                print(f"      ✓ Deleted empty collection")
                            except Exception as del_err:
                                print(f"      Could not delete empty collection: {del_err}")
                        
                        # Re-initialize - should now use existing collection
                        print(f"      Re-initializing to use existing collection...")
                        del self.vectorstore
                        self.vectorstore = Chroma(
                            persist_directory=VECTOR_DB_DIR,
                            embedding_function=embeddings,
                            collection_name=collection_name
                        )
                        
                        # Verify we're now using the correct collection
                        new_actual_id = self.vectorstore._collection.id
                        new_count = self.vectorstore._collection.count()
                        
                        if str(new_actual_id) == str(existing_col_id) and new_count == direct_count:
                            print(f"      ✓ SUCCESS: Now using existing collection with {new_count} documents")
                        else:
                            print(f"      ⚠️  Still not using correct collection")
                            print(f"         Current ID: {new_actual_id}, Expected: {existing_col_id}")
                            print(f"         Current count: {new_count}, Expected: {direct_count}")
                    elif actual_count == direct_count:
                        print(f"   ✓ Using existing collection with {actual_count} documents")
                    else:
                        print(f"   ⚠️  Using correct collection ID but count mismatch: {actual_count} vs {direct_count}")
                except Exception as verify_err:
                    print(f"   ⚠️  Could not verify collection: {verify_err}")
            
            # Verify we're using the right collection
            try:
                actual_count = self.vectorstore._collection.count()
                
                if collection_found_with_docs:
                    if actual_count == direct_count:
                        print(f"   ✓ Successfully using existing collection with {actual_count} documents")
                    elif actual_count == 0:
                        print(f"   ⚠️  WARNING: LangChain wrapper shows 0, but direct access shows {direct_count}")
                        print(f"   This may be a collection ID mismatch - trying to fix...")
                        # Try to force use the existing collection by getting it directly
                        # Re-initialize with explicit collection reference
                        try:
                            # Get the collection ID from direct access
                            existing_col = client.get_collection(collection_name)
                            # Delete the empty collection LangChain created
                            try:
                                empty_col = self.vectorstore._collection
                                if empty_col.count() == 0:
                                    # Don't delete - just re-initialize to use the correct one
                                    pass
                            except:
                                pass
                            # Re-initialize Chroma - it should now find the existing collection
                            self.vectorstore = Chroma(
                                persist_directory=VECTOR_DB_DIR,
                                embedding_function=embeddings,
                                collection_name=collection_name
                            )
                            actual_count = self.vectorstore._collection.count()
                            if actual_count > 0:
                                print(f"   ✓ Fixed: Now using collection with {actual_count} documents")
                            else:
                                print(f"   ⚠️  Still showing 0 - collection may have different ID")
                        except Exception as fix_err:
                            print(f"   ⚠️  Could not fix collection access: {fix_err}")
                    else:
                        print(f"   ⚠️  Count mismatch: direct={direct_count}, wrapper={actual_count}")
                else:
                    if actual_count > 0:
                        print(f"   ✓ Collection has {actual_count} documents")
                    else:
                        print(f"   ✓ Collection initialized (empty - ready for embeddings)")
            except Exception as verify_err:
                print(f"   ⚠️  Could not verify collection count: {verify_err}")
            
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
        
        # Create retriever - retrieve fewer but more relevant chunks
        # Focus on quality over quantity - get the most relevant source(s) first
        initial_k = 15  # Reduced to focus on most relevant sources
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": initial_k}  # Retrieve fewer but more relevant chunks
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

**🚨 CRITICAL INSTRUCTION: EXTRACT EVERY SINGLE DETAIL**
Your goal is to **extract, organize, and present** ALL relevant information from the context to answer the user's question completely and helpfully.
**DO NOT summarize. DO NOT skip details. DO NOT combine steps.**
**READ THE ENTIRE CONTEXT CAREFULLY and extract EVERY field, button, option, step, and detail mentioned.**
**🚨 DO NOT OUTPUT TEMPLATE SYNTAX: Never output template syntax like percent-sign block patterns, curly brace patterns, or double curly brace patterns. Output only plain text answers.**
**🚨 SPECIAL ATTENTION TO REVIEW STEPS: If the context mentions a review or confirmation step, you MUST break it down into ALL reviewable elements separately. Scan for phrases like "you can see", "you can edit", "you can manage", "you can change", "activate", "button" - each indicates a separate reviewable element that MUST be listed individually.**
**🚨 MULTIPLE SOURCES: If the context contains information from multiple sources (indicated by [Source: ...] markers), you MUST:**
- **COMBINE information from ALL sources to create a COMPLETE answer**
- **Extract ALL details from EVERY source - do not skip any fields, buttons, steps, or options from ANY source**
- **If ANY source mentions "artwork name", "quantity", "product weight", "production days", "upload artwork", "price calculator", "Apply All", "Apply Options Only" - you MUST include ALL of these in your answer**
- **If ANY source has a detailed review step, extract EVERY element separately (order amount, product details with Edit button, shipping address, billing address, payment request button, comments)**
- **COMBINE details from all sources: If one source has "artwork upload" and another doesn't, include "artwork upload"**
- **COMBINE details from all sources: If one source has "Online Price Calculator" and another doesn't, include "Online Price Calculator"**
- **COMBINE details from all sources: If one source has detailed product configuration steps and another has simplified steps, include BOTH sets of details**
- **The goal is to provide the MOST COMPLETE answer possible by combining ALL information from ALL sources**
- **DO NOT prioritize one source over another - extract and combine ALL details from ALL sources**
Typo in a question should not affect the answer. You should fix the typo internally and answer the question based on the corrected question.

**🚨 CRITICAL RULE: ANSWER ONLY FROM KNOWLEDGE BASE**
- **NEVER provide generic information, general knowledge, or information not found in the provided context.**
- **ONLY use information that appears in the context provided to you.**
- **If the context does not contain relevant information about the question, clearly state that the information is not available in the knowledge base.**
- **DO NOT fill gaps with general knowledge or common practices.**
- **DO NOT provide generic explanations, definitions, or examples that are not in the context.**
- **If asked about something not in the context, say: "The provided knowledge base does not contain information about [topic]. Please check other resources or contact support."**
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

5. **Extract and Present ALL Information - COMPLETENESS IS CRITICAL**
   - **MANDATORY: Extract EVERY step, field, option, button, toggle, and detail mentioned in the context.**
   - **READ THE ENTIRE CONTEXT WORD BY WORD** - do not skim or summarize.
   - **DO NOT summarize or skip steps** - list every single action, field, and option that appears in the workflow.
   - **SCAN THE CONTEXT FOR ALL MENTIONED ITEMS** and include them in your answer:
     - Every field name mentioned (e.g., "artwork name", "quantity", "product weight", "production days", etc.)
     - Every button name mentioned (e.g., "Apply All", "Apply Options Only", "Save & Add Another Product", etc.)
     - Every tool or feature mentioned (e.g., "Online Price Calculator", "upload file", etc.)
     - Every review or confirmation step mentioned
     - Every toggle, setting, or option mentioned
     - Any button name, field name, option, or action mentioned in the context
   - **Actively search** the context for any information related to the question, even if not explicitly stated.
   - If the context describes a process (e.g., "go to My Account → My Orders"), you can reasonably infer it works on mobile if the platform is web-based and mobile-responsive.
   - **Never start with** "The context does not mention..." or "However, it does not specify..." - instead, extract what IS available and present it helpfully.
   - If a process is described for "desktop" or "web browser", you can infer it likely works on mobile browsers unless explicitly stated otherwise.
   - **For workflow/process questions: Extract EVERY step in the exact order shown, including:**
     - Every button clicked
     - Every field filled
     - Every option selected
     - Every toggle switched
     - Every dropdown opened
     - Every file uploaded
     - Every confirmation dialog
     - Every review screen (CRITICAL: If context mentions a review step, break down ALL elements that can be reviewed. Scan the context for phrases like "you can see", "you can edit", "you can manage", "you can change", "activate", "button", "toggle" - these indicate reviewable elements. List EVERY element separately with its specific action/button/method. DO NOT combine them into a generic "review" statement)
     - Every notification or email sent

6. **Present All Relevant Details Clearly - NO SUMMARIES**
   - **CRITICAL: Include EVERY related detail** (features, options, steps, settings, fields, buttons, toggles, etc.) from the context.
   - **DO NOT combine steps** - if the context shows 10 steps, list all 10 steps separately.
   - **DO NOT say "you can add multiple products"** - instead, explain exactly HOW to add multiple products (e.g., "After adding the first product, click 'Save & Add Another Product' button").
   - Use clear structure with:
     - **Numbered steps** for processes (number EVERY step, don't skip any)
     - **Bullet points** for lists or options (list EVERY option, don't say "various options")
     - **Bold headings** for major sections
     - **Emojis** for visual organization (📱 for mobile, 🛠️ for admin, ✅ for confirmation, etc.)
   - Preserve exact **feature names**, **menu paths**, **button labels**, **field names**, and **option labels** from the context.
   - **For any configuration step: List ALL fields mentioned** (whatever fields the context specifies - list them all explicitly)
   - **For any tool or feature: Include ALL options and methods** (whatever options the context specifies - list them all explicitly)

7. **Structure by Question Type - BE EXHAUSTIVE FOR ALL QUESTIONS**
   - **CRITICAL: ALL questions must be answered in a VERY DETAILED manner** - extract and list every relevant detail from the context.
   - **Feature Explanation:** Describe purpose, use cases, menu path, and **ALL available options** (list every option, don't say "various options").  
   - **How-To / Process:** Provide **COMPLETE step-by-step instructions in order** - **DO NOT skip or combine steps**. List every single action, field, button, and option. If the context shows a review step, include it. If it shows a toggle, include it.  
   - **For ALL questions, you MUST extract and list EXPLICITLY - NO EXCEPTIONS:**
     - **Every field that needs to be filled** - if context lists multiple fields (e.g., "field1, field2, field3") → you MUST list all fields separately, don't combine them (e.g., "Add field1, field2, and field3" - list all separately)
     - **Every button or action** - if context mentions multiple buttons or actions → you MUST mention ALL of them with explanations (e.g., if context says "button A or button B" → mention both buttons and what each does)
     - **Every option or setting** - list each one explicitly, don't say "various options" or "necessary details"
     - **Every tool mentioned** - explain how to use it step-by-step with ALL options (if context mentions multiple methods or options for a tool, list ALL of them - e.g., if context says "either method A OR method B" → explain both methods)
     - **Every review or confirmation step** - **CRITICAL: This is MANDATORY** - explicitly mention the review step before confirmation and BREAK DOWN ALL elements that can be reviewed/edited. If the context mentions ANY of these, you MUST list them ALL:
       * Order amount or total (if mentioned, include that it can be seen and manually changed)
       * Product details (if mentioned, include that they can be seen and edited using Edit button)
       * Shipping address (if mentioned, include how to manage it - e.g., "click Manage Addresses")
       * Billing address (if mentioned, include how to manage it - e.g., "click Manage Addresses")
       * Payment request button/toggle (if mentioned, include activation/deactivation instructions)
       * Additional comments (if mentioned, include where/how to add them)
       * Any other reviewable element mentioned in context
       **DO NOT** just say "Review the order details" - you MUST break it down into specific elements with instructions for each
     - **Every file upload action** - explicitly mention the upload step as a separate step (whatever file type or upload action the context mentions)
     - **Every notification or email sent** - mention what notifications are sent
     - **Every menu path, feature name, and exact terminology** - use exact names from context
     - **CRITICAL: If the context lists multiple items in one sentence (e.g., "quantity, product weight, production days"), you MUST list them all separately in your answer, don't combine them**
     - **CRITICAL: If context says "either X or Y" → mention BOTH options (e.g., "either manually or use price calculator" → mention both methods)**
   - **Configuration / Settings:** List **ALL available fields, choices, and their effects** - don't summarize, list everything.  
   - **Comparisons:** Use structured tables or bullet comparisons with all details.

8. **Use Authentic Wording - LIST EVERYTHING EXPLICITLY**
   - Paraphrase accurately or quote small portions from the transcript.
   - Maintain OnPrintShop terminology exactly as it appears in the context (e.g., "Duplicate Templates," "Master Templates," "Resize Option").
   - **CRITICAL: When the context lists multiple items, you MUST list them all explicitly:**
     - ❌ WRONG: "Add details like [item1] and other information" (vague, combines items)
     - ✅ CORRECT: "Add [item1], [item2], [item3], and [item4]" (lists all items separately)
     - ❌ WRONG: "Use the [tool] to apply details" (vague, doesn't explain options)
     - ✅ CORRECT: "Click [Tool Name], select [option1], [option2], and [option3], then click either '[Button A]' (explains what it does) or '[Button B]' (explains what it does)" (lists all options and buttons with explanations)
     - ❌ WRONG: "Configure the [item]" (too generic)
     - ✅ CORRECT: "[Action1] [item1], [action2] [item2], [item3], [action4] [item4] using [tool/button], and [action5] [item5]" (lists all actions and items explicitly)

9. **Reasonable Inferences - ONLY FROM CONTEXT**
   - **CRITICAL: Only make inferences based on information that EXISTS in the context.**
   - If the context describes a web-based process accessible through a browser, you can state it works on mobile browsers (unless explicitly stated otherwise).
   - If the context mentions "My Account" and "My Orders" sections, you can describe how to access them on mobile.
   - **Only infer** what is reasonable based on what IS in the context - don't make up features or capabilities.
   - **DO NOT infer based on general knowledge or industry standards** - only use what's in the context.
   - **If the context doesn't mention something, DO NOT infer it exists** - acknowledge it's not in the knowledge base.

10. **When Information Is Truly Missing - NEVER USE GENERIC KNOWLEDGE**
   - **CRITICAL: If the context does not contain relevant information, DO NOT provide generic answers or general knowledge.**
   - **ONLY extract and present information that exists in the provided context.**
   - **If the question asks about something not in the context, clearly state: "The knowledge base does not contain information about [specific topic]. The available content covers [list what IS in the context]."**
   - **DO NOT provide generic definitions, explanations, or examples that are not in the context.**
   - **DO NOT say "3D modeling is..." or "Generally, 3D modeling involves..." if 3D modeling is not mentioned in the context.**
   - **DO NOT fill gaps with common knowledge or industry standards.**
   - **If specific details are missing but related information exists, extract what IS available from the context and present it.**
   - **ABSOLUTELY FORBIDDEN - NEVER USE THESE PHRASES**:
     - "The context does not specify..." (instead, say "The knowledge base does not contain information about...")
     - "The context does not mention..." (instead, say "This topic is not covered in the available knowledge base")
     - "The context does not include..." (instead, acknowledge what IS available)
     - "However, I can offer a general overview..." (NEVER provide generic information)
     - "Based on common knowledge..." (NEVER use general knowledge)
     - "Generally, [topic] involves..." (NEVER provide generic explanations)
     - Any variation that provides information not in the context
   - **CRITICAL**: If the context doesn't have the information:
     - **STOP immediately** - do not continue with generic information
     - Clearly state: "The knowledge base does not contain information about [topic]"
     - Optionally mention what topics ARE available in the context
     - **DO NOT provide any generic explanations, definitions, or examples**

11. **Completeness Checklist for ALL Questions**
   - **Before finishing your answer, verify you included:**
     - ✅ Every step in the process (count them - if context shows 15 steps, list all 15)
     - ✅ Every field that needs to be filled (whatever fields the context mentions - list them all)
     - ✅ Every button or action mentioned (whatever buttons/actions the context mentions - list them all)
     - ✅ Every option or setting (shipping methods, payment methods, toggles, etc.)
     - ✅ Every review or confirmation step (MANDATORY: Break down into specific sub-items. If context says "you can see X" → list "Review X (you can see X details)". If context says "you can edit Y" → list "Review Y (you can see Y details, and if needed, edit using Edit button)". If context says "you can manage Z" → list "Review Z (you can manage Z by clicking Manage Addresses)". If context mentions a button/toggle → list it with activation instructions. DO NOT combine these into one generic "review" statement)
     - ✅ Every file upload or management action
     - ✅ Every notification or email sent
     - ✅ Every post-completion action (order appears in dashboard, download invoice, etc.)
     - ✅ Every feature name, menu path, and exact terminology
   - **If you find yourself saying "you can add multiple products" or "various options" or "after entering details" - STOP and list the exact steps instead.**
   - **This checklist applies to ALL questions, not just workflow questions - always be comprehensive and detailed.**

12. **Do NOT Include - STRICTLY FORBIDDEN**
   - **Generic information, general knowledge, or common practices not in the context**
   - **Generic definitions or explanations (e.g., "3D modeling is the process of...") if not in the context**
   - **Industry standards or best practices not mentioned in the context**
   - **Common knowledge used to fill gaps**
   - External recommendations (support links, documentation references, etc.)
   - Information completely fabricated or not grounded in context
   - Personal opinions or filler phrases
   - Generic summaries when specific details are available in context
   - **Any information that does not appear in the provided context**

---

### 🧭 FORMAT SUMMARY

**Answer Format:**
1. Start with a direct, helpful answer to the question.
2. **FIRST: Scan the context for exact feature names** (capitalized terms like "Option Rules," "Additional Option Price," etc.) and **MUST include them** in your answer.
3. **Use specific examples and exact feature names** from the retrieved content - never paraphrase feature names.
4. **For ALL questions: List EVERY detail in order** - don't skip steps, don't combine steps, don't summarize. Include:
   - Every button, link, or menu item clicked
   - Every field filled (name it exactly as shown)
   - Every option selected or dropdown opened
   - Every toggle or switch mentioned
   - Every file upload step
   - Every review or confirmation screen
   - Every notification or email sent
   - Every tool, feature, or setting mentioned
5. **List ALL fields and options explicitly** - don't say "various fields" or "multiple options", list them all explicitly.
6. Organize by sections (using numbered or bulleted lists) - number every step, don't skip numbers.
7. Use clear formatting with headings, emojis, and structure.
8. Include all relevant menu paths, terminology, button labels, field names, and instructions from context.
9. **Ground every claim in the retrieved text** - avoid generic statements.
10. **When listing features, always use the exact capitalized names** found in the context (e.g., "Option Rules," "Combined Options Price," "Quantity Based Additional Options Price").
11. **CRITICAL: Completeness over brevity** - it's better to include too many details than to miss important steps or options. **ALL questions must be answered in a VERY DETAILED manner.**
12. **CRITICAL: Answer ONLY from knowledge base** - If the context doesn't contain information about the question, clearly state that the knowledge base doesn't contain that information. DO NOT provide generic explanations, definitions, or general knowledge.

**🔍 MANDATORY PRE-WRITING CHECKLIST - DO THIS BEFORE WRITING:**
1. **READ THE ENTIRE CONTEXT WORD BY WORD** - do not skim, read every sentence
2. **CREATE A DETAILED LIST** - write down EVERY single item mentioned in the context:
   - Every field name mentioned (whatever fields the context specifies)
   - Every button name mentioned (whatever buttons the context specifies)
   - Every option or setting mentioned
   - Every step in the process mentioned
   - Every tool or feature mentioned
   - Every review element mentioned (if a review step exists, list all elements that can be reviewed/edited)
3. **VERIFY YOUR LIST IS COMPLETE** - check that you've extracted:
   - ✅ All fields mentioned in the context (list them all)
   - ✅ All buttons mentioned in the context (list them all with explanations)
   - ✅ All tools/features mentioned in the context (list all options/methods for each)
   - ✅ **CRITICAL: All review steps with ALL reviewable elements broken down** - if review step exists:
     * Scan context for: "you can see", "you can edit", "you can manage", "you can change", "activate", "button", "toggle"
     * For EACH reviewable element found, create a separate sub-item with:
       - What can be reviewed (e.g., "order amount", "product details", "shipping address")
       - How to review/edit it (e.g., "you can see the order amount details, and if needed, change it manually")
       - Any buttons/toggles needed (e.g., "activate Payment Request button if needed, otherwise keep inactive")
     * DO NOT combine multiple reviewable elements into one statement
   - ✅ All file upload actions mentioned
   - ✅ All notifications/emails mentioned
4. **ONLY AFTER COMPLETING THE CHECKLIST** - start writing your answer
5. **INCLUDE EVERY ITEM FROM YOUR LIST** in the answer - do not skip any

**⚠️ CRITICAL EXAMPLES - THESE ARE MANDATORY (REAL EXAMPLES FROM CONTEXT):**

**Example 1 - Multiple Items in One Sentence:**
- Context says: "[item1], [item2], [item3]"
- ❌ WRONG: "Add [generic term] details"
- ✅ CORRECT: "Add [item1], [item2], and [item3]" (list all items separately, don't combine them)

**Example 2 - Multiple Buttons/Options:**
- Context says: "you can click on [tool], select [option1], [option2] and [option3]... you have over here the option [button A]... or you can click on [button B] and it will [explanation]"
- ❌ WRONG: "Use the [tool] to apply all necessary details"
- ✅ CORRECT: "Click [Tool Name], select [option1], [option2], and [option3]. Then click either '[Button A]' ([what it does]) or '[Button B]' ([what it does])" (mention ALL buttons/options with explanations)

**Example 3 - Configuration Steps:**
- Context says: "you can [action1] [item1]. Then you will need to add the details like say [item2], [item3], [item4], [item5] or [item6]. And you can [action2] [item7] from [tool/button]"
- ❌ WRONG: "Configure the [item] with necessary details"
- ✅ CORRECT: "[Action1] [item1]. Add [item2], [item3], [item4], and [item6]. [Action2] [item7] using [tool/button]" (list all items and actions explicitly)

**Example 4 - Review Step (CRITICAL - MUST BREAK DOWN ALL ELEMENTS):**
- Context says: "You can see the order amount details over here. If you want to change, you can change it manually from here. If you need to raise a payment request, you can activate this payment request button. Otherwise keep it inactive. Here you can see the card product details. Here you have the product details. You can edit the product details from this edit button if required. Shipping address you can manage from here by Manage Addresses. Billing address you can manage from here by clicking on Manage addresses. If you need, you can add the additional comments over here."
- ❌ WRONG: "Review the order details and confirm"
- ❌ WRONG: "Review the order amount, product details, and addresses, then confirm"
- ✅ CORRECT: "Review Order Details:
  - Review the order amount (you can see the order amount details over here, and if you want to change, you can change it manually from here)
  - Review product details (here you can see the card product details, here you have the product details, and if required, you can edit the product details from this edit button)
  - Review shipping address (you can manage the shipping address from here by clicking Manage Addresses)
  - Review billing address (you can manage the billing address from here by clicking on Manage Addresses)
  - Activate Payment Request button if you need to raise a payment request (otherwise keep it inactive)
  - Add any additional comments if needed (you can add the additional comments over here)
  After reviewing all details, click Confirm Order to place the order."

**MANDATORY RULES:**
- If context lists multiple items → List EVERY item separately
- If context mentions two buttons → Mention BOTH buttons
- If context says "you can X" → Include "X" as a step
- **DO NOT summarize** - if context lists 5 items, you list all 5 items explicitly
- **For review steps: Break down EVERY element that can be reviewed/edited** - this is MANDATORY:
  * Scan the context for phrases: "you can see", "you can edit", "you can manage", "you can change", "activate", "button", "toggle"
  * For EACH phrase found, create a separate bullet point with:
    - What is being reviewed (e.g., "order amount", "product details")
    - How to review/edit it (include the exact method from context)
    - Any buttons/toggles with activation instructions
  * DO NOT combine multiple elements into one statement
  * DO NOT say "Review the order details" - instead say "Review Order Details:" then list each element separately

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
    
    def _prioritize_dominant_source(self, source_docs: List, query: str = ""):
        """
        Prioritize chunks from the dominant source when multiple sources are present.
        Focus on 1-2 solid sources instead of many sources.
        Also considers source relevance based on query keywords.
        
        Args:
            source_docs: List of document chunks
            query: The original query to help identify most relevant source
            
        Returns:
            Tuple of (reordered_docs, source_counts, dominant_source_name, dominant_count, total_chunks)
        """
        from collections import defaultdict
        source_counts = defaultdict(int)
        source_relevance = defaultdict(float)  # Track relevance score per source
        source_content_richness = defaultdict(float)  # Track content richness per source
        
        # Extract keywords from query for relevance scoring
        query_lower = query.lower() if query else ""
        # Filter out common stop words and punctuation
        stop_words = {'the', 'a', 'an', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'what', 'how', 'when', 'where', 'who', 'which', 'why', 'from', 'one', 'need', 'doing', 'start', 'finish', 'process'}
        # Include words with 3+ characters, excluding stop words
        query_keywords = set([w.rstrip('.,!?;:') for w in query_lower.split() if len(w.rstrip('.,!?;:')) >= 3 and w.rstrip('.,!?;:') not in stop_words])
        
        # Track content richness per source (indicates comprehensive/detailed sources)
        source_content_richness = defaultdict(float)
        
        for doc in source_docs:
            source_key = doc.metadata.get('source_file', 'Unknown')
            source_counts[source_key] += 1
            
            # Score relevance based on content and metadata matching query
            content_lower = doc.page_content.lower()
            metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
            
            # Count keyword matches
            content_matches = sum(1 for kw in query_keywords if kw in content_lower)
            metadata_matches = sum(1 for kw in query_keywords if kw in metadata_text)
            
            # Calculate content richness (indicates comprehensive source):
            # - Longer content (more words) = more detailed
            # - More unique words = more diverse information
            # - More sentences = more structured information
            content_words = len(doc.page_content.split())
            unique_words = len(set(doc.page_content.lower().split()))
            sentences = doc.page_content.count('.') + doc.page_content.count('!') + doc.page_content.count('?')
            
            # Detect workflow/instructional content (indicates step-by-step detailed source):
            # Look for action verbs, step indicators, and instruction patterns
            action_verbs = ['click', 'select', 'add', 'enter', 'choose', 'upload', 'manage', 'edit', 'review', 'confirm', 'activate', 'save', 'apply', 'change', 'fill', 'specify', 'set']
            step_indicators = ['first', 'then', 'after', 'next', 'step', 'process', 'follow', 'proceed', 'before', 'once']
            instruction_patterns = ['you can', 'you will', 'you need to', 'you have to', 'if you', 'when you', 'to add', 'to select', 'to enter']
            
            action_count = sum(1 for verb in action_verbs if verb in content_lower)
            step_count = sum(1 for indicator in step_indicators if indicator in content_lower)
            instruction_count = sum(1 for pattern in instruction_patterns if pattern in content_lower)
            
            # Workflow score: indicates detailed step-by-step instructions (heavily weighted)
            workflow_score = (action_count * 3) + (step_count * 2) + (instruction_count * 2)
            
            # Richness score: combination of length, diversity, structure, and workflow content
            # Workflow content is the most important indicator of detailed sources
            richness_score = (content_words * 0.15) + (unique_words * 0.25) + (sentences * 0.15) + (workflow_score * 0.45)
            source_content_richness[source_key] += richness_score
            
            # Relevance score: query keywords + workflow content (workflow is VERY important for process questions)
            relevance_score = (content_matches * 2 + metadata_matches) + (workflow_score * 1.0)
            source_relevance[source_key] += relevance_score
        
        dominant_source_name = None
        dominant_count = 0
        total_chunks = len(source_docs)
        
        # Find the dominant source (has most chunks AND highest relevance)
        if source_counts and len(source_counts) > 1:
            # Sort sources by combined score: count + relevance
            # Normalize relevance scores
            max_relevance = max(source_relevance.values()) if source_relevance.values() else 1
            normalized_relevance = {k: v / max_relevance if max_relevance > 0 else 0 for k, v in source_relevance.items()}
            
            # Normalize content richness scores
            max_richness = max(source_content_richness.values()) if source_content_richness.values() else 1
            normalized_richness = {k: v / max_richness if max_richness > 0 else 0 for k, v in source_content_richness.items()}
            
            # Combined score: Prioritize sources with high relevance AND content richness
            # This ensures sources with detailed/comprehensive information are selected
            source_scores = {}
            for source, count in source_counts.items():
                relevance = normalized_relevance.get(source, 0)
                richness = normalized_richness.get(source, 0)
                
                # Filename relevance: Check if source filename matches query keywords
                # Sources with more specific/relevant filenames should be heavily prioritized
                filename_lower = source.lower()
                # Extract meaningful words from filename (remove common words, file extensions)
                # Keep "admin" as it's relevant for "from admin" queries
                filename_words = [w for w in re.findall(r'\b\w+\b', filename_lower) 
                                if len(w) > 2 and w not in ['from', 'onprintshop', 'insights', 'json', 'uploads', 'web', 'print', 'storefront', 'v100', 'v121', 'the']]
                
                # Check if query keywords appear in filename (strong match)
                # Use same threshold as query_keywords (>= 3 characters)
                filename_matches = sum(1 for kw in query_keywords if kw in filename_lower)
                # Also check if filename words match query keywords (bidirectional)
                filename_to_query = sum(1 for fw in filename_words if any(fw in kw or kw in fw for kw in query_keywords))
                
                # Strong boost for filenames with key query terms
                # For "add offline order" query, boost sources with "add", "order", "new", "offline" in filename
                key_terms_in_query = ['add', 'order', 'new', 'offline', 'customer', 'admin']
                # Direct matches: terms that appear in both query and filename
                key_term_matches = sum(1 for term in key_terms_in_query if term in query_lower and term in filename_lower)
                
                # Contextual matches: boost for relevant terms in filename even if not explicitly in query
                # For "add order" queries, "admin" in filename is highly relevant (implies "add order from admin")
                # For "add order" queries, "new" in filename is relevant (implies "add new order")
                contextual_boost = 0
                if any(term in query_lower for term in ['add', 'order', 'create', 'place']):
                    if 'admin' in filename_lower:
                        contextual_boost += 1.5  # Strong boost for admin sources when adding orders
                    if 'new' in filename_lower:
                        contextual_boost += 1.0  # Boost for "new order" sources
                
                # Filename relevance: combination of keyword matches, key term matches, and contextual boosts
                # Key term matches get 4x weight because they're very specific indicators
                # Use a more reasonable normalization that doesn't penalize for having many query keywords
                base_score = filename_matches * 2.0 + filename_to_query * 1.0 + key_term_matches * 4.0 + contextual_boost * 2.0
                # Normalize based on expected key term matches (most important) plus some keyword matches
                # This prevents over-penalization when query has many keywords
                expected_key_terms = sum(1 for term in key_terms_in_query if term in query_lower)
                max_possible = max(expected_key_terms * 4.0 + min(len(query_keywords), 5) * 2.0 + 2.5, 10.0)  # Added 2.5 for contextual boost
                filename_relevance = min(base_score / max_possible, 1.0)
                
                # Combined score: Heavily prioritize relevance, richness, and filename over chunk count
                # For workflow questions, sources with detailed instructions AND relevant filenames should score much higher
                # 3% chunk count, 45% relevance, 27% content richness, 25% filename relevance
                # Filename gets high weight because it's a strong indicator of source relevance
                combined_score = (count / total_chunks) * 0.03 + relevance * 0.45 + richness * 0.27 + filename_relevance * 0.25
                source_scores[source] = (combined_score, count, relevance, richness, filename_relevance)
            
            # Sort by combined score
            sorted_sources = sorted(source_scores.items(), key=lambda x: x[1][0], reverse=True)
            dominant_source = sorted_sources[0]
            dominant_source_name = dominant_source[0]
            dominant_count = dominant_source[1][1]
            dominant_relevance = dominant_source[1][2]
            dominant_richness = dominant_source[1][3]
            
            print(f"   Source analysis:")
            for source, (score, count, rel, rich, filename_rel) in sorted_sources[:3]:
                # Calculate breakdown for debugging (matching actual weights: 3%, 45%, 27%, 25%)
                count_score = (count / total_chunks) * 0.03
                rel_score = rel * 0.45
                rich_score = rich * 0.27
                filename_score = filename_rel * 0.25
                print(f"     - {source}: {count} chunks (count: {count_score:.3f}, rel: {rel_score:.3f}, rich: {rich_score:.3f}, filename: {filename_score:.3f}) = {score:.3f}")
            
            # Keep all sources that meet quality threshold (not artificially limited to 1-2)
            # Quality threshold: keep sources with score >= 50% of top source's score (lowered for better inclusion)
            # This ensures we get all good sources, especially those with relevant filenames but fewer chunks
            top_score = sorted_sources[0][1][0]
            quality_threshold = top_score * 0.5  # Keep sources with at least 50% of top score (lowered from 60%)
            
            kept_sources_list = []
            for source, (score, count, rel, rich, filename_rel) in sorted_sources:
                if score >= quality_threshold:
                    kept_sources_list.append(source)
                    print(f"   ✓ Keeping source: '{source}' (score: {score:.3f} >= threshold: {quality_threshold:.3f})")
                else:
                    print(f"   ✗ Excluding source: '{source}' (score: {score:.3f} < threshold: {quality_threshold:.3f})")
            
            print(f"   Selected {len(kept_sources_list)} source(s) based on quality threshold")
            
            # Filter to keep only quality sources
            kept_sources_set = set(kept_sources_list)
            filtered_docs = []
            for doc in source_docs:
                if doc.metadata.get('source_file') in kept_sources_set:
                    filtered_docs.append(doc)
            
            # Reorder: put chunks in order of source ranking (dominant source first, then others by score)
            source_order = {source: i for i, (source, _) in enumerate(sorted_sources)}
            
            # Sort docs by their source's ranking (lower index = higher rank)
            filtered_docs.sort(key=lambda doc: source_order.get(doc.metadata.get('source_file', ''), 999))
            
            print(f"   Using {len(filtered_docs)} chunks from {len(kept_sources_list)} source(s)")
            return filtered_docs, source_counts, dominant_source_name, dominant_count, len(filtered_docs)
        
        return source_docs, source_counts, dominant_source_name, dominant_count, total_chunks
    
    def _hybrid_search(self, query: str, keywords: List[str], k: int) -> List:
        """
        Perform hybrid search combining semantic similarity with keyword matching.
        Universal fix: Always performs keyword-based search when keywords are present,
        ensuring specific terms (like acronyms) are found even if semantic ranking is low.
        
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
        
        # Detect if keywords contain specific terms (acronyms, all caps, etc.)
        # These need more aggressive keyword matching
        # Check for: uppercase terms, short acronym-like terms (3-5 chars, alphanumeric)
        # Also check original query for uppercase terms since keywords are lowercased
        has_specific_terms = any(
            len(kw) <= 5 and (kw.isupper() or (kw.isalnum() and len(kw) >= 3 and len(kw) <= 5))
            for kw in keywords
        ) or any(
            len(word) >= 3 and len(word) <= 5 and word.isupper() and word.isalnum()
            for word in re.findall(r'\b\w+\b', query)
        )
        
        # For keyword-based search, retrieve more documents to ensure we find keyword matches
        # This is especially important for specific terms like acronyms
        search_multiplier = 3 if has_specific_terms else 2
        broader_k = k * search_multiplier
        
        # Also extract uppercase versions of keywords from original query for better matching
        # This helps catch acronyms that might be stored in different cases
        query_uppercase_words = [w for w in re.findall(r'\b\w+\b', query) if w.isupper() and len(w) >= 3]
        all_keywords = set(keywords)
        for upper_word in query_uppercase_words:
            all_keywords.add(upper_word.lower())  # Add lowercase version for matching
        
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
            
            # Count keyword matches in content and metadata (using enhanced keyword set)
            keyword_score = 0
            for keyword in all_keywords:
                if keyword in content_lower:
                    keyword_score += 2  # Content matches are more important
                if keyword in metadata_text:
                    keyword_score += 1  # Metadata matches are less important
            
            # Combine semantic position (earlier = better) with keyword score
            # Documents that appear early in semantic search AND have keywords get highest priority
            # Boost keyword matches significantly to ensure they rank high
            semantic_position_score = max(0, len(semantic_docs) - semantic_docs.index(doc))
            keyword_boost = keyword_score * 5 if keyword_score > 0 else 0  # Increased boost for keyword matches
            combined_score = semantic_position_score + keyword_boost
            scored_docs.append((combined_score, doc))
        
        # UNIVERSAL FIX: Always perform keyword-based search when keywords are present
        # This ensures specific terms (like "BXGX") are found even if semantic search ranks them low
        try:
            # Retrieve broader set of documents for keyword matching
            broader_docs = self.vectorstore.similarity_search(query, k=broader_k)
            
            # Use the enhanced keyword set already created above
            
            for doc in broader_docs:
                doc_id = id(doc)
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                
                content_lower = doc.page_content.lower()
                metadata_text = " ".join([str(v) for v in doc.metadata.values()]).lower()
                
                # Count keyword matches (check both lowercase keywords and uppercase versions)
                keyword_score = 0
                for keyword in all_keywords:
                    if keyword in content_lower:
                        keyword_score += 2
                    if keyword in metadata_text:
                        keyword_score += 1
                
                # If document has keyword matches, include it with high priority
                if keyword_score > 0:
                    # Documents with keyword matches get high score even if semantically lower
                    # This ensures acronyms and specific terms are found
                    keyword_boost = keyword_score * 4  # High boost for keyword matches
                    # Add small semantic position boost if it was in broader results
                    try:
                        broader_position = max(0, len(broader_docs) - broader_docs.index(doc))
                        combined_score = keyword_boost + (broader_position * 0.5)
                    except:
                        combined_score = keyword_boost
                    scored_docs.append((combined_score, doc))
        except Exception as e:
            print(f"   Warning: Keyword search expansion failed: {e}")
        
        # Sort by combined score (highest first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k documents, prioritizing those with keyword matches
        result_docs = [doc for _, doc in scored_docs[:k]]
        
        # If we still don't have enough, fill with remaining semantic results
        if len(result_docs) < k:
            for doc in semantic_docs:
                if doc not in result_docs:
                    result_docs.append(doc)
                    if len(result_docs) >= k:
                        break
        
        return result_docs[:k]
    
    def _enhance_query_for_specific_terms(self, question: str) -> str:
        """
        Enhance query by adding related context terms for specific acronyms/terms.
        This helps semantic search find relevant content even when the term itself
        might not have strong semantic similarity.
        
        Args:
            question: Original question
            
        Returns:
            Enhanced question with related context terms
        """
        enhanced = question
        question_upper = question.upper()
        
        # Dictionary of specific terms and their related context terms
        term_expansions = {
            'BXGX': ['BOGO', 'Buy X Get X', 'discount', 'offer', 'promotion', 'coupon', 'reward points'],
            'BOGO': ['Buy One Get One', 'discount', 'offer', 'promotion', 'coupon', 'BXGX'],
            'API': ['application programming interface', 'integration', 'endpoint', 'webhook'],
            'CMS': ['content management system', 'content', 'pages', 'blocks'],
            'B2B': ['business to business', 'corporate', 'store', 'private store'],
            'B2C': ['business to consumer', 'customer', 'public store'],
        }
        
        # Check for specific terms in the question
        for term, expansions in term_expansions.items():
            if term in question_upper or term.lower() in question.lower():
                # Add related terms to help semantic search
                expansion_text = ' '.join(expansions)
                enhanced = f"{enhanced} {expansion_text}"
                print(f"   Enhanced query with context for '{term}': added related terms")
                break  # Only enhance for the first matching term to avoid clutter
        
        return enhanced.strip()
    
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
    
    def _sanitize_answer(self, answer: str) -> str:
        """
        Sanitize answer by removing template syntax that might have been output by the LLM.
        Removes Django/Jinja2 template syntax like '%% block ... %%', '{%% ... %%}', '{{ ... }}'
        """
        try:
            if not answer:
                return answer
            
            # Convert to string if it's not already
            if not isinstance(answer, str):
                answer = str(answer)
            
            # Remove Django template syntax: {% ... %} and {{ ... }}
            answer = re.sub(r'\{\%.*?\%\}', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\{\{.*?\}\}', '', answer, flags=re.DOTALL)
            
            # Remove Django template syntax: % block ... % and % endblock %
            answer = re.sub(r'%\s*block\s+\w+\s*%', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'%\s*endblock\s*%', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'%\s*extends\s+.*?%', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'%\s*include\s+.*?%', '', answer, flags=re.IGNORECASE)
            
            # Remove any remaining template syntax patterns
            answer = re.sub(r'%\s*\w+.*?%', '', answer)
            
            # Remove standalone % characters that might cause errors (but preserve % in normal text like "50%")
            # Only remove % if it's followed by whitespace or at end of line, and not part of a number
            answer = re.sub(r'(?<!\d)%\s+(?!\d)', '', answer)  # % followed by space, not part of number
            answer = re.sub(r'(?<!\d)%$', '', answer, flags=re.MULTILINE)  # % at end of line, not part of number
            answer = re.sub(r'^\s*%\s*$', '', answer, flags=re.MULTILINE)  # Lines with only %
            
            # Remove any lines that are just template syntax or contain problematic patterns
            lines = answer.split('\n')
            cleaned_lines = []
            for line in lines:
                try:
                    stripped = line.strip()
                    # Skip lines that are template syntax patterns
                    if stripped and len(stripped) > 0:
                        # Skip if it's a template pattern (starts with % and is short, or contains template markers)
                        if stripped.startswith('%') and len(stripped) < 50 and 'endblock' not in stripped.lower():
                            # This is likely template syntax, skip it
                            continue
                        elif '{%' in stripped or '{{' in stripped:
                            # Contains template markers, skip the line
                            continue
                        else:
                            cleaned_lines.append(line)
                    else:
                        # Empty line, keep it for formatting
                        cleaned_lines.append(line)
                except Exception:
                    # If processing a line fails, skip it
                    continue
            answer = '\n'.join(cleaned_lines)
            
            # Final pass: remove any remaining problematic % patterns
            answer = re.sub(r'^\s*%\s+', '', answer, flags=re.MULTILINE)  # % at start of line followed by space
            
            # Clean up extra whitespace
            answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Multiple newlines to double
            answer = answer.strip()
            
            return answer
        except Exception as e:
            # If anything fails, return the original answer (better than crashing)
            print(f"   ⚠️  Warning: Sanitization error: {e}")
            return answer if isinstance(answer, str) else str(answer) if answer else ""
    
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
        
        # Enhance query for specific terms/acronyms to improve semantic search
        enhanced_question = self._enhance_query_for_specific_terms(enhanced_question)
        
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
        # Focus on retrieving fewer but more relevant sources (1-2 solid sources)
        # Check if question is asking about a process/workflow
        is_workflow_question = any(phrase in corrected_question.lower() for phrase in [
            'how to', 'how do', 'process', 'steps', 'procedure', 'workflow', 'from start to finish',
            'complete process', 'what is the process', 'what are the steps'
        ])
        
        # For workflow questions, retrieve more chunks to ensure we get the right source
        if is_workflow_question:
            print(f"   Workflow question detected - retrieving more chunks to capture all steps")
            original_k = self.retriever.search_kwargs.get("k", 15)
            # Retrieve more chunks to increase chances of getting the right source
            self.retriever.search_kwargs["k"] = 30  # More chunks for workflow questions
            source_docs = self._hybrid_search(corrected_question, keywords, k=30)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
            print(f"   Retrieved {len(source_docs)} chunks")
        else:
            # For most questions, retrieve fewer but more focused chunks
            source_docs = self._hybrid_search(corrected_question, keywords, k=15)
            print(f"   Retrieved {len(source_docs)} chunks")
        
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
                
                # CRITICAL FIX: Ensure chunks with key detail terms are included
                # Key terms that must be preserved in context (artwork name, quantity, weight, etc.)
                key_detail_terms = [
                    'artwork name', 'quantity', 'product weight', 'production days', 
                    'upload artwork', 'upload file', 'price calculator', 'apply all', 
                    'apply options only', 'apply only options', 'save and add another'
                ]
                
                # Get top chunks from re-ranking
                top_reranked = [doc for _, doc in scored_docs[:FINAL_CHUNKS]]
                
                # Find chunks with key detail terms that might have been filtered out
                detail_chunks = []
                for score, doc in scored_docs[FINAL_CHUNKS:]:  # Check chunks that didn't make top FINAL_CHUNKS
                    content_lower = doc.page_content.lower()
                    if any(term in content_lower for term in key_detail_terms):
                        detail_chunks.append(doc)
                
                # If we found detail chunks that were filtered out, add them
                if detail_chunks:
                    print(f"   Found {len(detail_chunks)} chunks with key details that were filtered out - including them")
                    # Add detail chunks, but limit total to FINAL_CHUNKS + a few more
                    max_chunks = FINAL_CHUNKS + min(5, len(detail_chunks))  # Add up to 5 more detail chunks
                    # Combine top reranked with detail chunks, removing duplicates
                    combined = top_reranked.copy()
                    for detail_doc in detail_chunks:
                        if detail_doc not in combined and len(combined) < max_chunks:
                            combined.append(detail_doc)
                    source_docs = combined
                    print(f"   Using {len(source_docs)} chunks (including {len(detail_chunks)} detail chunks)")
                else:
                    source_docs = top_reranked
                print(f"   Selected top {len(source_docs)} chunks after re-ranking")
            except Exception as e:
                print(f"   ⚠️  Re-ranking failed: {e}, using original order")
        
        # Prioritize dominant source to prevent mixing information from different sources
        source_docs, source_counts, dominant_source_name, dominant_count, total_chunks = self._prioritize_dominant_source(source_docs, corrected_question)
        
        # Hybrid approach: Use both video and text sources for context
        # IMPORTANT: Use prioritized source_docs (not all_retrieved) to respect source prioritization
        context_start = time.time()
        # Separate video and text sources from prioritized source_docs
        video_docs = []
        text_docs = []
        for doc in source_docs:
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
        formatted_context = self.format_docs(context_docs)
        
        # Add source combination indicator if multiple sources exist
        if len(source_counts) > 1:
            all_sources = list(source_counts.keys())
            source_indicator = f"""
=== 🎯 MULTIPLE SOURCES DETECTED - COMBINE ALL INFORMATION ===
The context contains information from {len(source_counts)} different sources. You MUST combine ALL details from ALL sources:

SOURCES FOUND:
{chr(10).join([f"- {source}" for source in all_sources[:5]])}
{'...' if len(all_sources) > 5 else ''}

**CRITICAL INSTRUCTION**: Extract and COMBINE EVERY detail from ALL sources above. 
- If ANY source mentions "artwork name", "quantity", "product weight", "production days" → Include ALL of these fields
- If ANY source mentions "upload artwork" or "upload file" → Include this step
- If ANY source mentions "Online Price Calculator" with "Apply All" and "Apply Options Only" → Include BOTH buttons with explanations
- If ANY source has a detailed review step → Break down ALL elements separately
- If one source has simplified steps and another has detailed steps → Include BOTH sets of details
- **DO NOT skip details from any source - combine ALL information to create the MOST COMPLETE answer possible**

**COMBINATION RULE**: When sources have different levels of detail:
- If Source A has: "Select product, set quantity, confirm"
- And Source B has: "Select product, set quantity, upload artwork, set weight, set production days, use price calculator, confirm"
- Your answer MUST include: "Select product, set quantity, upload artwork, set weight, set production days, use price calculator, confirm"
- **Always use the MORE DETAILED version and add any unique details from other sources**

===

"""
            formatted_context = source_indicator + formatted_context
        
        context_text = self._enhance_context_with_feature_names(formatted_context)
        
        # Escape % characters in context_text to prevent KeyError during template formatting
        # Python's string formatter interprets % as format placeholders, so we need to escape them as %%
        # This prevents KeyError when context contains literal '% ' or other % patterns
        # IMPORTANT: We need to escape ALL % characters, including those in %% patterns
        # The issue is that Python's formatter sees %% and interprets it, so we need to escape %% to %%%%
        # But %%%% is also interpreted! So we need to remove or replace problematic patterns entirely
        
        # Strategy: Replace problematic % patterns with a safe placeholder before escaping
        # This prevents Python's formatter from trying to interpret them
        # Replace patterns like '% block ... %' or '%% block ... %%' with a safe placeholder
        context_text = re.sub(r'%+\s*block\s+\w+.*?%+', 'TEMPLATE_BLOCK_PLACEHOLDER', context_text, flags=re.IGNORECASE | re.DOTALL)
        # Replace any remaining %% patterns (which would become %%%% and cause issues)
        context_text = re.sub(r'%%+', 'PERCENT_PLACEHOLDER', context_text)
        # Now escape remaining single %
        context_text = context_text.replace('%', '%%')
        # Restore the placeholders as safe text (no % characters, no curly braces that could be interpreted as placeholders)
        # Use square brackets to avoid any formatting issues
        context_text = context_text.replace('TEMPLATE_BLOCK_PLACEHOLDER', '[template syntax removed]')
        context_text = context_text.replace('PERCENT_PLACEHOLDER', '[percent sign]')
        # Escape any standalone curly braces in the context (but preserve {context} placeholder by doing this carefully)
        # We need to escape { and } that are NOT part of {context} or other known placeholders
        # Since we're using {context} as the placeholder, we need to be careful
        # For now, let's just escape any { or } that appear in the content itself
        # But we can't do a simple replace because it would break {context}
        # Actually, LangChain handles this - it only interprets {context} as a placeholder, not other {text}
        # So we don't need to escape curly braces at all
        
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
        # Debug: Check if context_text contains problematic patterns before formatting
        if '% ' in context_text or '%' in context_text:
            print(f"   ⚠️  WARNING: Context contains % characters. Escaping them...")
            # Double-check escaping is applied
            context_text = context_text.replace('%', '%%')
            print(f"   ✓ Escaped context (length: {len(context_text)})")
        
        try:
            formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
        except KeyError as ke:
            # If we still get a KeyError, handle it
            error_key = str(ke)
            print(f"   ❌ KeyError during template formatting: {ke}")
            print(f"   Error key: {error_key}")
            print(f"   Context preview (first 500 chars): {context_text[:500]}")
            
            # Handle template block pattern errors specifically (check first)
            if "template block pattern" in error_key.lower() or ("block" in error_key.lower() and "template" in error_key.lower()):
                print(f"   🔍 Found template block pattern error - removing all template patterns...")
                # Remove all template block patterns more aggressively
                context_text = re.sub(r'%+\s*block\s+\w+.*?%+', '[template syntax removed]', context_text, flags=re.IGNORECASE | re.DOTALL)
                context_text = re.sub(r'%+\s*endblock\s*%+', '', context_text, flags=re.IGNORECASE)
                context_text = re.sub(r'%+\s*\w+.*?%+', '', context_text)  # Remove any remaining % pattern %
                # Escape all remaining % characters
                context_text = context_text.replace('%', '%%')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except Exception as e2:
                    print(f"   ⚠️  Still failing after aggressive cleanup: {e2}")
                    # Last resort: remove all % characters entirely
                    context_text = context_text.replace('%', '').replace('%%', '')
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            # Handle both '% ' and '%% ' errors
            elif "'%% '" in error_key or "%% " in context_text:
                print(f"   🔍 Found '%% ' pattern - replacing with placeholder...")
                # Replace %%  with a safe placeholder that won't be interpreted
                context_text = context_text.replace('%% ', 'PERCENT_SPACE ')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except:
                    # If that still fails, try removing the pattern entirely
                    context_text = context_text.replace('PERCENT_SPACE ', '')
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            elif "'% '" in error_key or '% ' in context_text:
                print(f"   🔍 Found '% ' in context, attempting to escape again...")
                context_text = context_text.replace('% ', '%% ')
                context_text = context_text.replace('%', '%%')
                print(f"   Retrying with re-escaped context...")
                formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            else:
                # For any other KeyError, try aggressive cleanup
                print(f"   🔍 Unknown KeyError pattern ({error_key}) - attempting aggressive cleanup...")
                # Remove all template-like patterns
                context_text = re.sub(r'%+\s*\w+.*?%+', '', context_text)
                context_text = context_text.replace('%', '%%')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except Exception as e2:
                    print(f"   ⚠️  Error persists after cleanup: {e2}")
                    # Last resort: remove all % characters
                    context_text = context_text.replace('%', '').replace('%%', '')
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
        try:
            formatted_messages = conversation_prompt.format_messages()
        except KeyError as ke:
            # Handle KeyError from Python's formatter trying to interpret % patterns
            error_key = str(ke)
            print(f"   ❌ KeyError during message formatting: {ke}")
            print(f"   Error key: {error_key}")
            
            # If the error is about % patterns, we need to fix the context in the messages
            if '%' in error_key:
                print(f"   🔍 Detected % pattern in error - fixing context in messages...")
                # Re-escape the context in all messages
                fixed_messages = []
                for role, content in messages:
                    if isinstance(content, str) and '%' in content:
                        # Apply the same escaping logic to message content
                        fixed_content = content
                        fixed_content = re.sub(r'%+\s*block\s+\w+.*?%+', 'TEMPLATE_BLOCK_PLACEHOLDER', fixed_content, flags=re.IGNORECASE | re.DOTALL)
                        fixed_content = re.sub(r'%%+', 'PERCENT_PLACEHOLDER', fixed_content)
                        fixed_content = fixed_content.replace('%', '%%')
                        fixed_content = fixed_content.replace('TEMPLATE_BLOCK_PLACEHOLDER', '[template syntax removed]')
                        fixed_content = fixed_content.replace('PERCENT_PLACEHOLDER', '[percent sign]')
                        # Escape curly braces to prevent them from being interpreted as placeholders
                        fixed_content = fixed_content.replace('{', '{{').replace('}', '}}')
                        fixed_messages.append((role, fixed_content))
                    else:
                        fixed_messages.append((role, content))
                
                # Recreate the prompt with fixed messages
                from langchain_core.prompts import ChatPromptTemplate
                conversation_prompt = ChatPromptTemplate.from_messages(fixed_messages)
                formatted_messages = conversation_prompt.format_messages()
            else:
                # For any other KeyError, try aggressive cleanup on messages
                print(f"   🔍 Unknown KeyError pattern ({error_key}) - attempting aggressive cleanup...")
                # Fix all messages by removing template patterns
                fixed_messages = []
                for role, content in messages:
                    if isinstance(content, str):
                        # Remove all template-like patterns
                        fixed_content = re.sub(r'%+\s*\w+.*?%+', '', content)
                        fixed_content = fixed_content.replace('%', '%%')
                        # Escape curly braces
                        fixed_content = fixed_content.replace('{', '{{').replace('}', '}}')
                        fixed_messages.append((role, fixed_content))
                    else:
                        fixed_messages.append((role, content))
                
                # Recreate the prompt with fixed messages
                from langchain_core.prompts import ChatPromptTemplate
                conversation_prompt = ChatPromptTemplate.from_messages(fixed_messages)
                try:
                    formatted_messages = conversation_prompt.format_messages()
                except Exception as e2:
                    print(f"   ⚠️  Error persists after cleanup: {e2}")
                    # Last resort: remove all % characters from messages
                    final_fixed_messages = []
                    for role, content in fixed_messages:
                        if isinstance(content, str):
                            final_content = content.replace('%', '').replace('%%', '')
                            final_fixed_messages.append((role, final_content))
                        else:
                            final_fixed_messages.append((role, content))
                    conversation_prompt = ChatPromptTemplate.from_messages(final_fixed_messages)
                    formatted_messages = conversation_prompt.format_messages()
        except Exception as e:
            import traceback
            import sys
            error_str = str(e)
            error_trace = traceback.format_exc()
            
            # Print to both stderr and stdout with flush
            print(f"   ❌ Error formatting messages: {error_str}", file=sys.stderr, flush=True)
            print(f"   Traceback: {error_trace}", file=sys.stderr, flush=True)
            print(f"   ❌ Error formatting messages: {error_str}", flush=True)
            print(f"   Traceback: {error_trace}", flush=True)
            
            yield {
                "type": "error",
                "content": "Error preparing query. Please try again."
            }
            return
        
        try:
            # Debug: Log first chunk to see what LLM is outputting
            first_chunk_logged = False
            
            for chunk in self.llm.stream(formatted_messages):
                try:
                    if chunk.content:
                        # Debug: Log the first chunk to see if LLM is outputting without spaces
                        if not first_chunk_logged and chunk.content:
                            print(f"   🔍 DEBUG: First LLM chunk (first 100 chars): {repr(chunk.content[:100])}")
                            print(f"   🔍 DEBUG: First chunk has spaces: {' ' in chunk.content}")
                            first_chunk_logged = True
                        
                        # Don't sanitize individual chunks - let LangChain handle spacing naturally
                        # Only yield the raw chunk content to preserve proper token spacing
                        if chunk.content:
                            full_answer += chunk.content
                            yield {
                                "type": "content",
                                "content": chunk.content
                            }
                except Exception as stream_error:
                    # Log streaming error but continue
                    import traceback
                    import sys
                    error_str = str(stream_error)
                    error_trace = traceback.format_exc()
                    
                    # Print to both stderr and stdout
                    print(f"   ⚠️  Warning: Error processing stream chunk: {error_str}", file=sys.stderr, flush=True)
                    print(f"   Traceback: {error_trace}", file=sys.stderr, flush=True)
                    print(f"   ⚠️  Warning: Error processing stream chunk: {error_str}", flush=True)
                    print(f"   Traceback: {error_trace}", flush=True)
                    continue
        except Exception as llm_error:
            # Catch any errors from the LLM stream itself
            import traceback
            import sys
            error_str = str(llm_error)
            error_trace = traceback.format_exc()
            
            # Print to both stderr and stdout to ensure visibility
            print(f"   ❌ CRITICAL: Error in LLM stream: {error_str}", file=sys.stderr, flush=True)
            print(f"   Full traceback: {error_trace}", file=sys.stderr, flush=True)
            print(f"   ❌ CRITICAL: Error in LLM stream: {error_str}", flush=True)
            print(f"   Full traceback: {error_trace}", flush=True)
            
            # Yield error - escape special characters
            safe_error = error_str.replace('"', '\\"').replace('\n', ' ').replace('\r', '')[:200]
            yield {
                "type": "error",
                "content": f"Error generating response: {safe_error}"
            }
            return
        timing_info["llm_generation"] = time.time() - llm_start
        
        # Only send sources if they're relevant
        # Sanitize the final answer to remove any template syntax
        # Do this at the end to avoid interfering with streaming and token spacing
        try:
            full_answer = self._sanitize_answer(full_answer)
        except Exception as e:
            print(f"   ⚠️  Warning: Error sanitizing final answer: {e}")
            # Continue with unsanitized answer if sanitization fails
        
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
        
        # Enhance query for specific terms/acronyms to improve semantic search
        enhanced_question = self._enhance_query_for_specific_terms(enhanced_question)
        
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
        # Focus on retrieving fewer but more relevant sources (1-2 solid sources)
        # Check if question is asking about a process/workflow
        is_workflow_question = any(phrase in corrected_question.lower() for phrase in [
            'how to', 'how do', 'process', 'steps', 'procedure', 'workflow', 'from start to finish',
            'complete process', 'what is the process', 'what are the steps'
        ])
        
        # For workflow questions, retrieve more chunks to ensure we get the right source
        if is_workflow_question:
            print(f"   Workflow question detected - retrieving more chunks to capture all steps")
            original_k = self.retriever.search_kwargs.get("k", 15)
            # Retrieve more chunks to increase chances of getting the right source
            self.retriever.search_kwargs["k"] = 30  # More chunks for workflow questions
            source_docs = self._hybrid_search(corrected_question, keywords, k=30)
            self.retriever.search_kwargs["k"] = original_k  # Restore original
            print(f"   Retrieved {len(source_docs)} chunks")
        else:
            # For most questions, retrieve fewer but more focused chunks
            source_docs = self._hybrid_search(corrected_question, keywords, k=15)
            print(f"   Retrieved {len(source_docs)} chunks")
        
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
                
                # CRITICAL FIX: Ensure chunks with key detail terms are included
                # Key terms that must be preserved in context (artwork name, quantity, weight, etc.)
                key_detail_terms = [
                    'artwork name', 'quantity', 'product weight', 'production days', 
                    'upload artwork', 'upload file', 'price calculator', 'apply all', 
                    'apply options only', 'apply only options', 'save and add another'
                ]
                
                # Get top chunks from re-ranking
                top_reranked = [doc for _, doc in scored_docs[:FINAL_CHUNKS]]
                
                # Find chunks with key detail terms that might have been filtered out
                detail_chunks = []
                for score, doc in scored_docs[FINAL_CHUNKS:]:  # Check chunks that didn't make top FINAL_CHUNKS
                    content_lower = doc.page_content.lower()
                    if any(term in content_lower for term in key_detail_terms):
                        detail_chunks.append(doc)
                
                # If we found detail chunks that were filtered out, add them
                if detail_chunks:
                    print(f"   Found {len(detail_chunks)} chunks with key details that were filtered out - including them")
                    # Add detail chunks, but limit total to FINAL_CHUNKS + a few more
                    max_chunks = FINAL_CHUNKS + min(5, len(detail_chunks))  # Add up to 5 more detail chunks
                    # Combine top reranked with detail chunks, removing duplicates
                    combined = top_reranked.copy()
                    for detail_doc in detail_chunks:
                        if detail_doc not in combined and len(combined) < max_chunks:
                            combined.append(detail_doc)
                    source_docs = combined
                    print(f"   Using {len(source_docs)} chunks (including {len(detail_chunks)} detail chunks)")
                else:
                    source_docs = top_reranked
                print(f"   Selected top {len(source_docs)} chunks after re-ranking")
            except Exception as e:
                print(f"   ⚠️  Re-ranking failed: {e}, using original order")
        
        # Prioritize dominant source to prevent mixing information from different sources
        source_docs, source_counts, dominant_source_name, dominant_count, total_chunks = self._prioritize_dominant_source(source_docs, corrected_question)
        
        # Hybrid approach: Use both video and text sources for context
        # IMPORTANT: Use prioritized source_docs (not all_retrieved) to respect source prioritization
        context_start = time.time()
        # Separate video and text sources from prioritized source_docs
        video_docs = []
        text_docs = []
        for doc in source_docs:
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
        formatted_context = self.format_docs(context_docs)
        
        # Add source combination indicator if multiple sources exist
        if len(source_counts) > 1:
            all_sources = list(source_counts.keys())
            source_indicator = f"""
=== 🎯 MULTIPLE SOURCES DETECTED - COMBINE ALL INFORMATION ===
The context contains information from {len(source_counts)} different sources. You MUST combine ALL details from ALL sources:

SOURCES FOUND:
{chr(10).join([f"- {source}" for source in all_sources[:5]])}
{'...' if len(all_sources) > 5 else ''}

**CRITICAL INSTRUCTION**: Extract and COMBINE EVERY detail from ALL sources above. 
- If ANY source mentions "artwork name", "quantity", "product weight", "production days" → Include ALL of these fields
- If ANY source mentions "upload artwork" or "upload file" → Include this step
- If ANY source mentions "Online Price Calculator" with "Apply All" and "Apply Options Only" → Include BOTH buttons with explanations
- If ANY source has a detailed review step → Break down ALL elements separately
- If one source has simplified steps and another has detailed steps → Include BOTH sets of details
- **DO NOT skip details from any source - combine ALL information to create the MOST COMPLETE answer possible**

**COMBINATION RULE**: When sources have different levels of detail:
- If Source A has: "Select product, set quantity, confirm"
- And Source B has: "Select product, set quantity, upload artwork, set weight, set production days, use price calculator, confirm"
- Your answer MUST include: "Select product, set quantity, upload artwork, set weight, set production days, use price calculator, confirm"
- **Always use the MORE DETAILED version and add any unique details from other sources**

===

"""
            formatted_context = source_indicator + formatted_context
        
        context_text = self._enhance_context_with_feature_names(formatted_context)
        
        # Escape % characters in context_text to prevent KeyError during template formatting
        # Python's string formatter interprets % as format placeholders, so we need to escape them as %%
        # This prevents KeyError when context contains literal '% ' or other % patterns
        # IMPORTANT: We need to escape ALL % characters, including those in %% patterns
        # The issue is that Python's formatter sees %% and interprets it, so we need to escape %% to %%%%
        # But %%%% is also interpreted! So we need to remove or replace problematic patterns entirely
        
        # Strategy: Replace problematic % patterns with a safe placeholder before escaping
        # This prevents Python's formatter from trying to interpret them
        # Replace patterns like '% block ... %' or '%% block ... %%' with a safe placeholder
        context_text = re.sub(r'%+\s*block\s+\w+.*?%+', 'TEMPLATE_BLOCK_PLACEHOLDER', context_text, flags=re.IGNORECASE | re.DOTALL)
        # Replace any remaining %% patterns (which would become %%%% and cause issues)
        context_text = re.sub(r'%%+', 'PERCENT_PLACEHOLDER', context_text)
        # Now escape remaining single %
        context_text = context_text.replace('%', '%%')
        # Restore the placeholders as safe text (no % characters, no curly braces that could be interpreted as placeholders)
        # Use square brackets to avoid any formatting issues
        context_text = context_text.replace('TEMPLATE_BLOCK_PLACEHOLDER', '[template syntax removed]')
        context_text = context_text.replace('PERCENT_PLACEHOLDER', '[percent sign]')
        # Escape any standalone curly braces in the context (but preserve {context} placeholder by doing this carefully)
        # We need to escape { and } that are NOT part of {context} or other known placeholders
        # Since we're using {context} as the placeholder, we need to be careful
        # For now, let's just escape any { or } that appear in the content itself
        # But we can't do a simple replace because it would break {context}
        # Actually, LangChain handles this - it only interprets {context} as a placeholder, not other {text}
        # So we don't need to escape curly braces at all
        
        # Build messages with conversation history
        # First, format the original prompt template to get the system message
        # Debug: Check if context_text contains problematic patterns before formatting
        if '% ' in context_text or '%' in context_text:
            print(f"   ⚠️  WARNING: Context contains % characters. Escaping them...")
            # Double-check escaping is applied
            context_text = context_text.replace('%', '%%')
            print(f"   ✓ Escaped context (length: {len(context_text)})")
        
        try:
            formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
        except KeyError as ke:
            # If we still get a KeyError, handle it
            error_key = str(ke)
            print(f"   ❌ KeyError during template formatting: {ke}")
            print(f"   Error key: {error_key}")
            print(f"   Context preview (first 500 chars): {context_text[:500]}")
            
            # Handle template block pattern errors specifically (check first)
            if "template block pattern" in error_key.lower() or ("block" in error_key.lower() and "template" in error_key.lower()):
                print(f"   🔍 Found template block pattern error - removing all template patterns...")
                # Remove all template block patterns more aggressively
                context_text = re.sub(r'%+\s*block\s+\w+.*?%+', '[template syntax removed]', context_text, flags=re.IGNORECASE | re.DOTALL)
                context_text = re.sub(r'%+\s*endblock\s*%+', '', context_text, flags=re.IGNORECASE)
                context_text = re.sub(r'%+\s*\w+.*?%+', '', context_text)  # Remove any remaining % pattern %
                # Escape all remaining % characters
                context_text = context_text.replace('%', '%%')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except Exception as e2:
                    print(f"   ⚠️  Still failing after aggressive cleanup: {e2}")
                    # Last resort: remove all % characters entirely
                    context_text = context_text.replace('%', '').replace('%%', '')
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            # Handle both '% ' and '%% ' errors
            elif "'%% '" in error_key or "%% " in context_text:
                print(f"   🔍 Found '%% ' pattern - replacing with placeholder...")
                # Replace %%  with a safe placeholder that won't be interpreted
                context_text = context_text.replace('%% ', 'PERCENT_SPACE ')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except:
                    # If that still fails, try removing the pattern entirely
                    context_text = context_text.replace('PERCENT_SPACE ', '')
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            elif "'% '" in error_key or '% ' in context_text:
                print(f"   🔍 Found '% ' in context, attempting to escape again...")
                context_text = context_text.replace('% ', '%% ')
                context_text = context_text.replace('%', '%%')
                print(f"   Retrying with re-escaped context...")
                formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
            else:
                # For any other KeyError, try aggressive cleanup
                print(f"   🔍 Unknown KeyError pattern ({error_key}) - attempting aggressive cleanup...")
                # Remove all template-like patterns
                context_text = re.sub(r'%+\s*\w+.*?%+', '', context_text)
                context_text = context_text.replace('%', '%%')
                try:
                    formatted_system_messages = self.prompt_template.format_messages(context=context_text, question="")
                except Exception as e2:
                    print(f"   ⚠️  Error persists after cleanup: {e2}")
                    # Last resort: remove all % characters
                    context_text = context_text.replace('%', '').replace('%%', '')
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
        try:
            answer = self.llm.invoke(conversation_prompt.format_messages()).content
            # Debug: Log the raw LLM output to see if it has spaces
            if answer:
                print(f"   🔍 DEBUG: Raw LLM answer (first 200 chars): {repr(answer[:200])}")
                print(f"   🔍 DEBUG: Answer has spaces: {' ' in answer}")
        except Exception as e:
            print(f"   ⚠️  Error getting LLM response: {e}")
            raise
        
        # Sanitize answer: Remove any template syntax that might have been output
        # This prevents errors from template syntax like '%% block ... %%' or '{%% ... %%}'
        try:
            answer = self._sanitize_answer(answer)
        except Exception as e:
            print(f"   ⚠️  Warning: Error during sanitization: {e}")
            # If sanitization fails, try to continue with original answer (might have template syntax)
            # But remove obvious problematic patterns manually
            if answer:
                answer = answer.replace('% block', '').replace('% endblock', '').replace('{%', '').replace('%}', '')
            else:
                answer = "I apologize, but I encountered an error processing the response. Please try asking your question again."
        
        # Ensure answer is a string and not empty
        if not answer or not isinstance(answer, str):
            answer = "I apologize, but I encountered an error generating the response. Please try asking your question again."
        
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

