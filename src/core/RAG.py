from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
import time
import logging

# Configure logging for RAG module
logger = logging.getLogger(__name__)

load_dotenv('.env.local')

# Initialize reranker (loaded once, cached for performance)
print("🔄 Loading reranking model (BAAI/bge-reranker-large)...")
reranker = CrossEncoder('BAAI/bge-reranker-large')
print("✅ Reranker loaded successfully!\n")

# Initialize LLM for query enhancement
llm_for_enhancement = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize LLM for answer generation (separate instance)
llm_for_answers = ChatOpenAI(model="gpt-4o", temperature=0)

def is_followup_question(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> bool:
    """
    Detect if a question is a follow-up to previous conversation or a new topic.
    
    Args:
        query: Current question
        conversation_history: Previous conversation history
    
    Returns:
        True if question appears to be a follow-up, False if it's a new topic
    """
    if not conversation_history:
        return False
    
    query_lower = query.lower().strip()
    
    # Strong indicators of follow-up questions
    followup_indicators = [
        "it", "that", "this", "they", "those", "them",  # Pronouns
        "the above", "the previous", "the last", "mentioned",  # References
        "also", "additionally", "furthermore", "more",  # Continuation
        "how do i", "can i", "what about", "what if",  # Vague questions
        "tell me more", "explain", "elaborate", "details"  # Request for more info
    ]
    
    # Check if query contains follow-up indicators
    has_followup_indicators = any(indicator in query_lower for indicator in followup_indicators)
    
    # Check if query is very short (likely a follow-up)
    is_short = len(query.split()) <= 5
    
    # Check if query starts with vague question words (likely follow-up)
    vague_starters = ["how", "what", "can", "will", "does", "is", "are", "do"]
    starts_vague = any(query_lower.startswith(starter + " ") for starter in vague_starters) and is_short
    
    # Get last user question to check topic similarity
    last_user_question = None
    for msg in reversed(conversation_history):
        if msg.get('role') == 'user':
            last_user_question = msg.get('content', '').lower()
            break
    
    # Check if current question shares keywords with last question (topic continuity)
    topic_similarity = False
    if last_user_question:
        # Extract key terms (words longer than 3 chars, not common words)
        common_words = {'what', 'how', 'the', 'is', 'are', 'can', 'do', 'does', 'will', 'this', 'that', 'it'}
        last_keywords = {w for w in last_user_question.split() if len(w) > 3 and w not in common_words}
        current_keywords = {w for w in query_lower.split() if len(w) > 3 and w not in common_words}
        
        # If they share keywords, likely same topic
        if last_keywords and current_keywords:
            topic_similarity = len(last_keywords & current_keywords) > 0
    
    # Question is a follow-up if:
    # 1. Has follow-up indicators, OR
    # 2. Is short and vague, OR
    # 3. Shares topic keywords with previous question
    is_followup = has_followup_indicators or (is_short and starts_vague) or topic_similarity
    
    logger.info(f"🔍 [FOLLOWUP] Query: '{query}'")
    logger.info(f"🔍 [FOLLOWUP] Has indicators: {has_followup_indicators}, Is short: {is_short}, Topic similarity: {topic_similarity}")
    logger.info(f"🔍 [FOLLOWUP] Detected as: {'FOLLOW-UP' if is_followup else 'NEW TOPIC'}")
    
    return is_followup


def enhance_query(query: str, enable_enhancement: bool = True, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Enhance query using OpenAI to expand acronyms, add synonyms, and improve retrieval.
    
    Args:
        query: Original user query
        enable_enhancement: Whether to enable query enhancement (default: True)
        conversation_history: Optional conversation history to resolve references like "it", "that", etc.
    
    Returns:
        Enhanced query string
    """
    if not enable_enhancement:
        return query
    
    # Build context from conversation history if provided
    # Only use history for enhancement if this appears to be a follow-up question
    history_context = ""
    use_history_for_enhancement = False
    
    if conversation_history:
        is_followup = is_followup_question(query, conversation_history)
        use_history_for_enhancement = is_followup
        
        if use_history_for_enhancement:
            recent_history = conversation_history[-4:]  # Last 4 messages
            history_parts = []
            for msg in recent_history:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    history_parts.append(f"User: {content}")
                elif role == 'assistant':
                    history_parts.append(f"Assistant: {content[:200]}...")  # Truncate long answers
            
            if history_parts:
                history_context = f"""
PREVIOUS CONVERSATION CONTEXT:
{chr(10).join(history_parts)}

IMPORTANT: This appears to be a FOLLOW-UP question. Use the previous conversation to understand what the user is referring to with words like "it", "that", "this", "the above", etc.
Expand the current query to include the topic from previous conversation.
"""
        else:
            logger.info("🔍 [ENHANCE] Question appears to be a NEW TOPIC - not using history for enhancement")
    
    enhancement_prompt = f"""You are a query enhancement assistant for a RAG (Retrieval-Augmented Generation) system.
Your task is to enhance the user's query to improve document retrieval.

CRITICAL: You MUST expand acronyms and add related terms. This is essential for finding relevant documents.

{history_context}

CURRENT QUERY: {query}

ENHANCEMENT RULES:
1. ALWAYS expand acronyms - this is MANDATORY:
   - If query contains "BXGX" → expand to "Buy X Get X promotions" or "Buy X Get Y offers"
   - If query contains "BOGO" → expand to "Buy One Get One promotions"
   - If query contains "API" → expand to "application programming interface"
   - If query contains "CMS" → expand to "content management system"
   - If query contains "B2B" → expand to "business to business"
   - If query contains "B2C" → expand to "business to consumer"

2. Add synonyms and related terms:
   - For BXGX/BOGO: add "promotion", "discount", "offer", "deal", "coupon"
   - For setup/configuration: add "configuration", "settings", "options", "steps"

3. If query has references ("it", "that", "this"), use conversation context to resolve them

4. Keep the original question structure but make it more searchable

5. Return ONLY the enhanced query text - no explanations, no quotes, no markdown formatting

EXAMPLES:
- "What is BXGX?" → "What is Buy X Get X promotions or BXGX offers discounts"
- "How do I set it up?" (after BXGX question) → "How do I set up Buy X Get X promotions BXGX configuration steps"

ENHANCED QUERY:"""

    try:
        response = llm_for_enhancement.invoke(enhancement_prompt)
        enhanced = response.content.strip()
        
        # Remove quotes if LLM wrapped the response
        if enhanced.startswith('"') and enhanced.endswith('"'):
            enhanced = enhanced[1:-1]
        if enhanced.startswith("'") and enhanced.endswith("'"):
            enhanced = enhanced[1:-1]
        
        # Log enhancement result
        logger.info(f"🔧 [ENHANCE] Original: {query}")
        logger.info(f"🔧 [ENHANCE] Enhanced: {enhanced}")
        
        # Check if enhancement actually expanded acronyms or added useful terms
        query_upper = query.upper()
        enhanced_upper = enhanced.upper()
        
        # Check if common acronyms were expanded
        acronyms_expanded = False
        acronym_mappings = {
            'BXGX': ['BUY X GET X', 'BUY X GET Y'],
            'BOGO': ['BUY ONE GET ONE'],
            'API': ['APPLICATION PROGRAMMING INTERFACE'],
            'CMS': ['CONTENT MANAGEMENT SYSTEM'],
            'B2B': ['BUSINESS TO BUSINESS'],
            'B2C': ['BUSINESS TO CONSUMER']
        }
        
        for acronym, expansions in acronym_mappings.items():
            if acronym in query_upper:
                for expansion in expansions:
                    if expansion in enhanced_upper:
                        acronyms_expanded = True
                        logger.info(f"✅ [ENHANCE] Detected {acronym} expansion to {expansion}")
                        break
        
        # Check if synonyms/related terms were added
        synonyms_added = any(term in enhanced_upper for term in ['PROMOTION', 'DISCOUNT', 'OFFER', 'DEAL', 'COUPON', 'CONFIGURATION', 'SETUP', 'SETTINGS'])
        
        # Safety check: only reject if enhancement is way too short or way too long AND doesn't add value
        if len(enhanced) < len(query) * 0.3:
            logger.warning(f"⚠️  [ENHANCE] Enhancement too short ({len(enhanced)} vs {len(query)}), using original")
            return query
        
        # If enhancement expanded acronyms or added synonyms, accept it even if longer
        if acronyms_expanded or synonyms_added:
            logger.info(f"✅ [ENHANCE] Enhancement accepted (acronyms expanded: {acronyms_expanded}, synonyms added: {synonyms_added})")
            return enhanced
        
        # If enhanced is same as original (after trimming), still return it (might have resolved references)
        if enhanced.lower().strip() == query.lower().strip():
            logger.info(f"🔧 [ENHANCE] Enhanced query same as original, but may have resolved references")
            return enhanced
        
        # If enhancement is reasonable length (not more than 5x), accept it
        if len(enhanced) <= len(query) * 5:
            logger.info(f"✅ [ENHANCE] Enhancement accepted (reasonable length: {len(enhanced)} vs {len(query)})")
            return enhanced
        
        # Only reject if it's extremely long and doesn't add value
        logger.warning(f"⚠️  [ENHANCE] Enhancement too long ({len(enhanced)} vs {len(query)}), using original")
        return query
    except Exception as e:
        logger.error(f"⚠️  Query enhancement failed: {e}")
        logger.error("   Using original query...")
        return query


def get_windowed_chunks(
    chunks: List[Document], 
    scores: List[float],
    vectorstore: Chroma,  # Type hint for vectorstore
    window_size_chars: int = 3500,  # ±3500 chars (≈4-5 chunks of 800) - increased for better coverage
    min_window_gap: int = 500,      # Minimum gap to create separate windows
    max_total_chunks: int = 30      # Safety limit (increased to accommodate more windows)
) -> List[Document]:
    """
    Get chunks in windows around top reranked chunks for focused, complete context.
    This ensures step completeness while reducing context size.
    
    Args:
        chunks: Retrieved chunks after reranking
        scores: Reranking scores for chunks
        vectorstore: ChromaDB vectorstore instance
        window_size_chars: Window size in characters around each chunk (default: 2100)
        min_window_gap: Minimum gap between windows to keep them separate (default: 500)
        max_total_chunks: Maximum total chunks to retrieve (default: 25)
    
    Returns:
        List of chunks within windows, sorted by position
    """
    from collections import defaultdict
    
    # 1. Group chunks by file with their positions and scores
    file_chunks = defaultdict(list)
    
    for chunk, score in zip(chunks, scores):
        file_path = chunk.metadata.get('file', 'unknown')
        start_idx = chunk.metadata.get('start_index', 0)
        file_chunks[file_path].append({
            'chunk': chunk,
            'position': start_idx,
            'score': score
        })
    
    print(f"\n📁 Creating windows around top chunks from {len(file_chunks)} source files:")
    
    all_windowed_chunks = []
    
    # 2. Process each file
    for file_path, chunk_list in file_chunks.items():
        file_name = os.path.basename(file_path)
        
        # Sort chunks by position
        chunk_list.sort(key=lambda x: x['position'])
        
        # 3. Create windows around each chunk
        windows = []
        for chunk_info in chunk_list:
            center_pos = chunk_info['position']
            window_start = max(0, center_pos - window_size_chars)
            window_end = center_pos + window_size_chars
            windows.append({
                'start': window_start,
                'end': window_end,
                'score': chunk_info['score'],
                'center': center_pos
            })
        
        # 4. Merge overlapping windows
        if not windows:
            continue
            
        windows.sort(key=lambda x: x['start'])
        merged_windows = []
        current_window = windows[0].copy()
        
        for next_window in windows[1:]:
            # If windows overlap or are close (within min_window_gap), merge them
            if next_window['start'] <= current_window['end'] + min_window_gap:
                # Merge: extend end position, keep highest score
                current_window['end'] = max(current_window['end'], next_window['end'])
                current_window['score'] = max(current_window['score'], next_window['score'])
            else:
                # No overlap, save current and start new
                merged_windows.append(current_window)
                current_window = next_window.copy()
        
        merged_windows.append(current_window)
        
        print(f"\n  📄 {file_name}:")
        print(f"     Top chunks: {len(chunk_list)}, Windows created: {len(merged_windows)}")
        
        # 5. Retrieve chunks within merged windows
        file_windowed_chunks = []
        
        try:
            # Get all chunks from this file
            results = vectorstore._collection.get(
                where={"file": file_path}
            )
            
            if results and len(results.get('ids', [])) > 0:
                # Build list of all chunks with positions
                all_file_chunks = []
                for i in range(len(results['ids'])):
                    chunk_doc = Document(
                        page_content=results['documents'][i],
                        metadata=results['metadatas'][i]
                    )
                    start_idx = results['metadatas'][i].get('start_index', 0)
                    all_file_chunks.append((start_idx, chunk_doc))
                
                # Sort by position
                all_file_chunks.sort(key=lambda x: x[0])
                
                # Get chunks within each window
                for window in merged_windows:
                    window_chunks = [
                        (pos, chunk) for pos, chunk in all_file_chunks
                        if window['start'] <= pos <= window['end']
                    ]
                    file_windowed_chunks.extend(window_chunks)
                    
                    print(f"     Window [{window['start']}-{window['end']}]: {len(window_chunks)} chunks")
                
                # Remove duplicates (chunks might be in multiple windows)
                seen_positions = set()
                unique_chunks = []
                for pos, chunk in file_windowed_chunks:
                    if pos not in seen_positions:
                        seen_positions.add(pos)
                        unique_chunks.append(chunk)
                
                # Filter out chunks that are mostly whitespace or have very low content
                filtered_chunks = []
                for chunk in unique_chunks:
                    content = chunk.page_content.strip()
                    # Skip if mostly whitespace or too short
                    if len(content) < 50:  # Less than 50 chars after stripping
                        continue
                    # Count actual words (non-whitespace sequences)
                    words = content.split()
                    if len(words) < 10:  # Less than 10 words
                        continue
                    # Check if content is mostly whitespace characters
                    non_whitespace = sum(1 for c in content if not c.isspace() and c != '\ufffc')
                    if non_whitespace < 30:  # Less than 30 non-whitespace chars
                        continue
                    filtered_chunks.append(chunk)
                
                if len(filtered_chunks) < len(unique_chunks):
                    print(f"     ⚠️  Filtered out {len(unique_chunks) - len(filtered_chunks)} empty/low-content chunks")
                
                all_windowed_chunks.extend(filtered_chunks)
                print(f"     ✅ Total unique chunks: {len(filtered_chunks)}")
            else:
                # Fallback: use chunks we already have
                existing_chunks = [chunk_info['chunk'] for chunk_info in chunk_list]
                all_windowed_chunks.extend(existing_chunks)
                print(f"     ⚠️  Using {len(existing_chunks)} existing chunks (no additional retrieval)")
        except Exception as e:
            # Fallback: use chunks we already have
            existing_chunks = [chunk_info['chunk'] for chunk_info in chunk_list]
            all_windowed_chunks.extend(existing_chunks)
            print(f"     ⚠️  Error: {e}, using {len(existing_chunks)} existing chunks")
    
    # 6. Sort all chunks by file, then by position
    all_windowed_chunks.sort(
        key=lambda x: (
            x.metadata.get('file', ''),
            x.metadata.get('start_index', 0)
        )
    )
    
    # 7. Apply safety limit if needed
    if len(all_windowed_chunks) > max_total_chunks:
        print(f"\n⚠️  Limiting total chunks to {max_total_chunks} (had {len(all_windowed_chunks)})")
        # Keep chunks from highest-scoring files first
        all_windowed_chunks = all_windowed_chunks[:max_total_chunks]
    
    print(f"\n📦 Total windowed chunks: {len(all_windowed_chunks)}\n")
    
    return all_windowed_chunks


def build_context(chunks: List[Document]) -> str:
    """
    Combine retrieved chunks into formatted context for LLM.
    Uses improved formatting with source attribution and position info.
    
    Args:
        chunks: List of retrieved document chunks
    
    Returns:
        Formatted context string with chunk content and metadata
    """
    context_parts = []
    current_file = None
    file_chunk_count = {}
    
    # Count chunks per file
    for chunk in chunks:
        file_name = chunk.metadata.get('file', 'unknown')
        file_chunk_count[file_name] = file_chunk_count.get(file_name, 0) + 1
    
    # Format chunks with source attribution
    source_counter = {}
    for doc in chunks:
        file_path = doc.metadata.get('file', 'unknown')
        file_name = os.path.basename(file_path)
        
        # Track source number (unique per file)
        if file_path not in source_counter:
            source_counter[file_path] = len(source_counter) + 1
        
        source_num = source_counter[file_path]
        start_idx = doc.metadata.get('start_index', '?')
        total_chunks = file_chunk_count.get(file_path, '?')
        
        # Enhanced formatting with metadata
        chunk_text = f"[Source {source_num}: {file_name} | Position: {start_idx} | Chunk {list(source_counter.keys()).index(file_path) + 1}/{total_chunks}]\n{doc.page_content}\n"
        context_parts.append(chunk_text)
    
    return "\n---\n".join(context_parts)


def create_answer_prompt(query: str, context: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Create prompt for LLM to generate answer from context.
    
    Args:
        query: User's current question
        context: Formatted context from retrieved chunks
        conversation_history: Optional conversation history for context (last 4-6 messages)
    
    Returns:
        Complete prompt for LLM
    """
    # Format conversation history if provided (limit to last 4 messages to avoid overwhelming)
    history_section = ""
    if conversation_history:
        # Check if this is a follow-up or new topic
        is_followup = is_followup_question(query, conversation_history)
        
        logger.info(f"📖 [RAG] Using conversation history: {len(conversation_history)} total messages")
        logger.info(f"📖 [RAG] Question type: {'FOLLOW-UP' if is_followup else 'NEW TOPIC'}")
        
        recent_history = conversation_history[-4:]  # Last 4 messages (2 exchanges)
        logger.info(f"📖 [RAG] Using last {len(recent_history)} messages for context")
        history_parts = []
        for msg in recent_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                history_parts.append(f"User: {content}")
            elif role == 'assistant':
                history_parts.append(f"Assistant: {content}")
        
        if history_parts:
            logger.debug(f"📖 [RAG] History context preview: {chr(10).join(history_parts[:2])}...")
            
            if is_followup:
                history_section = f"""
PREVIOUS CONVERSATION (for context only - do not answer these, they are just for reference):
{chr(10).join(history_parts)}

IMPORTANT: This is a FOLLOW-UP question. The previous conversation is provided to help you understand references like "it", "that", "the above", etc. 
Your PRIMARY task is to answer the CURRENT QUESTION below using the RETRIEVED CONTEXT. 
Use the previous conversation ONLY to resolve references, not to repeat previous answers.
"""
            else:
                history_section = f"""
PREVIOUS CONVERSATION (for context only - this appears to be a NEW TOPIC):
{chr(10).join(history_parts)}

IMPORTANT: This appears to be a NEW TOPIC question, different from the previous conversation.
The previous conversation is provided ONLY for general context, but you should focus on answering the CURRENT QUESTION using the RETRIEVED CONTEXT.
Do NOT relate this question to the previous conversation unless the user explicitly asks about it.
"""
    else:
        logger.info("📖 [RAG] No conversation history provided")
    
    prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant. 
Your job is to answer the user's CURRENT question ONLY using the information provided in the RETRIEVED CONTEXT below. 
The context comes from multiple video transcripts and documentation chunks.
{history_section}
RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:

1. PRIMARY FOCUS: Answer the CURRENT QUESTION below using the RETRIEVED CONTEXT.
   - The previous conversation (if shown) is ONLY for understanding references and context.
   - Do NOT let previous conversation override or distract from the current question.
   - If the current question is a follow-up, use previous context to understand references, but base your answer on RETRIEVED CONTEXT.

2. Treat each item in the RETRIEVED CONTEXT as a separate "chunk" of information. 
   A chunk may be a paragraph, transcript segment, bullet list, or section from documentation.

3. Use EVERY relevant chunk in the RETRIEVED CONTEXT to form your answer.
   - Identify all chunks that relate to the CURRENT question.
   - Combine information from all relevant chunks.
   - Do NOT omit important details found in any chunk.
   - When multiple options, choices, or field values exist, list ALL of them.
   - Include ALL field descriptions and configuration options mentioned in the context.
   - CRITICAL: If the context mentions specific features, buttons, toggles, or steps, you MUST include them in your answer.

4. Synthesize the information into ONE coherent, non-contradicting explanation.
   - If multiple chunks repeat the same idea, merge them.
   - If chunks contain different steps, combine them into a complete end-to-end process.
   - Maintain the correct chronological order of steps as described in the chunks.
   - Do NOT summarize away important details or options.
   - CRITICAL: For workflow questions, include EVERY step mentioned in the context, even if they seem minor.

5. When the question asks for instructions, explanations, or setup steps:
   - Write the answer like a clear tutorial or guide.
   - Use headings and numbered steps.
   - Add short clarifying notes when necessary for readability.
   - Include ALL available options, choices, and field values mentioned in the context.
   - CRITICAL: List ALL steps in the exact order they appear in the context.
   - CRITICAL: Include ALL buttons, toggles, dropdowns, and options mentioned (e.g., "Apply All", "Apply Only Options", "Payment Request toggle", etc.).
   - CRITICAL: If the context mentions "you can" or "you have the option to", include it as a step or option.

6. Do NOT cite sources in your answer. Write the answer naturally without any source citations like (Source 1), (Source 2), etc.

7. Do NOT include a "Sources Used:" section at the end

8. If the answer is NOT present in the RETRIEVED CONTEXT, say exactly:
   "I don't have enough information to answer this question."

9. Do NOT add any outside knowledge. Stay strictly within the provided RETRIEVED CONTEXT.

10. Never contradict the context. If two chunks conflict:
   - Prefer explicit instructions over summaries.
   - Prefer more complete steps over partial steps.
   - Prefer updated or detailed information when identifiable.

11. Keep the answer well-organized, complete, and easy to follow. Prioritize completeness over brevity. Include all relevant details, options, and field descriptions mentioned in the context.

CURRENT QUESTION:
{query}

ANSWER:
"""


    
    return prompt


def generate_answer(query: str, chunks: List[Document], llm, conversation_history: Optional[List[Dict[str, str]]] = None, max_retries: int = 3) -> str:
    """
    Generate answer from retrieved chunks using LLM with error handling.
    
    Args:
        query: User's current question
        chunks: Retrieved document chunks
        llm: LLM instance for answer generation
        conversation_history: Optional conversation history for context
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Generated answer string
    """
    import time
    
    # Build context
    context = build_context(chunks)
    
    # Create prompt with conversation history
    prompt = create_answer_prompt(query, context, conversation_history)
    
    # Generate answer with retry logic
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            # Validate answer
            if not answer or len(answer) < 10:
                raise ValueError("Answer too short or empty")
            
            return answer
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                error_msg = f"I apologize, but I encountered an error generating the answer: {str(e)}. Please try again."
                print(f"⚠️  Answer generation failed after {max_retries} attempts: {e}")
                return error_msg
            else:
                # Wait before retry
                print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
    
    return "I apologize, but I encountered an error generating the answer. Please try again."


def vector_store(index_path: str = "chroma_index", collection_name: str = "langchain_onprintshop_chroma"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(
    persist_directory=index_path,
    embedding_function=embeddings,
    collection_name=collection_name
    )
    return vectorstore

def query_vector_store(
    query: str,
    vectorstore: Chroma,  # Pass vectorstore as parameter instead of creating new one
    initial_k: int = 50,  # How many to retrieve initially
    final_k: int = 12,     # Number of top chunks to create windows around (increased from 8)
    use_reranking: bool = True,
    enhance_query_flag: bool = True,  # Enable query enhancement
    generate_answer_flag: bool = True,  # Enable answer generation
    window_size_chars: int = 3500,  # Window size around chunks (increased from 2100 to ±3500 chars ≈4-5 chunks)
    max_total_chunks: int = 30,  # Maximum total chunks (increased from 25 to accommodate more windows)
    conversation_history: Optional[List[Dict[str, str]]] = None  # Conversation history for enhancement and answer generation
):
    """
    Universal RAG query with reranking and query enhancement.
    
    Args:
        query: User's question
        vectorstore: ChromaDB vectorstore instance to use for retrieval
        initial_k: Number of candidates to retrieve initially (default: 50)
        final_k: Final number of chunks after reranking (default: 8)
        use_reranking: Whether to use reranking (default: True)
        enhance_query_flag: Whether to enhance query with LLM (default: True)
    """
    print("=" * 80)
    print("🚀 STARTING RAG QUERY WITH RERANKING")
    print("=" * 80)
    original_query = query  # Store original query for answer generation
    print(f"Original Query: {query}\n")
    
    # ============================================================================
    # STEP 0: QUERY ENHANCEMENT (if enabled)
    # ============================================================================
    if enhance_query_flag:
        print("-" * 80)
        print("STEP 0: QUERY ENHANCEMENT")
        print("-" * 80)
        print("🔄 Enhancing query with OpenAI to expand acronyms and add synonyms...")
        if conversation_history:
            print(f"   Using {len(conversation_history)} conversation history messages for context")
        
        enhanced_query = enhance_query(query, enable_enhancement=True, conversation_history=conversation_history)
        
        if enhanced_query != query:
            print(f"✅ Query enhanced!")
            print(f"   Original: {query}")
            print(f"   Enhanced:  {enhanced_query}\n")
            query = enhanced_query  # Use enhanced query for retrieval
        else:
            print(f"   ⚠️  Query not enhanced (same as original)")
            print(f"   Original: {query}")
            print(f"   Enhanced: {enhanced_query}\n")
    else:
        print("(Query enhancement disabled)\n")
    
    # Use provided vectorstore
    doc_count = vectorstore._collection.count()
    print(f"📊 Vector Store: {doc_count} documents loaded\n")
    
    # ============================================================================
    # STEP 1: INITIAL RETRIEVAL (Broad Search)
    # ============================================================================
    print("-" * 80)
    print("STEP 1: INITIAL RETRIEVAL (Broad Similarity Search)")
    print("-" * 80)
    print(f"🔍 Retrieving top {initial_k} candidates using similarity search...")
    
    candidates_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)
    
    print(f"✅ Retrieved {len(candidates_with_scores)} candidates\n")
    print("Top 10 candidates (before reranking):")
    for i, (doc, score) in enumerate(candidates_with_scores[:10], 1):
        file_name = os.path.basename(doc.metadata.get('file', 'unknown'))
        print(f"  {i:2d}. Score: {score:.4f} | File: {file_name[:60]}...")
    
    # Extract documents
    documents = [doc for doc, score in candidates_with_scores]
    initial_scores = [score for doc, score in candidates_with_scores]
    
    print(f"\n📝 Total documents prepared for reranking: {len(documents)}\n")
    
    # ============================================================================
    # STEP 2: RERANKING (if enabled)
    # ============================================================================
    if use_reranking:
        print("-" * 80)
        print("STEP 2: RERANKING (Scoring Query-Document Relevance)")
        print("-" * 80)
        print("🔄 Preparing query-document pairs for reranking...")
        
        # Prepare pairs: (query, document_content) for each document
        pairs = [[query, doc.page_content] for doc in documents]
        print(f"✅ Prepared {len(pairs)} pairs\n")
        
        print("🔄 Running reranker on all candidates...")
        print("   (This may take 1-3 seconds on CPU, ~200-500ms on GPU)")
        
        # Rerank all candidates
        rerank_scores = reranker.predict(pairs)
        
        print(f"✅ Reranking complete! Scored {len(rerank_scores)} documents\n")
        
        # Combine documents with their reranking scores
        scored_docs = list(zip(documents, rerank_scores, initial_scores))
        
        # Sort by reranking score (highest = most relevant)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 15 documents after reranking:")
        print("Rank | Rerank Score | Initial Score | File")
        print("-" * 80)
        for i, (doc, rerank_score, initial_score) in enumerate(scored_docs[:15], 1):
            file_name = os.path.basename(doc.metadata.get('file', 'unknown'))
            print(f" {i:2d}  |    {rerank_score:.4f}   |    {initial_score:.4f}   | {file_name[:50]}...")
        
        # Select top K documents for file identification
        top_reranked_chunks = [doc for doc, rerank_score, _ in scored_docs[:final_k]]
        top_reranked_scores = [rerank_score for _, rerank_score, _ in scored_docs[:final_k]]
        
        print(f"\n✅ Selected top {final_k} chunks after reranking")
        print(f"🔄 Creating windows around top chunks for focused, complete context...\n")
        
        # Get windowed chunks around top reranked chunks (smart approach)
        final_chunks = get_windowed_chunks(
            top_reranked_chunks,
            top_reranked_scores,
            vectorstore,
            window_size_chars=window_size_chars,
            min_window_gap=500,      # Minimum gap between windows
            max_total_chunks=max_total_chunks
        )
        final_scores = [1.0] * len(final_chunks)  # All chunks from top files are included
        
        # Track source files from final chunks (after filtering) - only files that actually contributed
        source_files_used = list(dict.fromkeys([
            os.path.basename(doc.metadata.get('file', 'unknown'))
            for doc in final_chunks
        ]))
        
    else:
        # No reranking - just use initial results
        print("-" * 80)
        print("STEP 2: SKIPPED (Reranking disabled)")
        print("-" * 80)
        final_chunks = documents[:final_k]
        final_scores = initial_scores[:final_k]
        
        print(f"✅ Selected top {final_k} documents from initial retrieval\n")
        
        # Track source files from final chunks (only files that actually contributed)
        source_files_used = list(dict.fromkeys([
            os.path.basename(doc.metadata.get('file', 'unknown'))
            for doc in final_chunks
        ]))
    
    # ============================================================================
    # STEP 3: DISPLAY FINAL RESULTS
    # ============================================================================
    print("-" * 80)
    print("STEP 3: FINAL RESULTS")
    print("-" * 80)
    print(f"📋 Final {len(final_chunks)} chunks selected:\n")
    
    for i, (doc, score) in enumerate(zip(final_chunks, final_scores), 1):
        file_name = os.path.basename(doc.metadata.get('file', 'unknown'))
        print(f"{'='*80}")
        print(f"RESULT {i} (Score: {score:.4f})")
        print(f"{'='*80}")
        print(f"File: {file_name}")
        print(f"Content Preview: {doc.page_content[:300]}...")
        print()
    
    # ============================================================================
    # STEP 4: ANSWER GENERATION (if enabled)
    # ============================================================================
    answer = None
    if generate_answer_flag:
        # print("-" * 80)
        # print("STEP 4: ANSWER GENERATION")
        # print("-" * 80)
        # print("🤖 Generating answer from retrieved context...")
        
        answer = generate_answer(original_query, final_chunks, llm_for_answers, conversation_history)
        
        # print("✅ Answer generated!\n")
        # print("=" * 80)
        # print("FINAL ANSWER")
        # print("=" * 80)
        # print(answer)
        # print("=" * 80)
        # print()
    else:
        print("-" * 80)
        print("STEP 4: SKIPPED (Answer generation disabled)")
        print("-" * 80)
        print()
    
    # print("=" * 80)
    # print("✅ QUERY PROCESSING COMPLETE")
    # print("=" * 80)
    
    # Return comprehensive results with only sources that were actually used
    # (sources from top reranked chunks, not all chunks in final context)
    
    # If answer indicates insufficient information, don't include sources
    if answer and generate_answer_flag:
        answer_lower = answer.lower().strip()
        insufficient_info_phrases = [
            "i don't have enough information",
            "i don't have sufficient information",
            "i cannot answer",
            "not enough information",
            "i don't have information",
            "unable to answer",
            "cannot provide an answer"
        ]
        if any(phrase in answer_lower for phrase in insufficient_info_phrases):
            source_files_used = []  # No sources if answer couldn't be generated
    
    return {
        'answer': answer,
        'chunks': final_chunks,
        'scores': final_scores,
        'sources': source_files_used,  # Only files from top reranked chunks (actually used)
        'query': original_query,
        'enhanced_query': query if enhance_query_flag and query != original_query else None
    }


class VideoRAGQuery:
    """Wrapper class for RAG query system compatible with router.py interface."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.vectorstore = vector_store(
            index_path="chroma_index", 
            collection_name="langchain_onprintshop_chroma"
        )
        self.video_filename_map = self._load_video_filename_map()
        self.reranker = reranker  # Expose reranker for stats endpoint
    
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
    
    def _format_sources(self, source_files: List[str], chunks: List[Document]) -> List[Dict]:
        """Format source files into the expected format with metadata.
        
        Note: source_file should be the JSON filename (basename), not the original filename.
        The router will convert it using _get_original_video_filename.
        """
        sources = []
        seen_files = set()
        
        # Create a mapping from file basenames to chunks (source_files are basenames)
        file_to_chunks = {}
        for chunk in chunks:
            file_path = chunk.metadata.get('file', 'unknown')
            file_basename = os.path.basename(file_path)
            if file_basename not in file_to_chunks:
                file_to_chunks[file_basename] = []
            file_to_chunks[file_basename].append(chunk)
        
        for source_file in source_files:
            if source_file in seen_files:
                continue
            seen_files.add(source_file)
            
            # Find chunks from this source file (source_file is already a basename)
            matching_chunks = file_to_chunks.get(source_file, [])
            
            # Get metadata from first chunk if available
            video_name = "Unknown"
            video_id = "Unknown"
            content_type = "Unknown"
            
            if matching_chunks:
                first_chunk = matching_chunks[0]
                video_name = first_chunk.metadata.get('video_name', 'Unknown')
                video_id = first_chunk.metadata.get('video_id', 'Unknown')
                content_type = first_chunk.metadata.get('content_type', 'Unknown')
            
            # Keep JSON filename (basename) - router will convert it
            sources.append({
                "video_name": video_name,
                "video_id": video_id,
                "content_type": content_type,
                "source_file": source_file  # JSON filename (basename), router will convert
            })
        
        return sources
    
    def _build_query_with_history(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build query for retrieval - pass question as-is.
        History will be used in query enhancement to resolve references.
        """
        # Return question as-is - enhancement will use history to resolve references
        # This keeps retrieval focused but allows enhancement to add context
        return question
    
    def query(self, question: str, return_sources: bool = True, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
        """
        Query the RAG system and return answer with sources.
        
        Args:
            question: User's current question
            return_sources: Whether to return source references
            conversation_history: Optional conversation history for context (passed to LLM, not retrieval)
        
        Returns:
            Dictionary with 'answer', 'question', and optionally 'sources'
        """
        # Use ONLY the current question for retrieval (enhancement will use history to resolve references)
        clean_question = self._build_query_with_history(question, conversation_history)
        
        # Query the vector store - pass history for both enhancement and answer generation
        results = query_vector_store(
            query=clean_question,  # Question for retrieval (enhancement will use history)
            vectorstore=self.vectorstore,  # Use the initialized vectorstore
            initial_k=50,
            final_k=12,
            use_reranking=True,
            enhance_query_flag=True,
            generate_answer_flag=True,
            window_size_chars=3500,
            max_total_chunks=30,
            conversation_history=conversation_history  # Pass history for enhancement AND answer generation
        )
        
        # Format sources
        sources = None
        if return_sources and results.get('sources'):
            sources = self._format_sources(results['sources'], results.get('chunks', []))
        
        return {
            'answer': results.get('answer', ''),
            'question': question,  # Return original question, not enhanced
            'sources': sources
        }
    
    def query_stream(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None):
        """
        Stream query response from the RAG system.
        
        Args:
            question: User's current question
            conversation_history: Optional conversation history for context (passed to LLM, not retrieval)
        
        Yields:
            Dictionary chunks with 'type' field ('content', 'sources', 'done')
        """
        # Use ONLY the current question for retrieval (enhancement will use history to resolve references)
        clean_question = self._build_query_with_history(question, conversation_history)
        
        # Query the vector store - pass history for both enhancement and answer generation
        results = query_vector_store(
            query=clean_question,  # Question for retrieval (enhancement will use history)
            vectorstore=self.vectorstore,  # Use the initialized vectorstore
            initial_k=50,
            final_k=12,
            use_reranking=True,
            enhance_query_flag=True,
            generate_answer_flag=True,
            window_size_chars=3500,
            max_total_chunks=30,
            conversation_history=conversation_history  # Pass history for enhancement AND answer generation
        )
        
        answer = results.get('answer', '')
        
        # Stream answer in chunks (simulate streaming)
        chunk_size = 50  # Characters per chunk
        for i in range(0, len(answer), chunk_size):
            chunk_text = answer[i:i + chunk_size]
            yield {
                'type': 'content',
                'content': chunk_text
            }
            time.sleep(0.01)  # Small delay to simulate streaming
        
        # Send sources
        if results.get('sources'):
            sources = self._format_sources(results['sources'], results.get('chunks', []))
            yield {
                'type': 'sources',
                'sources': sources
            }
        
        # Send done signal
        yield {
            'type': 'done'
        }


if __name__ == "__main__":
    query = "Tell me what BXGX is and how do I set it up?"
    query = "I need to add an offline order for one of my customers. What is the process of doing this from start to finish?"
    # query = "I need to send a quote from the admin panel. How do I do this from start to finish in OnPrintShop?"
    # Initialize vectorstore for testing
    test_vectorstore = vector_store(index_path="chroma_index", collection_name="langchain_onprintshop_chroma")
    results = query_vector_store(
        query=query,
        vectorstore=test_vectorstore,  # Pass vectorstore
        initial_k=50,              # Retrieve 50 candidates initially
        final_k=12,                # Create windows around top 12 chunks (increased from 8)
        use_reranking=True,        # Enable reranking
        enhance_query_flag=True,   # Enable query enhancement
        generate_answer_flag=True, # Enable answer generation
        window_size_chars=3500,    # Window size: ±3500 chars (≈4-5 chunks) - increased for better coverage
        max_total_chunks=30        # Maximum total chunks (increased from 25)
    )
    
    # Display results
    if results['answer']:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Query: {results['query']}")
        if results['enhanced_query']:
            print(f"Enhanced Query: {results['enhanced_query']}")
        print(f"Sources Used: {len(results['sources'])} documents")
        print(f"Answer Length: {len(results['answer'])} characters")
        print(f"Answer: {results['answer']}")
        print(f"Sources: {results['sources']}")

