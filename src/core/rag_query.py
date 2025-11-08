"""
RAG Query System - Ask questions about your video content.
Uses OpenAI GPT-4o-mini with your embedded video insights.
"""
import os
import json
from typing import List, Dict
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
OPENAI_MODEL = "gpt-4o"  # Change to "gpt-4o" for better quality
TEMPERATURE = 0.0  # Lower temperature for more consistent, factual answers

# Re-ranking Configuration
USE_RERANKING = True  # Enable re-ranking for better relevance
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast and effective re-ranker
FINAL_CHUNKS = 5  # Number of chunks to use after re-ranking


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
        """Get original video filename with extension from JSON insights filename."""
        # First try to find in mapping
        if json_filename in self.video_filename_map:
            return self.video_filename_map[json_filename]
        
        # Fallback: extract base name and try common video extensions
        # JSON files are named like "Video Name_insights.json"
        base_name = json_filename.replace('_insights.json', '')
        
        # Try to find the original file with common video extensions
        common_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        results_dir = Path('data/results')
        
        if results_dir.exists():
            for ext in common_extensions:
                potential_file = results_dir / f"{base_name}{ext}"
                if potential_file.exists():
                    return f"{base_name}{ext}"
        
        # If not found, default to .mp4 (most common)
        return f"{base_name}.mp4"
    
    def __init__(self):
        """Initialize the RAG system."""
        # Load video filename mapping from batch_progress.json if available
        self.video_filename_map = self._load_video_filename_map()
        
        # Load vector store
        if not os.path.exists(VECTOR_DB_DIR):
            raise ValueError(f"Vector database not found: {VECTOR_DB_DIR}")
        
        print(f"Loading vector database from: {VECTOR_DB_DIR}...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store - Chroma will use the default collection
        self.vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        
        # Debug: Check what collections exist and their counts
        try:
            import chromadb
            client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            collections = client.list_collections()
            print(f"   Found {len(collections)} collection(s) in database")
            for col in collections:
                count = col.count()
                print(f"   - Collection '{col.name}': {count} documents")
        except Exception as e:
            print(f"   Warning: Could not list collections: {e}")
        
        # Try to get collection count
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            print(f"   Using collection '{collection.name}' with {count} documents")
        except Exception as e:
            print(f"   Warning: Could not get collection count: {e}")
            count = "unknown"
        
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
        
        # Create retriever - retrieve more chunks to ensure we get video sources
        # Increased k significantly to ensure video files are included
        initial_k = 30  # Get more chunks to ensure video sources are included
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": initial_k}  # Retrieve more chunks to ensure video sources
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
        
        # Create custom prompt template - use only what's in the context, extract what is available
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based ONLY on the provided context from video transcripts and insights.

CRITICAL RULES - STRICTLY FOLLOW THESE:
1. **ONLY USE PROVIDED CONTEXT**: Your answer MUST be based EXCLUSIVELY on the information provided in the context below. Do NOT add any information that is not explicitly stated in the context.

2. **NO ASSUMPTIONS OR INFERENCES**: Do NOT make assumptions, inferences, or add information that is not directly stated in the context. If the context doesn't explicitly state something, do NOT include it in your answer.

3. **EXTRACT WHAT IS AVAILABLE**: If the context contains relevant information that partially answers the question, provide that information. If the context contains related information but not the exact answer, provide what is available and clearly state what information is missing.

4. **EXACT MATCHING**: Match the question to the exact information in the context. If the context contains the answer, provide it directly. If it contains related information, provide that. If it doesn't contain relevant information, explicitly state that the context does not contain the specific information requested.

5. **QUOTE DIRECTLY**: When possible, use the exact wording from the context. Paraphrase only when necessary for clarity, but stay true to the original meaning.

6. **IF INFORMATION IS MISSING**: If the context does not contain the specific information needed to fully answer the question, state clearly what information is missing. However, still provide any related information that IS in the context. Do NOT suggest contacting support, consulting documentation, or any other actions not mentioned in the context.

7. **STRUCTURE**: Provide a clear, direct answer based on what is available in the context. If the context doesn't fully answer the question, state what information is available and what is missing. Do NOT add suggestions, recommendations, or next steps that are not in the context.

Context from video transcripts and insights:
{context}"""),
            ("human", "{question}")
        ])
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
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
        print(f"✓ Ready to answer questions!\n")
    
    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sources based on video_id and video_name.
        Keeps the first occurrence of each unique video.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Deduplicated list of sources
        """
        seen = set()
        deduplicated = []
        
        for source in sources:
            # Create a unique key from video_id and video_name
            video_key = (source.get("video_id", "Unknown"), source.get("video_name", "Unknown"))
            
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
    
    def query_stream(self, question: str):
        """
        Stream query response from the RAG system.
        
        Args:
            question: User's question
            
        Yields:
            Dictionary chunks with answer text and optional sources
        """
        print(f"🔍 Searching for relevant content...")
        
        # Get source documents first to check retrieval
        source_docs = self._retriever.invoke(question)
        
        if not source_docs:
            yield {
                "type": "error",
                "content": "No relevant content found in the video database for this question."
            }
            return
        
        # Debug: Show what was retrieved
        print(f"   Found {len(source_docs)} relevant chunks")
        
        # Store ALL retrieved documents before re-ranking (to find video sources)
        all_retrieved = source_docs.copy()
        
        # Don't filter before re-ranking - use all sources for context
        # We'll filter sources for display later, but use all available info for answering
        # Re-rank documents if re-ranker is available
        if self.reranker and len(source_docs) > FINAL_CHUNKS:
            print(f"   Re-ranking to get top {FINAL_CHUNKS} most relevant chunks...")
            pairs = [[question, doc.page_content] for doc in source_docs]
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
        
        # Prioritize video sources BEFORE generating answer
        # IMPORTANT: Search through ALL retrieved documents, not just re-ranked top 5
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
        
        # If we have video sources, prioritize them for context
        if video_docs:
            # Use ALL video sources first (up to FINAL_CHUNKS), then add text sources if needed
            context_docs = video_docs[:FINAL_CHUNKS]
            if len(context_docs) < FINAL_CHUNKS and text_docs:
                # Add text sources to fill up to FINAL_CHUNKS
                remaining = FINAL_CHUNKS - len(context_docs)
                context_docs.extend(text_docs[:remaining])
            print(f"   Using {len([d for d in context_docs if not self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} video and {len([d for d in context_docs if self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} text chunks for context")
        else:
            # No video sources, use re-ranked or original top documents
            context_docs = source_docs[:FINAL_CHUNKS]
            print(f"   No video sources found, using top {len(context_docs)} documents")
        
        # Format context - prioritize video sources
        context_text = self.format_docs(context_docs)
        
        # Prepare sources from context_docs (which already prioritizes video sources)
        sources = []
        for doc in context_docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            mapped_filename = self._get_original_video_filename(source_file)
            is_text_file = mapped_filename.endswith('.txt')
            
            sources.append({
                "video_name": doc.metadata.get("video_name", "Unknown"),
                "video_id": doc.metadata.get("video_id", "Unknown"),
                "content_type": doc.metadata.get("content_type", "Unknown"),
                "source_file": source_file,
                "is_text_file": is_text_file,
            })
        
        # Prioritize video sources for display: if we have video sources, filter out text sources
        video_sources = [s for s in sources if not s.get("is_text_file", False)]
        text_sources = [s for s in sources if s.get("is_text_file", False)]
        
        # If we have video sources, use only those for display. Otherwise, use all sources.
        if video_sources:
            sources = video_sources
            print(f"   Displaying {len(video_sources)} video sources (filtered out {len(text_sources)} text sources)")
        
        # Deduplicate sources
        sources = self._deduplicate_sources(sources)
        
        # Stream answer using the selected context
        messages = self.prompt_template.format_messages(
            context=context_text,
            question=question
        )
        
        # Stream the response
        full_answer = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_answer += chunk.content
                yield {
                    "type": "content",
                    "content": chunk.content
                }
        
        # Only send sources if they're relevant
        if self._should_show_sources(question, full_answer, sources):
            yield {
                "type": "sources",
                "sources": sources
            }
        
        yield {
            "type": "done"
        }
    
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        print(f"🔍 Searching for relevant content...")
        
        # Get source documents first to check retrieval
        source_docs = self._retriever.invoke(question)
        
        if not source_docs:
            return {
                "answer": "No relevant content found in the video database for this question.",
                "question": question,
                "sources": []
            }
        
        # Debug: Show what was retrieved
        print(f"   Found {len(source_docs)} relevant chunks")
        
        # Store ALL retrieved documents before re-ranking (to find video sources)
        all_retrieved = source_docs.copy()
        
        # Don't filter before re-ranking - use all sources for context
        # We'll filter sources for display later, but use all available info for answering
        # Re-rank documents if re-ranker is available
        if self.reranker and len(source_docs) > FINAL_CHUNKS:
            print(f"   Re-ranking to get top {FINAL_CHUNKS} most relevant chunks...")
            pairs = [[question, doc.page_content] for doc in source_docs]
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
        
        # Prioritize video sources BEFORE generating answer
        # IMPORTANT: Search through ALL retrieved documents, not just re-ranked top 5
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
        
        # If we have video sources, prioritize them for context
        if video_docs:
            # Use ALL video sources first (up to FINAL_CHUNKS), then add text sources if needed
            context_docs = video_docs[:FINAL_CHUNKS]
            if len(context_docs) < FINAL_CHUNKS and text_docs:
                # Add text sources to fill up to FINAL_CHUNKS
                remaining = FINAL_CHUNKS - len(context_docs)
                context_docs.extend(text_docs[:remaining])
            print(f"   Using {len([d for d in context_docs if not self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} video and {len([d for d in context_docs if self._get_original_video_filename(d.metadata.get('source_file', '')).endswith('.txt')])} text chunks for context")
        else:
            # No video sources, use re-ranked or original top documents
            context_docs = source_docs[:FINAL_CHUNKS]
            print(f"   No video sources found, using top {len(context_docs)} documents")
        
        # Format context - prioritize video sources
        context_text = self.format_docs(context_docs)
        
        # Get answer using the selected context
        answer = self.llm.invoke(
            self.prompt_template.format_messages(
                context=context_text,
                question=question
            )
        ).content
        
        response = {
            "answer": answer,
            "question": question
        }
        
        if return_sources:
            # Prepare sources from context_docs (which already prioritizes video sources)
            sources = []
            for doc in context_docs:
                source_file = doc.metadata.get("source_file", "Unknown")
                mapped_filename = self._get_original_video_filename(source_file)
                is_text_file = mapped_filename.endswith('.txt')
                
                sources.append({
                    "video_name": doc.metadata.get("video_name", "Unknown"),
                    "video_id": doc.metadata.get("video_id", "Unknown"),
                    "content_type": doc.metadata.get("content_type", "Unknown"),
                    "source_file": source_file,
                    "is_text_file": is_text_file,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            # Prioritize video sources for display: if we have video sources, filter out text sources
            video_sources = [s for s in sources if not s.get("is_text_file", False)]
            text_sources = [s for s in sources if s.get("is_text_file", False)]
            
            # If we have video sources, use only those for display. Otherwise, use all sources.
            if video_sources:
                sources = video_sources
                print(f"   Displaying {len(video_sources)} video sources (filtered out {len(text_sources)} text sources)")
            
            # Deduplicate sources
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
            
            # Group sources by video to avoid duplicates
            video_sources = {}
            for source in sources:
                video_name = source['video_name']
                if video_name not in video_sources:
                    video_sources[video_name] = {
                        'video_name': video_name,
                        'content_types': set(),
                        'source_file': source['source_file']
                    }
                video_sources[video_name]['content_types'].add(source['content_type'])
            
            # Display grouped sources
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

