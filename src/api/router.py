"""
FastAPI router for video insights RAG query system.
"""
import os
# Allow OAuth scope changes (Google may return different scope order or add openid)
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.RAG import VideoRAGQuery
from processing.batch_process_videos_gdrive import VideoBatchProcessor

load_dotenv('.env.local')

# Google OAuth configuration
GOOGLE_CREDENTIALS_PATH = project_root / os.getenv('GOOGLE_CREDENTIALS_PATH')
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]
GOOGLE_REDIRECT_URI = "http://localhost:8000/auth/callback"

# Store OAuth state temporarily (use Redis/DB in production)
oauth_states: Dict[str, Flow] = {}

# Store user credentials (use database in production)
user_credentials: Dict[str, any] = {}

# Initialize FastAPI app
app = FastAPI(
    title="Video Insights RAG API",
    description="API for querying video insights using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (initialized on startup)
rag_system: Optional[VideoRAGQuery] = None

# Conversation memory: store conversation history per session
# Format: {session_id: [{"role": "user"/"assistant", "content": "..."}, ...]}
conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query."""
    question: str = Field(..., description="The question to ask about the video content")
    return_sources: bool = Field(True, description="Whether to return source references")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory. If not provided, a new session will be created.")
    user_email: Optional[str] = Field(None, description="User email to filter results (for multi-tenant queries)")


class SourceInfo(BaseModel):
    """Source information model."""
    video_name: str
    video_id: str
    content_type: str
    source_file: str


class QueryResponse(BaseModel):
    """Response model for query."""
    answer: str
    question: str
    sources: Optional[List[SourceInfo]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        print("Initializing RAG system...")
        rag_system = VideoRAGQuery()
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Video Insights RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query the video insights",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
            "/auth/google/login": "GET - Start Google OAuth flow",
            "/auth/callback": "GET - Google OAuth callback",
            "/drive/folders": "GET - List user's Google Drive folders",
            "/drive/folders/{folder_id}/files": "GET - List files in a folder",
            "/drive/files/{file_id}/download": "POST - Download a file from Google Drive",
            "/drive/files/{file_id}/process": "POST - Process a video (transcribe + insights)",
            "/drive/folders/{folder_id}/process": "POST - Process all videos in a folder"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Check if vector database is accessible (Qdrant)
        client = rag_system.vectorstore.client
        collection_info = client.get_collection("video_insights")
        count = collection_info.points_count
        return {
            "status": "healthy",
            "vector_db": "qdrant_cloud",
            "vector_db_documents": count,
            "rag_system_initialized": True
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG system error: {str(e)}")


@app.get("/auth/google/login")
async def google_login():
    """Start Google OAuth flow - redirects user to Google consent screen."""
    flow = Flow.from_client_secrets_file(
        GOOGLE_CREDENTIALS_PATH,
        scopes=GOOGLE_SCOPES,
        redirect_uri=GOOGLE_REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(access_type='offline')

    # Store flow for callback (keyed by state for CSRF protection)
    oauth_states[state] = flow

    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def google_callback(state: str, code: str):
    """Handle Google OAuth callback - exchanges code for tokens."""
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    flow = oauth_states.pop(state)
    flow.fetch_token(code=code)
    creds = flow.credentials

    # Get user info (email) from Google
    from googleapiclient.discovery import build as build_service
    oauth2_service = build_service('oauth2', 'v2', credentials=creds)
    user_info = oauth2_service.userinfo().get().execute()
    user_email = user_info.get('email', '')

    # Use email as user_id for consistency
    user_id = user_email if user_email else str(uuid.uuid4())

    # Store credentials with user info
    user_credentials[user_id] = {
        "creds": creds,
        "email": user_email,
        "name": user_info.get('name', '')
    }

    # Redirect to frontend with user info
    from urllib.parse import urlencode
    params = urlencode({
        "user_id": user_id,
        "email": user_email,
        "name": user_info.get('name', '')
    })
    # Redirect to frontend (port 3000 in dev, or same origin in production)
    frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    return RedirectResponse(f"{frontend_url}/?{params}")


def get_user_creds(user_id: str):
    """Helper to get user credentials and email."""
    if user_id not in user_credentials:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login via /auth/google/login")
    user_data = user_credentials[user_id]
    return user_data["creds"], user_data.get("email", user_id)


@app.get("/drive/folders")
async def list_drive_files(user_id: str):
    """List folders from user's Google Drive."""
    creds, _ = get_user_creds(user_id)
    results = VideoBatchProcessor(creds=creds)._list_folders()

    return {
        "folders": results.get('files', [])
    }


@app.get("/drive/folders/{folder_id}/files")
async def list_folder_files(folder_id: str, user_id: str):
    """List all files in a specific Google Drive folder."""
    creds, _ = get_user_creds(user_id)
    results = VideoBatchProcessor(creds=creds)._list_files_in_folder(folder_id)

    return {
        "folder_id": folder_id,
        "files": results.get('files', [])
    }


@app.post("/drive/files/{file_id}/download")
async def download_drive_file(file_id: str, user_id: str, file_name: str):
    """Download a file from Google Drive."""
    creds, _ = get_user_creds(user_id)
    processor = VideoBatchProcessor(creds=creds)

    try:
        local_path = processor.download_file(file_id=file_id, file_name=file_name)
        return {
            "status": "downloaded",
            "file_id": file_id,
            "file_name": file_name,
            "local_path": local_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.post("/drive/files/{file_id}/process")
async def process_drive_file(file_id: str, user_id: str, file_name: str):
    """Process a video file: download, transcribe, extract insights, save, and embed."""
    creds, user_email = get_user_creds(user_id)
    processor = VideoBatchProcessor(creds=creds, user_email=user_email)

    try:
        result = processor.process_video(file_id=file_id, file_name=file_name)
        return {
            "status": "processed_and_embedded",
            "user_email": user_email,
            "file_id": file_id,
            "file_name": file_name,
            "insights": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/drive/folders/{folder_id}/process")
async def process_drive_folder(folder_id: str, user_id: str):
    """Process all video files in a Google Drive folder and embed them."""
    creds, user_email = get_user_creds(user_id)
    processor = VideoBatchProcessor(creds=creds, user_email=user_email)

    try:
        results = processor.process_folder(folder_id=folder_id)
        return {
            "status": "completed",
            "user_email": user_email,
            "folder_id": folder_id,
            "processed": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder processing failed: {str(e)}")


@app.post("/query/stream")
async def query_videos_stream(request: QueryRequest):
    """
    Stream query response from the video insights RAG system.
    
    Args:
        request: Query request with question and optional session_id
    
    Returns:
        StreamingResponse with Server-Sent Events (SSE)
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Get or create session ID
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"📝 [STREAM] Session ID: {session_id}")
    
    # Get conversation history for this session
    history = conversation_history[session_id]
    logger.info(f"📚 [STREAM] Conversation history retrieved - Messages: {len(history)}")
    if history:
        logger.debug(f"📚 [STREAM] History content: {json.dumps(history, indent=2)}")
    else:
        logger.info("📚 [STREAM] No previous conversation history (new session)")
    
    async def generate():
        try:
            logger.info(f"🔍 [STREAM] Query: {request.question[:100]}...")
            logger.info(f"📤 [STREAM] Passing {len(history)} history messages to RAG system")
            if request.user_email:
                logger.info(f"🔐 [STREAM] Filtering by user: {request.user_email}")
            full_answer = ""
            for chunk in rag_system.query_stream(request.question, conversation_history=history, user_email=request.user_email):
                try:
                    if chunk["type"] == "content":
                        content = chunk.get("content", "")
                        if content:  # Only add non-empty content
                            full_answer += content
                            yield f"data: {json.dumps(chunk)}\n\n"
                    elif chunk["type"] == "sources" and request.return_sources:
                        # Map JSON filenames to original video filenames
                        sources = []
                        for src in chunk["sources"]:
                            json_filename = src.get("source_file", "Unknown")
                            original_video_file = rag_system._get_original_video_filename(json_filename)
                            
                            # Use filename without extension as video_name if it's "Unknown"
                            video_name = src.get("video_name", "Unknown")
                            if video_name == "Unknown":
                                # Extract filename without extension
                                video_name = Path(original_video_file).stem
                            
                            sources.append({
                                "video_name": video_name,
                                "video_id": src.get("video_id", "Unknown"),
                                "content_type": src.get("content_type", "Unknown"),
                                "source_file": original_video_file
                            })
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                    elif chunk["type"] == "done":
                        # Store conversation history after response is complete
                        conversation_history[session_id].append({
                            "role": "user",
                            "content": request.question
                        })
                        conversation_history[session_id].append({
                            "role": "assistant",
                            "content": full_answer
                        })
                        logger.info(f"💾 [STREAM] Stored conversation - Total messages in session: {len(conversation_history[session_id])}")
                        logger.debug(f"💾 [STREAM] Updated history: {json.dumps(conversation_history[session_id], indent=2)}")
                        # Send session_id with done message
                        chunk["session_id"] = session_id
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as chunk_error:
                    # Log chunk processing error but continue
                    import traceback
                    import sys
                    error_str = str(chunk_error)
                    error_trace = traceback.format_exc()
                    
                    # Print to both stderr and stdout with flush
                    print(f"❌ Error processing chunk: {error_str}", file=sys.stderr, flush=True)
                    print(f"Chunk data: {chunk}", file=sys.stderr, flush=True)
                    print(f"Traceback: {error_trace}", file=sys.stderr, flush=True)
                    print(f"❌ Error processing chunk: {error_str}", flush=True)
                    print(f"Chunk data: {chunk}", flush=True)
                    print(f"Traceback: {error_trace}", flush=True)
                    # Skip this chunk and continue
                    continue
        except Exception as e:
            # Log the full error for debugging - use print with flush to ensure it shows
            import traceback
            import sys
            error_details = traceback.format_exc()
            error_str = str(e)
            error_type = type(e).__name__
            
            # Print to stderr to ensure it's visible
            print(f"❌ CRITICAL ERROR in query_stream: {error_str}", file=sys.stderr, flush=True)
            print(f"Error type: {error_type}", file=sys.stderr, flush=True)
            print(f"Full traceback:\n{error_details}", file=sys.stderr, flush=True)
            
            # Also print to stdout
            print(f"❌ CRITICAL ERROR in query_stream: {error_str}", flush=True)
            print(f"Error type: {error_type}", flush=True)
            print(f"Full traceback:\n{error_details}", flush=True)
            
            # Send error to client - escape the error message to prevent JSON issues
            # Handle specific error cases
            user_friendly_error = None
            
            # Handle ChromaDB/SQLite database errors
            if "Cannot open" in error_str or "data_level0" in error_str or "database" in error_str.lower():
                print(f"   🔍 DETECTED DATABASE ERROR - ChromaDB/SQLite access issue", flush=True)
                user_friendly_error = (
                    "Database access error. This may be due to:\n"
                    "1. Database files are locked by another process\n"
                    "2. Too many concurrent database connections\n"
                    "3. Database corruption\n\n"
                    "Solutions:\n"
                    "- Restart the API server\n"
                    "- Check if embedding process is running\n"
                    "- If persistent, you may need to reset the database"
                )
            # Handle the specific '% ' error case
            elif error_str == "'% '" or "'% '" in error_str:
                print(f"   🔍 DETECTED '% ' ERROR - This is likely a string formatting issue", flush=True)
                print(f"   Error repr: {repr(error_str)}", flush=True)
                user_friendly_error = "Template formatting error. Please try again with a different question."
            
            try:
                if user_friendly_error:
                    safe_error = user_friendly_error.replace('"', '\\"').replace('\n', ' ').replace('\r', '')
                else:
                    safe_error = error_str.replace('"', '\\"').replace('\n', ' ').replace('\r', '')[:200]
                error_json = json.dumps({'type': 'error', 'content': f'Error processing query. {safe_error}'})
                yield f"data: {error_json}\n\n"
            except Exception as json_error:
                # If JSON encoding fails, send a simple error message
                print(f"   ❌ ERROR: Could not encode error to JSON: {json_error}", flush=True)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Error processing query. Please try again.'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse)
async def query_videos(request: QueryRequest):
    """
    Query the video insights using RAG.
    
    Args:
        request: Query request with question and optional parameters
    
    Returns:
        QueryResponse with answer and sources
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"📝 [QUERY] Session ID: {session_id}")
        
        # Get conversation history for this session
        history = conversation_history[session_id]
        logger.info(f"📚 [QUERY] Conversation history retrieved - Messages: {len(history)}")
        if history:
            logger.debug(f"📚 [QUERY] History content: {json.dumps(history, indent=2)}")
        else:
            logger.info("📚 [QUERY] No previous conversation history (new session)")
        
        # Query the RAG system
        try:
            logger.info(f"🔍 [QUERY] Query: {request.question[:100]}...")
            logger.info(f"📤 [QUERY] Passing {len(history)} history messages to RAG system")
            if request.user_email:
                logger.info(f"🔐 [QUERY] Filtering by user: {request.user_email}")
            response = rag_system.query(
                question=request.question,
                return_sources=request.return_sources,
                conversation_history=history,
                user_email=request.user_email
            )
            
            # Validate response structure
            if not response or "answer" not in response:
                raise ValueError("Invalid response structure from RAG system")
            
            # Ensure answer is valid
            if not response.get("answer") or not isinstance(response["answer"], str):
                response["answer"] = "I apologize, but I encountered an error generating the response. Please try asking your question again."
        except Exception as query_error:
            # Log the error but don't expose internal details
            import traceback
            error_details = traceback.format_exc()
            error_str = str(query_error)
            print(f"❌ Error in RAG query: {query_error}")
            print(f"Error details: {error_details}")
            
            # Provide user-friendly error messages for common issues
            if "Cannot open" in error_str or "data_level0" in error_str or "database" in error_str.lower():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Database access error. The vector database may be locked or corrupted. "
                        "Please try restarting the API server. If the issue persists, you may need to reset the database."
                    )
                )
            else:
                raise HTTPException(status_code=500, detail=f"Error processing query: {str(query_error)}")
        
        # Store conversation history
        conversation_history[session_id].append({
            "role": "user",
            "content": request.question
        })
        conversation_history[session_id].append({
            "role": "assistant",
            "content": response["answer"]
        })
        logger.info(f"💾 [QUERY] Stored conversation - Total messages in session: {len(conversation_history[session_id])}")
        logger.debug(f"💾 [QUERY] Updated history: {json.dumps(conversation_history[session_id], indent=2)}")
        
        # Format sources if present
        sources = None
        if request.return_sources and response.get("sources"):
            sources = []
            for src in response["sources"]:
                json_filename = src.get("source_file", "Unknown")
                # Get original video filename with extension
                original_video_file = rag_system._get_original_video_filename(json_filename)
                
                # Use filename without extension as video_name if it's "Unknown"
                video_name = src.get("video_name", "Unknown")
                if video_name == "Unknown":
                    # Extract filename without extension
                    video_name = Path(original_video_file).stem
                
                sources.append(
                    SourceInfo(
                        video_name=video_name,
                        video_id=src.get("video_id", "Unknown"),
                        content_type=src.get("content_type", "Unknown"),
                        source_file=original_video_file
                    )
                )
        
        return QueryResponse(
            answer=response["answer"],
            question=response["question"],
            sources=sources
        )
    
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error processing query: {e}")
        print(f"Error details: {error_details}")
        # Return a user-friendly error message
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


class ClearConversationRequest(BaseModel):
    """Request model for clearing conversation."""
    session_id: str = Field(..., description="Session ID to clear")


@app.post("/conversation/clear")
async def clear_conversation(request: ClearConversationRequest):
    """Clear conversation history for a session."""
    logger.info(f"🗑️  [CLEAR] Clearing conversation for session: {request.session_id}")
    if request.session_id in conversation_history:
        messages_count = len(conversation_history[request.session_id])
        conversation_history[request.session_id] = []
        logger.info(f"🗑️  [CLEAR] Cleared {messages_count} messages from session {request.session_id}")
        return {"status": "cleared", "session_id": request.session_id, "messages_cleared": messages_count}
    logger.warning(f"🗑️  [CLEAR] Session {request.session_id} not found")
    return {"status": "not_found", "session_id": request.session_id}


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector database."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        client = rag_system.vectorstore.client
        collection_info = client.get_collection("video_insights")
        count = collection_info.points_count

        return {
            "total_documents": count,
            "vector_db": "qdrant_cloud",
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-4o",
            "reranking_enabled": rag_system.reranker is not None,
            "active_conversations": len(conversation_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

