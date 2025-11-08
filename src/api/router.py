"""
FastAPI router for video insights RAG query system.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.rag_query import VideoRAGQuery

load_dotenv('.env.local')

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


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query."""
    question: str = Field(..., description="The question to ask about the video content")
    return_sources: bool = Field(True, description="Whether to return source references")


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
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Check if vector database is accessible
        collection = rag_system.vectorstore._collection
        count = collection.count()
        return {
            "status": "healthy",
            "vector_db_documents": count,
            "rag_system_initialized": True
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG system error: {str(e)}")


@app.post("/query/stream")
async def query_videos_stream(request: QueryRequest):
    """
    Stream query response from the video insights RAG system.
    
    Args:
        request: Query request with question
    
    Returns:
        StreamingResponse with Server-Sent Events (SSE)
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    async def generate():
        try:
            for chunk in rag_system.query_stream(request.question):
                if chunk["type"] == "sources" and request.return_sources:
                    # Map JSON filenames to original video filenames
                    sources = []
                    for src in chunk["sources"]:
                        json_filename = src.get("source_file", "Unknown")
                        original_video_file = rag_system._get_original_video_filename(json_filename)
                        sources.append({
                            "video_name": src.get("video_name", "Unknown"),
                            "video_id": src.get("video_id", "Unknown"),
                            "content_type": src.get("content_type", "Unknown"),
                            "source_file": original_video_file
                        })
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                else:
                    yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
    
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
        # Query the RAG system
        response = rag_system.query(
            question=request.question,
            return_sources=request.return_sources
        )
        
        # Format sources if present
        sources = None
        if request.return_sources and response.get("sources"):
            sources = []
            for src in response["sources"]:
                json_filename = src.get("source_file", "Unknown")
                # Get original video filename with extension
                original_video_file = rag_system._get_original_video_filename(json_filename)
                sources.append(
                    SourceInfo(
                        video_name=src.get("video_name", "Unknown"),
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
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector database."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        collection = rag_system.vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "gpt-4o",
            "reranking_enabled": rag_system.reranker is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

