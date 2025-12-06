# Video GPT - Video Insights RAG System

Azure Video Indexer integration for video analysis and RAG (Retrieval-Augmented Generation) pipeline.

## Project Structure

```
video_gpt/
├── src/
│   ├── core/
│   │   ├── azure_video_indexer.py    # Azure Video Indexer client
│   │   └── rag_query.py              # RAG query system
│   ├── processing/
│   │   ├── batch_process_videos.py   # Batch video processing
│   │   └── embed_insights.py         # Embedding script
│   └── api/
│       └── router.py                 # FastAPI router
├── tests/
│   ├── test_rag.py                   # RAG system tests
│   └── test_embeddings.py            # Embedding tests
├── data/
│   ├── results/                      # Video insights JSON files
│   ├── vector_db/                   # ChromaDB vector database
│   ├── batch_progress.json           # Batch processing progress
│   └── embedding_progress.json      # Embedding progress
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env.local` file with your credentials:
```env
AZURE_VIDEO_INDEXER_API_KEY=your_key_here
DROPBOX_ACCESS_TOKEN=your_token_here
DROPBOX_FOLDER_PATH=/Onprintshop videos
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### 1. Process Files (Batch)

Process videos and text files from Dropbox (recursively including subfolders) and extract insights:

```bash
python src/processing/batch_process_videos.py
```

**Features:**
- Traverses all subfolders recursively
- Processes both video files (.mp4, .avi, .mov, etc.) and text files (.txt)
- Automatically embeds processed files to vector database
- Videos are indexed with Azure Video Indexer
- Text files are converted to the same format and embedded directly

**Configuration:**
Edit the script to adjust:
- `START_INDEX`: Starting file index (0-based)
- `VIDEO_COUNT`: Number of files to process (None = all remaining)
- `auto_embed`: Set to `False` in `VideoBatchProcessor(auto_embed=False)` to disable auto-embedding

### 2. Embed Insights

Embed the extracted insights for RAG. The repository includes `data/results/` but not the generated `chroma_index/`. You need to regenerate it:

**Option A: Using OpenAI embeddings (for API/RAG.py):**
```bash
python src/processing/embedding_chroma.py
```
This creates `chroma_index/` used by the API server.

**Option B: Using HuggingFace embeddings (for CLI/rag_query.py):**
```bash
python src/processing/embed_insights.py
```
This creates `data/vector_db/` used by the CLI query tool.

### 3. Query RAG System (CLI)

Query the video insights interactively:

```bash
python src/core/rag_query.py
```

Or query from command line:

```bash
python src/core/rag_query.py "What is OnPrintShop?"
```

### 4. API Server

Start the FastAPI server:

```bash
python src/api/router.py
```

Or using uvicorn directly:

```bash
uvicorn src.api.router:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### API Endpoints

#### POST `/query`
Query the video insights.

**Request:**
```json
{
  "question": "What is OnPrintShop?",
  "return_sources": true
}
```

**Response:**
```json
{
  "answer": "...",
  "question": "What is OnPrintShop?",
  "sources": [
    {
      "video_name": "Video Name",
      "video_id": "abc123",
      "content_type": "transcript",
      "source_file": "Video Name.mp4"
    }
  ]
}
```

#### GET `/health`
Check API health and vector database status.

#### GET `/stats`
Get statistics about the vector database.

## Features

- Upload videos from Dropbox
- Extract transcripts, keywords, labels, and summaries
- Embed video insights for semantic search
- RAG pipeline with OpenAI GPT-4o
- Re-ranking for better relevance
- FastAPI REST API
- Resume capability for batch processing

## Frontend

The VideoGPT frontend is a ChatGPT-like interface for querying video insights.

### Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Features

- 🎨 Modern ChatGPT-inspired UI
- ⚡ Real-time streaming responses
- 📱 Responsive design
- 🎯 Source references display
- 🚀 Fast and lightweight

## Testing

Run tests:

```bash
python tests/test_rag.py
python tests/test_embeddings.py
```
