# VideoGPT - Video Insights RAG System

AI-powered video analysis platform with Google Drive integration, Qdrant Cloud vector storage, and a modern dark futuristic UI.

## Features

- **Google Drive Integration** - Process videos directly from your Google Drive
- **AI Transcription** - Automatic transcription using Azure Video Indexer
- **Semantic Search** - Qdrant Cloud vector database for fast similarity search
- **RAG Pipeline** - GPT-4o powered answers with context from your videos
- **Re-ranking** - BAAI/bge-reranker-large for improved relevance
- **Streaming Responses** - Real-time streaming answers via SSE
- **Conversation Memory** - Multi-turn conversations with context
- **Multi-tenant** - User-specific video filtering via email
- **Modern UI** - Dark futuristic interface with glassmorphism effects

## Project Structure

```
video_gpt/
├── src/
│   ├── core/
│   │   ├── RAG.py                         # Main RAG query system (Qdrant)
│   │   ├── rag_query.py                   # Legacy CLI query tool
│   │   └── azure_video_indexer.py         # Azure Video Indexer client
│   ├── processing/
│   │   ├── batch_process_videos_gdrive.py # Google Drive batch processor
│   │   └── embedding_qdrant.py            # Qdrant Cloud embedding pipeline
│   └── api/
│       └── router.py                      # FastAPI server with OAuth
├── frontend/
│   ├── src/
│   │   ├── App.jsx                        # Main app component
│   │   ├── components/                    # React components
│   │   └── *.css                          # Dark futuristic styles
│   ├── package.json
│   └── vite.config.js
├── tests/
│   └── test_rag.py                        # RAG system tests
├── data/
│   └── results/                           # Video insights JSON files
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API key
- Qdrant Cloud account
- Google Cloud project with OAuth credentials
- Azure Video Indexer account (for video processing)

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Configure Environment Variables

Create `.env.local` in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant Cloud
QDRANT_ENDPOINT=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Google OAuth (download credentials JSON from Google Cloud Console)
GOOGLE_CREDENTIALS_PATH=credentials/google_oauth_credentials.json

# Azure Video Indexer
AZURE_VIDEO_INDEXER_ACCOUNT_ID=your_account_id
AZURE_VIDEO_INDEXER_API_KEY=your_api_key
AZURE_VIDEO_INDEXER_LOCATION=trial

# Frontend URL (for OAuth redirect)
FRONTEND_URL=http://localhost:3000
```

### 4. Set Up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Web application)
5. Add authorized redirect URI: `http://localhost:8000/auth/callback`
6. Download credentials JSON and save to path specified in `GOOGLE_CREDENTIALS_PATH`

### 5. Set Up Qdrant Cloud

1. Create account at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster
3. Copy the endpoint URL and API key to `.env.local`

## Usage

### Start the Backend API

```bash
python src/api/router.py
```

Or with uvicorn:

```bash
uvicorn src.api.router:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Start the Frontend

```bash
cd frontend
npm run dev
```

Frontend available at: http://localhost:3000

### Embed Video Insights

If you have JSON insight files in `data/results/`, embed them to Qdrant:

```bash
python src/processing/embedding_qdrant.py
```

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/google/login` | GET | Start Google OAuth flow |
| `/auth/callback` | GET | OAuth callback handler |

### Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query videos (non-streaming) |
| `/query/stream` | POST | Query videos (SSE streaming) |
| `/conversation/clear` | POST | Clear conversation history |

### Google Drive

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/drive/folders` | GET | List user's Drive folders |
| `/drive/folders/{id}/files` | GET | List files in folder |
| `/drive/files/{id}/process` | POST | Process a video file |
| `/drive/folders/{id}/process` | POST | Process all videos in folder |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + document count |
| `/stats` | GET | Vector database statistics |

### Query Request Example

```json
{
  "question": "What is the onboarding process?",
  "return_sources": true,
  "session_id": "optional-session-id",
  "user_email": "user@example.com"
}
```

### Query Response Example

```json
{
  "answer": "The onboarding process involves...",
  "question": "What is the onboarding process?",
  "sources": [
    {
      "video_name": "Onboarding Tutorial",
      "video_id": "abc123",
      "content_type": "transcript",
      "source_file": "onboarding.mp4"
    }
  ]
}
```

## Frontend

The VideoGPT frontend features a modern dark futuristic design:

### Design Elements

- **Color Scheme**: Cyan (#00d4ff) and purple (#a855f7) on deep dark backgrounds
- **Effects**: Glassmorphism, neon glows, animated gradients
- **Typography**: Inter font with gradient text effects
- **Animations**: Smooth transitions, hover effects, pulsing elements

### Features

- Google OAuth login
- Real-time streaming responses
- Conversation memory (session-based)
- Source citations with video references
- Google Drive file browser
- Video processing status tracking
- Responsive design

## Testing

Run the test suite:

```bash
# Check setup and run basic query test
python tests/test_rag.py

# Run specific tests
python tests/test_rag.py --setup      # Check environment
python tests/test_rag.py --connection # Test Qdrant connection
python tests/test_rag.py --query      # Test RAG query
python tests/test_rag.py --stream     # Test streaming
python tests/test_rag.py --history    # Test conversation history

# Run all tests
python tests/test_rag.py --all
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Frontend  │────▶│  FastAPI    │────▶│  Qdrant Cloud   │
│   (React)   │◀────│  Backend    │◀────│  Vector Store   │
└─────────────┘     └──────┬──────┘     └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │  OpenAI   │ │  Google   │
              │  GPT-4o   │ │  Drive    │
              └───────────┘ └───────────┘
```

### RAG Pipeline

1. **Query Enhancement** - GPT-4o-mini expands acronyms and adds synonyms
2. **Vector Search** - Retrieve top 50 candidates from Qdrant
3. **Re-ranking** - BAAI/bge-reranker-large scores relevance
4. **Windowing** - Expand context around top 12 chunks (±3500 chars)
5. **Answer Generation** - GPT-4o generates answer from context
6. **Streaming** - Response streamed via Server-Sent Events

## Tech Stack

**Backend:**
- FastAPI
- LangChain
- OpenAI GPT-4o / GPT-4o-mini
- Qdrant Cloud
- BAAI/bge-reranker-large
- Google OAuth 2.0

**Frontend:**
- React 19
- Vite 7
- react-markdown
- CSS (custom dark futuristic theme)

## Security Notes

- Never commit `.env.local` to version control
- Store Google OAuth credentials securely
- Use environment variables for all secrets
- In production, restrict CORS origins
- Use HTTPS in production

## License

MIT
