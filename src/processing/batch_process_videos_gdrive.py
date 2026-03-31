import os
import io
import json
import tempfile
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from faster_whisper import WhisperModel
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys

# Load environment variables
PROJECT_ROOT = Path(find_dotenv('.env.local')).parent
load_dotenv(PROJECT_ROOT / '.env.local')

# Initialize OpenAI client
openai_client = OpenAI()

# Qdrant configuration
QDRANT_ENDPOINT = os.getenv('QDRANT_ENDPOINT')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = "video_insights"

# Results directory for saving insights
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'

# Downloads directory for video files
DOWNLOADS_DIR = PROJECT_ROOT / 'data' / 'downloads'


class VideoBatchProcessor:
    """Process videos from Google Drive using Whisper + GPT-4o-mini with auto-embedding."""

    def __init__(self, creds: any, user_email: str = None, whisper_model: str = "base"):
        """
        Initialize the processor with Google Drive credentials and Whisper model.

        Args:
            creds: Google OAuth credentials
            user_email: User's email for multi-tenant embedding
            whisper_model: Whisper model size (tiny, base, small, medium, large-v2)
        """
        self.creds = creds
        self.user_email = user_email
        self.whisper = WhisperModel(whisper_model, compute_type="int8")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            length_function=len,
            add_start_index=True  # Required for windowed chunk retrieval
        )

    def _list_folders(self):
        service = build('drive', 'v3', credentials=self.creds)

        results = service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name)"
        ).execute()

        return results

    def _list_files_in_folder(self, folder_id: str):
        service = build('drive', 'v3', credentials=self.creds)

        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)"
        ).execute()

        return results

    def download_file(self, file_id: str, file_name: str, download_dir: str = None, use_temp: bool = False) -> str:
        """
        Download a file from Google Drive.

        Args:
            file_id: The Google Drive file ID
            file_name: The name of the file (for saving locally)
            download_dir: Directory to save the file (defaults to data/downloads/)
            use_temp: If True, use temp directory (for processing pipelines)

        Returns:
            str: Path to the downloaded file
        """
        service = build('drive', 'v3', credentials=self.creds)

        # Determine download directory
        if use_temp:
            download_dir = tempfile.mkdtemp()
        elif download_dir is None:
            download_dir = str(DOWNLOADS_DIR)

        os.makedirs(download_dir, exist_ok=True)

        file_path = os.path.join(download_dir, file_name)

        # Request file content
        request = service.files().get_media(fileId=file_id)

        # Download the file
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download progress: {int(status.progress() * 100)}%")

        print(f"Downloaded: {file_name} -> {file_path}")
        return file_path

    def transcribe_video(self, video_path: str) -> str:
        """
        Transcribe video using faster-whisper (local, free).

        Args:
            video_path: Path to the video file

        Returns:
            str: Transcribed text
        """
        print(f"Transcribing: {video_path}")
        segments, info = self.whisper.transcribe(video_path)

        transcript = " ".join([segment.text for segment in segments])
        print(f"Transcription complete. Language: {info.language}, Duration: {info.duration:.1f}s")
        return transcript

    def extract_insights(self, transcript: str, video_name: str) -> dict:
        """
        Extract keywords and summary using GPT-4o-mini.

        Args:
            transcript: The video transcript
            video_name: Name of the video

        Returns:
            dict: Extracted insights (keywords, summary)
        """
        print(f"Extracting insights for: {video_name}")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Analyze this video transcript and extract:
1. keywords: List of 10-15 important keywords/phrases
2. summary: A comprehensive summary (3-5 sentences)
3. topics: List of main topics covered

Transcript:
{transcript}

Return as JSON only, no other text:
{{"keywords": [...], "summary": "...", "topics": [...]}}"""
            }],
            response_format={"type": "json_object"}
        )

        insights = json.loads(response.choices[0].message.content)
        return insights

    def process_video(self, file_id: str, file_name: str) -> dict:
        """
        Full pipeline: download -> transcribe -> extract insights -> save -> embed.

        Args:
            file_id: Google Drive file ID
            file_name: Name of the video file

        Returns:
            dict: Complete video insights
        """
        # Download video to temp (will be cleaned up after processing)
        video_path = self.download_file(file_id, file_name, use_temp=True)

        try:
            # Transcribe
            transcript = self.transcribe_video(video_path)

            # Extract insights
            insights = self.extract_insights(transcript, file_name)

            # Build result structure
            result = {
                "id": file_id,
                "name": file_name,
                "state": "Processed",
                "transcript": transcript,
                "insights": insights,
                "source": "google_drive",
                "user_email": self.user_email
            }

            # Save to results directory
            json_path = self.save_insights(result, file_name)

            # Auto-embed to Qdrant
            self.embed_document(result, json_path)

            return result

        finally:
            # Cleanup temp file
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Cleaned up temp file: {video_path}")

    def embed_document(self, result: dict, json_path: str) -> bool:
        """
        Embed a single document to Qdrant with user-specific metadata.

        Args:
            result: The insights dictionary
            json_path: Path to the saved JSON file

        Returns:
            bool: Success status
        """
        try:
            print(f"📤 Embedding to Qdrant Cloud (user: {self.user_email})...")

            transcript = result.get('transcript', '')
            if not transcript:
                print("⚠ No transcript to embed")
                return False

            # Create document
            doc = Document(
                page_content=transcript,
                metadata={
                    "file": json_path,
                    "video_name": result.get('name', 'Unknown'),
                    "video_id": result.get('id', 'unknown'),
                    "user_email": self.user_email or 'anonymous',
                    "source": "google_drive",
                    "platform": "OnPrintShop",
                    "type": "video_tutorial"
                }
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            # Enrich chunks with metadata
            for chunk in chunks:
                meta = chunk.metadata
                enriched_text = (
                    f"[VIDEO: {meta.get('video_name', 'Unknown')}] "
                    f"[USER: {meta.get('user_email', 'anonymous')}] "
                    f"{chunk.page_content}"
                )
                chunk.page_content = enriched_text

            # Get Qdrant client
            client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

            # Add to Qdrant (append to existing collection)
            Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                url=QDRANT_ENDPOINT,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
                force_recreate=False  # Don't recreate, append to existing
            )

            # Ensure payload indexes exist for efficient filtering
            self._ensure_payload_indexes(client)

            print(f"✅ Embedded {len(chunks)} chunks for user: {self.user_email}")
            return True

        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return False

    def _ensure_payload_indexes(self, client: QdrantClient):
        """Create necessary payload indexes if they don't exist."""
        indexes = ['metadata.user_email', 'metadata.file']
        for field_name in indexes:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass  # Index already exists

    def save_insights(self, result: dict, file_name: str) -> str:
        """
        Save insights to JSON file in data/results/.

        Args:
            result: The insights dictionary
            file_name: Original video filename

        Returns:
            str: Path to saved JSON file
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Create filename without extension + _insights.json
        base_name = Path(file_name).stem
        json_path = RESULTS_DIR / f"{base_name}_insights.json"

        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Saved insights: {json_path}")
        return str(json_path)

    def process_folder(self, folder_id: str) -> list:
        """
        Process all video files in a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID

        Returns:
            list: Results for all processed videos
        """
        # Get files in folder
        files_result = self._list_files_in_folder(folder_id)
        files = files_result.get('files', [])

        # Filter for video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
        video_files = [f for f in files if f['name'].lower().endswith(video_extensions)]

        print(f"Found {len(video_files)} video files in folder")

        results = []
        for i, file in enumerate(video_files):
            print(f"\n[{i+1}/{len(video_files)}] Processing: {file['name']}")
            try:
                result = self.process_video(file['id'], file['name'])
                results.append({"status": "success", "file": file['name'], "result": result})
            except Exception as e:
                print(f"Error processing {file['name']}: {e}")
                results.append({"status": "error", "file": file['name'], "error": str(e)})

        return results



