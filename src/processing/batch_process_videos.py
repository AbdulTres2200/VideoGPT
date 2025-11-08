"""
Batch process videos from Dropbox using Azure Video Indexer.
Downloads videos, indexes them, and saves insights JSON files.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import dropbox
from dropbox.exceptions import AuthError, ApiError
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.azure_video_indexer import AzureVideoIndexer

# Import embedder for auto-embedding
sys.path.insert(0, str(Path(__file__).parent))
from embed_insights import InsightsEmbedder

load_dotenv('.env.local')

# Configuration
DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')
DROPBOX_FOLDER_PATH = os.getenv('DROPBOX_FOLDER_PATH', '/Onprintshop videos')
RESULTS_DIR = 'data/results'
PROGRESS_FILE = 'data/batch_progress.json'

# Processing parameters - Adjust these values
START_INDEX = 250  # Start from this file index (0-based)
VIDEO_COUNT = None  # Number of files to process (None = process all remaining)

# File extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
TEXT_EXTENSIONS = {'.txt'}


class VideoBatchProcessor:
    """Process videos from Dropbox in batches."""
    
    def __init__(self, auto_embed: bool = True):
        """
        Initialize the processor with Dropbox and Azure Video Indexer clients.
        
        Args:
            auto_embed: If True, automatically embed processed files to vector DB
        """
        self.dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN) if DROPBOX_ACCESS_TOKEN else None
        self.video_indexer = AzureVideoIndexer()
        self.progress = self._load_progress()
        self.auto_embed = auto_embed
        
        # Initialize embedder if auto-embedding is enabled
        self.embedder = None
        if self.auto_embed:
            try:
                self.embedder = InsightsEmbedder()
                print("✓ Auto-embedding enabled - files will be embedded automatically after processing")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize embedder: {e}")
                print("   Files will be processed but not auto-embedded")
                self.auto_embed = False
        
        # Create results directory
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        
        if not self.dbx:
            raise ValueError("DROPBOX_ACCESS_TOKEN not found in .env.local")
        
        # Check token expiration before starting
        self._check_token_expiration()
    
    def _check_token_expiration(self):
        """Check if token is expired or expiring soon and warn user."""
        import base64
        import json
        from datetime import datetime, timedelta
        
        token = os.getenv('AZURE_VIDEO_INDEXER_API_KEY')
        if not token:
            return
        
        try:
            parts = token.split('.')
            if len(parts) >= 2:
                payload = parts[1]
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += '=' * padding
                decoded = base64.urlsafe_b64decode(payload)
                token_data = json.loads(decoded)
                exp = token_data.get('exp', 0)
                
                if exp:
                    exp_time = datetime.fromtimestamp(exp)
                    now = datetime.now()
                    time_left = exp_time - now
                    
                    if time_left.total_seconds() <= 0:
                        print("\n" + "="*60)
                        print("⚠️  WARNING: Your access token has EXPIRED!")
                        print("="*60)
                        print("Get a new token from: https://www.videoindexer.ai/ → Profile → API access")
                        print("Then update AZURE_VIDEO_INDEXER_API_KEY in .env.local")
                        print("="*60 + "\n")
                        raise ValueError("Token expired. Please refresh it.")
                    elif time_left.total_seconds() < 3600:  # Less than 1 hour left
                        hours_left = time_left.total_seconds() / 3600
                        print("\n" + "⚠️  WARNING: Token expires in {:.1f} hours".format(hours_left))
                        print("   If batch processing takes longer, you may need to refresh the token.\n")
        except Exception:
            pass  # If we can't parse, assume token is valid
    
    def _load_progress(self) -> Dict:
        """Load progress from previous run."""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            'processed': [],
            'failed': [],
            'last_file_index': -1,
            'last_video_index': -1  # Keep for backward compatibility
        }
    
    def _save_progress(self):
        """Save current progress."""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def list_files(self, folder_path: str = None) -> List[Dict]:
        """
        List all video and text files in Dropbox folder (recursively including subfolders).
        
        Args:
            folder_path: Dropbox folder path (defaults to DROPBOX_FOLDER_PATH)
        
        Returns:
            List of file info dictionaries with 'type' field ('video' or 'text')
        """
        folder_path = folder_path or DROPBOX_FOLDER_PATH
        files = []
        
        try:
            # Handle root folder
            search_path = folder_path if folder_path != '/' else ''
            
            result = self.dbx.files_list_folder(search_path, recursive=True)
            
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        file_ext = Path(entry.name).suffix.lower()
                        if file_ext in VIDEO_EXTENSIONS:
                            files.append({
                                'name': entry.name,
                                'path': entry.path_display,
                                'size': entry.size,
                                'type': 'video',
                                'modified': entry.server_modified.isoformat() if entry.server_modified else None
                            })
                        elif file_ext in TEXT_EXTENSIONS:
                            files.append({
                                'name': entry.name,
                                'path': entry.path_display,
                                'size': entry.size,
                                'type': 'text',
                                'modified': entry.server_modified.isoformat() if entry.server_modified else None
                            })
                
                if not result.has_more:
                    break
                result = self.dbx.files_list_folder_continue(result.cursor)
        
        except AuthError as e:
            raise ValueError(f"Dropbox authentication error: {e}")
        except ApiError as e:
            raise ValueError(f"Dropbox API error: {e}")
        
        return files
    
    def list_videos(self, folder_path: str = None) -> List[Dict]:
        """
        List all video files in Dropbox folder (for backward compatibility).
        
        Args:
            folder_path: Dropbox folder path (defaults to DROPBOX_FOLDER_PATH)
        
        Returns:
            List of video file info dictionaries
        """
        all_files = self.list_files(folder_path)
        return [f for f in all_files if f.get('type') == 'video']
    
    def download_file(self, dropbox_path: str, local_filename: str) -> str:
        """
        Download a file from Dropbox.
        
        Args:
            dropbox_path: Path in Dropbox
            local_filename: Local filename to save as
        
        Returns:
            Local file path
        """
        local_path = os.path.join(RESULTS_DIR, local_filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            return local_path
        
        try:
            metadata, response = self.dbx.files_download(dropbox_path)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
        except Exception as e:
            raise Exception(f"Download failed: {e}")
    
    def download_video(self, dropbox_path: str, local_filename: str) -> str:
        """
        Download a video file from Dropbox (for backward compatibility).
        
        Args:
            dropbox_path: Path in Dropbox
            local_filename: Local filename to save as
        
        Returns:
            Local file path
        """
        return self.download_file(dropbox_path, local_filename)
    
    def process_video(self, video_info: Dict) -> Optional[Dict]:
        """
        Process a single video: download, upload to Azure, index, and save results.
        
        Args:
            video_info: Video information dictionary
        
        Returns:
            Processing result dictionary or None if failed
        """
        video_name = video_info['name']
        video_path = video_info['path']
        
        # Check if already processed
        video_id_key = video_info['path']
        if any(p.get('path') == video_id_key for p in self.progress['processed']):
            print(f"  ⏭️  Already processed, skipping...")
            return None
        
        try:
            # Step 1: Download video
            print(f"  📥 Downloading...")
            local_path = self.download_video(video_path, video_name)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  ✓ Downloaded ({size_mb:.2f} MB)")
            
            # Step 2: Upload to Azure Video Indexer
            # Sanitize and truncate video name for Azure (max 80 chars)
            sanitized_name = Path(video_name).stem
            # Remove/replace problematic characters
            sanitized_name = sanitized_name.replace('+', ' ').replace('&', 'and').replace('/', '_')
            # Remove/replace other special characters that might cause issues
            sanitized_name = ''.join(c if c.isalnum() or c in (' ', '-', '_', '.') else '_' for c in sanitized_name)
            # Trim to 80 chars max (Azure limit)
            MAX_NAME_LENGTH = 80
            if len(sanitized_name) > MAX_NAME_LENGTH:
                sanitized_name = sanitized_name[:MAX_NAME_LENGTH].strip()
            
            print(f"  📤 Uploading to Azure Video Indexer...")
            try:
                result = self.video_indexer.upload_video_file(
                    local_path,
                    video_name=sanitized_name,
                    language="en-US"
                )
            except Exception as upload_error:
                error_str = str(upload_error)
                # Handle 403 Forbidden - usually happens right after account switch, retry once
                if "403" in error_str or "Forbidden" in error_str or "USER_NOT_SIGNED_IN_AAD" in error_str:
                    print(f"  ⚠️  403 error (might be account switch). Retrying once...")
                    import time
                    time.sleep(1)  # Brief pause
                    try:
                        result = self.video_indexer.upload_video_file(
                            local_path,
                            video_name=sanitized_name,
                            language="en-US"
                        )
                        print(f"  ✓ Upload succeeded on retry")
                    except Exception as retry_error:
                        # If retry also fails, treat as normal error
                        raise upload_error  # Raise original error
                # Handle 409 Conflict - video already exists
                elif "409" in error_str or "Conflict" in error_str:
                    print(f"  ⚠️  Video with this name already exists. Trying with unique name...")
                    # Add timestamp suffix, ensure total length <= 80
                    import time
                    timestamp = str(int(time.time()))
                    # Reserve space for underscore and timestamp (typically 10-11 chars)
                    max_base_length = MAX_NAME_LENGTH - len(timestamp) - 1
                    if max_base_length < 10:
                        max_base_length = 10  # Minimum base name length
                    truncated_base = sanitized_name[:max_base_length].rstrip()
                    unique_name = f"{truncated_base}_{timestamp}"
                    # Final check - should be <= 80 now
                    if len(unique_name) > MAX_NAME_LENGTH:
                        unique_name = unique_name[:MAX_NAME_LENGTH]
                    
                    result = self.video_indexer.upload_video_file(
                        local_path,
                        video_name=unique_name,
                        language="en-US"
                    )
                    print(f"  ✓ Uploaded with unique name: {unique_name}")
                # Handle 400 Bad Request - name too long (in case truncation didn't work)
                elif "400" in error_str and ("VIDEO_NAME_TOO_LONG" in error_str or "too long" in error_str.lower()):
                    print(f"  ⚠️  Name too long. Using shorter name with timestamp...")
                    import time
                    timestamp = str(int(time.time()))
                    # Use first part of filename + timestamp
                    short_name = Path(video_name).stem[:50].strip()
                    short_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in short_name)
                    # Combine with timestamp, ensure <= 80
                    max_base = MAX_NAME_LENGTH - len(timestamp) - 1
                    short_base = short_name[:max_base] if len(short_name) > max_base else short_name
                    unique_name = f"{short_base}_{timestamp}"
                    if len(unique_name) > MAX_NAME_LENGTH:
                        unique_name = unique_name[:MAX_NAME_LENGTH]
                    
                    result = self.video_indexer.upload_video_file(
                        local_path,
                        video_name=unique_name,
                        language="en-US"
                    )
                    print(f"  ✓ Uploaded with shortened name: {unique_name}")
                else:
                    raise  # Re-raise if it's a different error
            
            video_id = result.get('id')
            print(f"  ✓ Uploaded (Video ID: {video_id})")
            
            # Step 3: Wait for indexing
            print(f"  ⏳ Indexing... (this may take a while)")
            index_result = self.video_indexer.wait_for_indexing(video_id, timeout=3600)
            print(f"  ✓ Indexing completed!")
            
            # Step 4: Save insights JSON with proper naming
            # Use video name (sanitized) as filename
            safe_name = "".join(c for c in Path(video_name).stem if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name[:100]  # Limit length
            output_file = os.path.join(RESULTS_DIR, f"{safe_name}_insights.json")
            
            with open(output_file, 'w') as f:
                json.dump(index_result, f, indent=2)
            
            print(f"  💾 Saved insights to: {Path(output_file).name}")
            
            # Step 5: Clean up local video file to save space
            try:
                os.remove(local_path)
                print(f"  🗑️  Cleaned up local video file")
            except:
                pass  # Continue even if cleanup fails
            
            # Extract summary info
            summary = {
                'video_id': video_id,
                'original_name': video_name,
                'dropbox_path': video_path,
                'duration': index_result.get('duration'),
                'state': index_result.get('state'),
                'processed_at': datetime.now().isoformat(),
                'insights_file': Path(output_file).name
            }
            
            # Get transcript info
            videos = index_result.get('videos', [])
            if videos:
                insights = videos[0].get('insights', {})
                transcript = insights.get('transcript', [])
                if transcript:
                    full_text = ' '.join([t.get('text', '') for t in transcript])
                    summary['transcript_length'] = len(full_text)
                    summary['transcript_segments'] = len(transcript)
            
            # Mark as processed
            self.progress['processed'].append({
                'video_id': video_id,
                'name': video_name,
                'path': video_path,
                'type': 'video',
                'summary': summary
            })
            self._save_progress()
            
            # Step 6: Auto-embed if enabled
            if self.auto_embed and self.embedder:
                try:
                    print(f"  🔄 Auto-embedding...")
                    json_path = Path(output_file)
                    documents = self.embedder.process_json_file(json_path, force_reprocess=False)
                    if documents:
                        # Get current count before adding (to verify addition)
                        try:
                            current_count = self.embedder.vectorstore._collection.count()
                        except:
                            current_count = 0
                        
                        # Add documents (this APPENDS to existing embeddings, does not overwrite)
                        self.embedder.vectorstore.add_documents(documents)
                        
                        # Verify addition
                        try:
                            new_count = self.embedder.vectorstore._collection.count()
                            added_count = new_count - current_count
                            if added_count != len(documents):
                                print(f"  ⚠️  Warning: Expected {len(documents)} documents, but {added_count} were added")
                        except:
                            pass
                        
                        # Update embedding progress
                        self.embedder.progress['processed'].append({
                            'filename': json_path.name,
                            'chunks': len(documents),
                            'index': len(self.embedder.progress['processed'])
                        })
                        self.embedder.progress['total_chunks'] += len(documents)
                        self.embedder._save_progress()
                        print(f"  ✓ Auto-embedded {len(documents)} chunks (Total in DB: {new_count if 'new_count' in locals() else 'unknown'})")
                except Exception as e:
                    print(f"  ⚠️  Auto-embedding failed: {e}")
                    # Continue anyway - file is processed, just not embedded yet
            
            return summary
        
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            
            # Check if it's a token expiration error
            if "expired" in error_msg.lower() or "401" in error_msg or "Unauthorized" in error_msg:
                print("\n" + "="*60)
                print("🔄 AUTHENTICATION ERROR DETECTED")
                print("="*60)
                
                # Try to reload token from .env.local
                try:
                    print("  Attempting to reload token from .env.local...")
                    self.video_indexer.reload_token()
                    print("  ✓ Token reloaded! Retrying this file...")
                    print("="*60 + "\n")
                    
                    # Retry processing this video
                    return self.process_video(video_info)
                except Exception as reload_error:
                    print(f"  ✗ Token reload failed: {reload_error}")
                    print("\n  Manual steps:")
                    print("  1. Get new token: https://www.videoindexer.ai/ → Profile → API access")
                    print("  2. Update AZURE_VIDEO_INDEXER_API_KEY in .env.local")
                    print("  3. Resume from this video (progress is saved)")
                    print("="*60 + "\n")
            
            # Check if it's a connection error (499, timeout, etc.) that should be retried
            # The upload_video_file method already has retry logic, so if we get here,
            # it means all retries failed. But we should still mark it as failed.
            is_connection_error = any(keyword in error_msg.lower() for keyword in [
                '499', 'connection', 'timeout', 'closed', 'client connection'
            ])
            
            if is_connection_error:
                print(f"  ⚠️  Connection error detected. The upload method already retried.")
                print(f"  This may be due to network issues or file size. The file will be marked as failed.")
                print(f"  You can resume processing later - progress is saved.")
            
            self.progress['failed'].append({
                'name': video_name,
                'path': video_path,
                'reason': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            self._save_progress()
            return None
    
    def process_text_file(self, text_info: Dict) -> Optional[Dict]:
        """
        Process a single text file: download and convert to insights format.
        
        Args:
            text_info: Text file information dictionary
        
        Returns:
            Processing result dictionary or None if failed
        """
        text_name = text_info['name']
        text_path = text_info['path']
        
        # Check if already processed
        file_key = text_info['path']
        if any(p.get('path') == file_key for p in self.progress['processed']):
            print(f"  ⏭️  Already processed, skipping...")
            return None
        
        try:
            # Step 1: Download text file
            print(f"  📥 Downloading...")
            local_path = self.download_file(text_path, text_name)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  ✓ Downloaded ({size_kb:.2f} KB)")
            
            # Step 2: Read text content
            with open(local_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                print(f"  ⚠️  File is empty, skipping...")
                return None
            
            # Step 3: Create insights JSON in the same format as video insights
            # This ensures compatibility with the embedding pipeline
            safe_name = "".join(c for c in Path(text_name).stem if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name[:100]  # Limit length
            
            # Create a mock insights structure that matches video insights format
            insights_data = {
                'id': f"txt_{hash(text_path) % 1000000}",  # Generate a unique ID
                'name': safe_name,
                'state': 'Processed',
                'videos': [{
                    'insights': {
                        'transcript': [
                            {'text': text_content}
                        ]
                    }
                }],
                'summarizedInsights': {
                    'keywords': [],
                    'labels': [],
                    'summary': {}
                }
            }
            
            # Save as JSON (same format as video insights)
            output_file = os.path.join(RESULTS_DIR, f"{safe_name}_insights.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(insights_data, f, indent=2)
            
            print(f"  💾 Saved insights to: {Path(output_file).name}")
            
            # Step 4: Clean up local text file
            try:
                os.remove(local_path)
                print(f"  🗑️  Cleaned up local text file")
            except:
                pass  # Continue even if cleanup fails
            
            # Extract summary info
            summary = {
                'file_id': insights_data['id'],
                'original_name': text_name,
                'dropbox_path': text_path,
                'processed_at': datetime.now().isoformat(),
                'insights_file': Path(output_file).name,
                'content_length': len(text_content),
                'type': 'text'
            }
            
            # Mark as processed
            self.progress['processed'].append({
                'file_id': insights_data['id'],
                'name': text_name,
                'path': text_path,
                'type': 'text',
                'summary': summary
            })
            self._save_progress()
            
            # Step 5: Auto-embed if enabled
            if self.auto_embed and self.embedder:
                try:
                    print(f"  🔄 Auto-embedding...")
                    json_path = Path(output_file)
                    documents = self.embedder.process_json_file(json_path, force_reprocess=False)
                    if documents:
                        self.embedder.vectorstore.add_documents(documents)
                        # Update embedding progress
                        self.embedder.progress['processed'].append({
                            'filename': json_path.name,
                            'chunks': len(documents),
                            'index': len(self.embedder.progress['processed'])
                        })
                        self.embedder.progress['total_chunks'] += len(documents)
                        self.embedder._save_progress()
                        print(f"  ✓ Auto-embedded {len(documents)} chunks")
                except Exception as e:
                    print(f"  ⚠️  Auto-embedding failed: {e}")
                    # Continue anyway - file is processed, just not embedded yet
            
            return summary
        
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            
            self.progress['failed'].append({
                'name': text_name,
                'path': text_path,
                'type': 'text',
                'reason': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            self._save_progress()
            return None
    
    def process_batch(self, start_index: int = 0, count: int = None):
        """
        Process a batch of files (videos and text files).
        Traverses subfolders recursively.
        
        Args:
            start_index: Index to start from (0-based)
            count: Number of files to process (None = process all remaining)
        """
        print("=" * 70)
        print("FILE BATCH PROCESSOR")
        print("=" * 70)
        print(f"📂 Scanning Dropbox folder: {DROPBOX_FOLDER_PATH} (recursive)")
        print()
        
        # List all files (videos and text files)
        all_files = self.list_files()
        total_files = len(all_files)
        
        videos_count = len([f for f in all_files if f.get('type') == 'video'])
        texts_count = len([f for f in all_files if f.get('type') == 'text'])
        
        print(f"✓ Found {total_files} files:")
        print(f"  - Videos: {videos_count}")
        print(f"  - Text files: {texts_count}")
        print()
        
        # Determine which files to process
        end_index = start_index + count if count else total_files
        files_to_process = all_files[start_index:end_index]
        
        print(f"📋 Processing files {start_index} to {end_index-1} ({len(files_to_process)} files)")
        print("=" * 70)
        print()
        
        for i, file_info in enumerate(files_to_process):
            current_index = start_index + i
            file_type = file_info.get('type', 'unknown')
            file_name = file_info['name']
            
            print(f"\n[{current_index + 1}/{total_files}] {file_name}")
            print(f"  Type: {file_type.upper()}")
            
            if file_type == 'video':
                size_mb = file_info['size'] / 1024 / 1024
                print(f"  Size: {size_mb:.2f} MB")
                result = self.process_video(file_info)
            elif file_type == 'text':
                size_kb = file_info['size'] / 1024
                print(f"  Size: {size_kb:.2f} KB")
                result = self.process_text_file(file_info)
            else:
                print(f"  ⚠️  Unknown file type, skipping...")
                result = None
            
            if result:
                print(f"  ✅ Success!")
            else:
                print(f"  ❌ Failed or skipped")
            
            # Update progress
            self.progress['last_file_index'] = current_index
            self._save_progress()
            
            # Small delay between files
            if i < len(files_to_process) - 1:  # Don't delay after last file
                time.sleep(1)  # Shorter delay for text files
        
        # Final summary
        processed_videos = len([p for p in self.progress['processed'] if p.get('type') == 'video'])
        processed_texts = len([p for p in self.progress['processed'] if p.get('type') == 'text'])
        
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total files processed: {len(self.progress['processed'])}")
        print(f"  - Videos: {processed_videos}")
        print(f"  - Text files: {processed_texts}")
        print(f"Failed: {len(self.progress['failed'])}")
        print(f"Last processed index: {self.progress.get('last_file_index', -1)}")
        print(f"\nResults saved to: {RESULTS_DIR}/")
        if self.auto_embed:
            print(f"✓ Auto-embedding enabled - files embedded to vector database")
        print("=" * 70)


def main():
    """Main function."""
    try:
        processor = VideoBatchProcessor()
        processor.process_batch(start_index=START_INDEX, count=VIDEO_COUNT)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

