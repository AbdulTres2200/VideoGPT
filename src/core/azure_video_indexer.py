"""
Azure Video Indexer client for video analysis.
"""
import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

load_dotenv('.env.local')


class AzureVideoIndexer:
    """Client for Azure Video Indexer API."""
    
    BASE_URL = "https://api.videoindexer.ai"
    
    def __init__(self, api_key: Optional[str] = None, subscription_key: Optional[str] = None):
        """
        Initialize the Azure Video Indexer client.
        
        Args:
            api_key: Access token (JWT) from Azure Video Indexer. If not provided, 
                    will load from AZURE_VIDEO_INDEXER_API_KEY environment variable.
            subscription_key: Optional subscription key. If provided, will be used to
                            get access tokens. If not provided, api_key will be used as access token.
        """
        self.access_token = api_key or os.getenv('AZURE_VIDEO_INDEXER_API_KEY')
        self.subscription_key = subscription_key or os.getenv('AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY')
        
        # Simplified: Just use access token directly
        # Token cache for checking expiration
        self._cached_token = None
        self._token_expires_at = None
        
        if not self.access_token:
            raise ValueError(
                "AZURE_VIDEO_INDEXER_API_KEY is required in .env.local\n"
                "Get token from: https://www.videoindexer.ai/ → Profile → API access → Generate token"
            )
        
        # Extract account ID and location from the access token (if it's a JWT)
        try:
            self.account_id = self._extract_account_id()
            self.location = self._extract_location()
        except:
            # If extraction fails, try to get from environment or use defaults
            self.account_id = os.getenv('AZURE_VIDEO_INDEXER_ACCOUNT_ID')
            self.location = os.getenv('AZURE_VIDEO_INDEXER_LOCATION', 'eastus')
            
            if not self.account_id:
                # Try to extract from any available token
                if self.access_token:
                    try:
                        self.account_id = self._extract_account_id()
                        self.location = self._extract_location()
                    except:
                        pass
    
    def _is_token_expired(self, token: str) -> bool:
        """Check if a JWT token is expired."""
        import base64
        import json
        import time
        
        try:
            parts = token.split('.')
            if len(parts) < 2:
                return True  # Invalid token, consider expired
            
            # Decode the payload
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            token_data = json.loads(decoded)
            
            # Check expiration
            exp = token_data.get('exp', 0)
            if exp == 0:
                return False  # No expiration set
            
            # Add 60 second buffer
            return time.time() >= (exp - 60)
        except:
            return False  # If we can't parse it, assume it's valid
    
    def _get_access_token(self) -> str:
        """
        Get an access token for API calls.
        Auto-refreshes using subscription key if token is expired.
        """
        # If no token at all, we need one to start
        if not self.access_token:
            raise ValueError(
                "Access token is required. Set AZURE_VIDEO_INDEXER_API_KEY in .env.local\n"
                "Get token from: https://www.videoindexer.ai/ → Profile → API access\n\n"
                "For automatic refresh, also set AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY"
            )
        
        # Check if token is expired or about to expire (within 5 minutes)
        is_expired = self._is_token_expired(self.access_token)
        
        if is_expired:
            # Try to auto-refresh using subscription key
            if self.subscription_key:
                print("🔄 Token expired. Attempting automatic refresh...")
                try:
                    new_token = self._refresh_token_with_subscription_key()
                    
                    # Extract account info from new token (subscription key generates trial account token)
                    import base64
                    import json
                    try:
                        parts = new_token.strip('"').split('.')
                        if len(parts) >= 2:
                            payload = parts[1]
                            padding = 4 - len(payload) % 4
                            if padding != 4:
                                payload += '=' * padding
                            decoded = base64.urlsafe_b64decode(payload)
                            data = json.loads(decoded)
                            new_account_id = data.get('AccountId')
                            new_location = data.get('IssuerLocation', 'trial')
                            
                            # Update account info to match new token
                            old_account = self.account_id
                            self.account_id = new_account_id
                            self.location = new_location
                            self.access_token = new_token.strip('"')
                            
                            if old_account and old_account != new_account_id:
                                print(f"⚠️  Account switched: {old_account[:8]}... → {new_account_id[:8]}...")
                                print(f"   Using {new_location} account (from subscription key)")
                            else:
                                print("✓ Token refreshed automatically!")
                            
                            return self.access_token
                    except Exception as e:
                        # If we can't extract, just use the token
                        self.access_token = new_token.strip('"')
                        print("✓ Token refreshed automatically!")
                        return self.access_token
                except Exception as e:
                    error_msg = str(e)
                    # If subscription key method fails, provide manual instructions
                    account_id = self.account_id or "your_account_id"
                    location = self.location or "eastus"
                    
                    exp_time = "unknown"
                    try:
                        import base64
                        import json
                        from datetime import datetime
                        parts = self.access_token.split('.')
                        if len(parts) >= 2:
                            payload = parts[1]
                            padding = 4 - len(payload) % 4
                            if padding != 4:
                                payload += '=' * padding
                            decoded = base64.urlsafe_b64decode(payload)
                            token_data = json.loads(decoded)
                            exp = token_data.get('exp', 0)
                            if exp:
                                exp_time = datetime.fromtimestamp(exp).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                    
                    raise ValueError(
                        f"❌ Access token expired (expired: {exp_time})\n"
                        f"❌ Auto-refresh failed: {error_msg}\n\n"
                        f"Manual fix:\n"
                        f"1. Go to: https://www.videoindexer.ai/\n"
                        f"2. Profile → API access → Generate new token\n"
                        f"3. Copy the new token\n"
                        f"4. Update AZURE_VIDEO_INDEXER_API_KEY in .env.local\n\n"
                        f"Or fix subscription key in .env.local for auto-refresh to work.\n"
                        f"Account: {account_id}, Location: {location}"
                    )
            else:
                # No subscription key or missing account info - manual refresh required
                account_id = self.account_id or "your_account_id"
                location = self.location or "eastus"
                
                exp_time = "unknown"
                try:
                    import base64
                    import json
                    from datetime import datetime
                    parts = self.access_token.split('.')
                    if len(parts) >= 2:
                        payload = parts[1]
                        padding = 4 - len(payload) % 4
                        if padding != 4:
                            payload += '=' * padding
                        decoded = base64.urlsafe_b64decode(payload)
                        token_data = json.loads(decoded)
                        exp = token_data.get('exp', 0)
                        if exp:
                            exp_time = datetime.fromtimestamp(exp).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
                
                raise ValueError(
                    f"❌ Access token expired (expired: {exp_time})\n\n"
                    f"Quick fix:\n"
                    f"1. Go to: https://www.videoindexer.ai/\n"
                    f"2. Profile → API access → Generate new token\n"
                    f"3. Copy the new token\n"
                    f"4. Update AZURE_VIDEO_INDEXER_API_KEY in .env.local\n\n"
                    f"💡 TIP: Set AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY in .env.local for automatic refresh!\n"
                    f"Account: {account_id}, Location: {location}"
                )
        
        return self.access_token
    
    def _refresh_token_with_subscription_key(self) -> str:
        """Refresh access token using subscription key (for automatic refresh)."""
        if not self.subscription_key:
            raise ValueError("Subscription key required for auto-refresh")
        
        # For subscription key, use trial account if env vars are set, otherwise use current account
        # The subscription key from test_token.py works with trial account
        subscription_account_id = os.getenv('AZURE_VIDEO_INDEXER_SUBSCRIPTION_ACCOUNT_ID', 'a9696e33-fd7a-4070-83f6-fc8d2e06633a')
        subscription_location = os.getenv('AZURE_VIDEO_INDEXER_SUBSCRIPTION_LOCATION', 'trial')
        
        # Try subscription account first (where key works), then fallback to current account
        account_id = subscription_account_id
        location = subscription_location
        
        if not account_id or not location:
            # Try to extract from current token (even if expired, we can read account info)
            try:
                account_id = self._extract_account_id()
                location = self._extract_location()
            except:
                raise ValueError(
                    "Cannot auto-refresh: account_id and location not available.\n"
                    "Set AZURE_VIDEO_INDEXER_SUBSCRIPTION_ACCOUNT_ID and AZURE_VIDEO_INDEXER_SUBSCRIPTION_LOCATION\n"
                    "Or keep AZURE_VIDEO_INDEXER_API_KEY in .env.local (even if expired) to extract account info."
                )
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key.strip()
        }
        
        # Use the exact method from test_token.py - direct API call
        # Note: Use capitalized "Auth" in path (this is the correct endpoint)
        url = f"{self.BASE_URL}/Auth/{location}/Accounts/{account_id}/AccessToken?allowEdit=true"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Token comes as string with quotes, strip them
            token = response.text.strip('"')
            return token
        
        # If failed, provide helpful error
        error_detail = ""
        try:
            error_json = response.json()
            error_detail = error_json.get('Message', error_json.get('ErrorType', str(error_json)))
        except:
            error_detail = response.text[:200]
        
        if "USER_NOT_REGISTERED" in error_detail or "not registered to APIM" in error_detail:
            raise ValueError(
                f"Subscription key authentication failed.\n"
                f"Error: {error_detail}\n\n"
                f"Check:\n"
                f"1. Subscription key is correct: {self.subscription_key[:10]}...\n"
                f"2. Account ID matches: {account_id}\n"
                f"3. Location is correct: {location}\n"
                f"4. Subscription key is active in API Portal"
            )
        else:
            raise ValueError(
                f"Subscription key authentication failed ({response.status_code}).\n"
                f"Error: {error_detail}\n\n"
                f"Verify subscription key, account ID, and location are correct."
            )
    
    def upload_video(
        self,
        video_url: str,
        video_name: Optional[str] = None,
        language: str = "en-US",
        privacy: str = "Private",
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a video for indexing from a URL.
        
        Args:
            video_url: URL of the video to index
            video_name: Name for the video (optional)
            language: Language of the video (default: "en-US")
            privacy: Privacy setting - "Private" or "Public" (default: "Private")
            callback_url: Optional callback URL for notifications
        
        Returns:
            Dictionary containing video ID and upload status
        """
        # Get account ID and location
        account_id = self.account_id or self._extract_account_id()
        location = self.location or self._extract_location()
        
        # Get access token
        access_token = self._get_access_token()
        
        # Upload video
        url = f"{self.BASE_URL}/{location}/Accounts/{account_id}/Videos"
        params = {
            "accessToken": access_token,
            "name": video_name or "video",
            "privacy": privacy,
            "videoUrl": video_url,
            "language": language
        }
        
        if callback_url:
            params["callbackUrl"] = callback_url
        
        response = requests.post(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def upload_video_file(
        self,
        file_path: str,
        video_name: Optional[str] = None,
        language: str = "en-US",
        privacy: str = "Private",
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Upload a video file from local filesystem with retry logic for connection errors.
        
        Args:
            file_path: Path to the video file
            video_name: Name for the video (optional)
            language: Language of the video (default: "en-US")
            privacy: Privacy setting - "Private" or "Public" (default: "Private")
            max_retries: Maximum number of retry attempts for connection errors (default: 3)
        
        Returns:
            Dictionary containing video ID and upload status
        """
        # Get account ID and location
        account_id = self.account_id or self._extract_account_id()
        location = self.location or self._extract_location()
        
        # Calculate timeout based on file size (minimum 10 minutes, +1 minute per 10MB)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        timeout = max(600, int(600 + (file_size_mb / 10) * 60))  # At least 10 min, +1 min per 10MB
        
        # Upload video file with retry logic
        url = f"{self.BASE_URL}/{location}/Accounts/{account_id}/Videos"
        
        for attempt in range(max_retries):
            try:
                # Get access token (refresh if needed)
                access_token = self._get_access_token()
                
                params = {
                    "accessToken": access_token,
                    "name": video_name or os.path.basename(file_path),
                    "privacy": privacy,
                    "language": language
                }
                
                with open(file_path, 'rb') as video_file:
                    files = {'file': (os.path.basename(file_path), video_file, 'video/mp4')}
                    
                    # Use longer timeout for large files
                    response = requests.post(
                        url, 
                        params=params, 
                        files=files, 
                        timeout=timeout,
                        stream=False  # Don't stream to avoid connection issues
                    )
                    response.raise_for_status()
                    return response.json()
                    
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"  ⚠️  Upload timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Upload timeout after {max_retries} attempts. File may be too large or connection too slow.")
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"  ⚠️  Connection error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Connection error after {max_retries} attempts: {str(e)}")
                    
            except requests.exceptions.HTTPError as e:
                # Provide better error messages
                if hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                    try:
                        error_detail = e.response.json()
                        error_msg = error_detail.get('message', str(error_detail))
                        error_type = error_detail.get('ErrorType', '')
                    except:
                        error_msg = e.response.text[:500] if e.response.text else "Unknown error"
                        error_type = ''
                    
                    # Handle 499 (Client Connection Failure) with retry
                    if status_code == 499 or (status_code >= 500 and attempt < max_retries - 1):
                        wait_time = (2 ** attempt) * 5  # Exponential backoff
                        print(f"  ⚠️  Server error {status_code} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    elif status_code == 400:
                        raise Exception(f"Bad Request (400): {error_msg}")
                    elif status_code == 409:
                        # Video with same name already exists - try to find it or use a different name
                        raise Exception(f"Conflict (409): A video with this name already exists. {error_msg}")
                    elif status_code == 401:
                        raise Exception(f"Unauthorized (401): Authentication failed. Please check your access token.")
                    else:
                        raise Exception(f"HTTP Error ({status_code}): {error_msg}")
                raise
                
            except Exception as e:
                # For other exceptions, check if it's a connection-related error
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['connection', 'timeout', '499', 'closed']):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        print(f"  ⚠️  Connection issue (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                raise
        
        # Should not reach here, but just in case
        raise Exception(f"Upload failed after {max_retries} attempts")
    
    def get_video_index(
        self,
        video_id: str,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Get the indexing results for a video.
        
        Args:
            video_id: ID of the video
            include_insights: Whether to include insights (default: True)
        
        Returns:
            Dictionary containing video indexing results
        """
        # Get account ID and location
        account_id = self.account_id or self._extract_account_id()
        location = self.location or self._extract_location()
        
        # Get access token
        access_token = self._get_access_token()
        
        url = f"{self.BASE_URL}/{location}/Accounts/{account_id}/Videos/{video_id}/Index"
        params = {
            "accessToken": access_token
        }
        if include_insights:
            params["includeInsights"] = "true"
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def wait_for_indexing(
        self,
        video_id: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Wait for video indexing to complete.
        
        Args:
            video_id: ID of the video
            timeout: Maximum time to wait in seconds (default: 3600)
            poll_interval: Interval between polls in seconds (default: 10)
        
        Returns:
            Dictionary containing final indexing results
        
        Raises:
            TimeoutError: If indexing doesn't complete within timeout
        """
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Video indexing timed out after {timeout} seconds")
            
            index_data = self.get_video_index(video_id, include_insights=False)
            state = index_data.get('state', '').lower()
            
            if state == 'processed':
                return self.get_video_index(video_id, include_insights=True)
            elif state == 'failed':
                raise Exception(f"Video indexing failed: {index_data.get('processingResult', {}).get('errorMessage', 'Unknown error')}")
            
            time.sleep(poll_interval)
    
    def _extract_account_id(self) -> str:
        """Extract account ID from the access token (JWT token)."""
        import base64
        import json
        
        try:
            # JWT has three parts separated by dots
            token = self.access_token or self.subscription_key
            parts = token.split('.')
            if len(parts) < 2:
                raise ValueError("Invalid API key format")
            
            # Decode the payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            token_data = json.loads(decoded)
            
            account_id = token_data.get('AccountId', '')
            if not account_id:
                raise ValueError("AccountId not found in token")
            return account_id
        except Exception as e:
            raise ValueError(f"Failed to extract account ID from API key: {e}")
    
    def _extract_location(self) -> str:
        """Extract location from the access token (JWT token)."""
        import base64
        import json
        
        try:
            token = self.access_token or self.subscription_key
            parts = token.split('.')
            if len(parts) < 2:
                raise ValueError("Invalid API key format")
            
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += '=' * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            token_data = json.loads(decoded)
            
            return token_data.get('IssuerLocation', 'eastus')
        except Exception as e:
            raise ValueError(f"Failed to extract location from API key: {e}")
    
    def reload_token(self):
        """
        Reload the access token from environment variables.
        Useful when the token is updated in .env.local while the script is running.
        """
        # Reload environment variables
        load_dotenv('.env.local', override=True)
        
        # Get new token from environment
        new_token = os.getenv('AZURE_VIDEO_INDEXER_API_KEY')
        
        if not new_token:
            raise ValueError(
                "AZURE_VIDEO_INDEXER_API_KEY not found in .env.local after reload.\n"
                "Make sure the token is set in .env.local"
            )
        
        # Update the token
        old_token_preview = self.access_token[:20] + "..." if self.access_token else "None"
        new_token_preview = new_token[:20] + "..."
        
        self.access_token = new_token
        
        # Try to extract account info from new token
        try:
            self.account_id = self._extract_account_id()
            self.location = self._extract_location()
            print(f"✓ Token reloaded successfully")
            print(f"  Old token: {old_token_preview}")
            print(f"  New token: {new_token_preview}")
            print(f"  Account: {self.account_id[:8]}..., Location: {self.location}")
        except Exception as e:
            print(f"⚠️  Token reloaded but could not extract account info: {e}")
            # Keep using old account_id and location if extraction fails
        
        # Clear cached token to force refresh
        self._cached_token = None
        self._token_expires_at = None


def main():
    """Example usage of Azure Video Indexer."""
    client = AzureVideoIndexer()
    
    # Example: Upload video from URL
    # video_url = "https://example.com/video.mp4"
    # result = client.upload_video(video_url, video_name="My Video")
    # video_id = result['id']
    # print(f"Video uploaded. ID: {video_id}")
    # 
    # # Wait for indexing to complete
    # print("Waiting for indexing to complete...")
    # index_result = client.wait_for_indexing(video_id)
    # 
    # # Extract insights
    # insights = index_result.get('videos', [{}])[0].get('insights', {})
    # print(f"Transcript: {insights.get('transcript', 'N/A')}")
    
    print("Azure Video Indexer client initialized successfully!")
    print(f"Account ID: {client._extract_account_id()}")
    print(f"Location: {client._extract_location()}")


if __name__ == "__main__":
    main()

