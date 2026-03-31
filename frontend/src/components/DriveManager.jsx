import { useState, useEffect } from 'react'
import './DriveManager.css'

function DriveManager({ userEmail, onClose }) {
  const [folders, setFolders] = useState([])
  const [files, setFiles] = useState([])
  const [selectedFolder, setSelectedFolder] = useState(null)
  const [loading, setLoading] = useState(false)
  const [processing, setProcessing] = useState({})
  const [error, setError] = useState(null)
  const [view, setView] = useState('folders') // 'folders' or 'files'

  // Load folders on mount
  useEffect(() => {
    loadFolders()
  }, [])

  const loadFolders = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/drive/folders?user_id=${encodeURIComponent(userEmail)}`)
      if (!response.ok) throw new Error('Failed to load folders')
      const data = await response.json()
      setFolders(data.folders || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadFiles = async (folderId, folderName) => {
    setLoading(true)
    setError(null)
    setSelectedFolder({ id: folderId, name: folderName })
    try {
      const response = await fetch(`/api/drive/folders/${folderId}/files?user_id=${encodeURIComponent(userEmail)}`)
      if (!response.ok) throw new Error('Failed to load files')
      const data = await response.json()
      setFiles(data.files || [])
      setView('files')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const processFile = async (fileId, fileName) => {
    setProcessing(prev => ({ ...prev, [fileId]: 'processing' }))
    try {
      const response = await fetch(
        `/api/drive/files/${fileId}/process?user_id=${encodeURIComponent(userEmail)}&file_name=${encodeURIComponent(fileName)}`,
        { method: 'POST' }
      )
      if (!response.ok) throw new Error('Failed to process file')
      const data = await response.json()
      setProcessing(prev => ({ ...prev, [fileId]: 'done' }))
    } catch (err) {
      setProcessing(prev => ({ ...prev, [fileId]: 'error' }))
      setError(err.message)
    }
  }

  const processFolder = async () => {
    if (!selectedFolder) return
    setProcessing(prev => ({ ...prev, [selectedFolder.id]: 'processing' }))
    try {
      const response = await fetch(
        `/api/drive/folders/${selectedFolder.id}/process?user_id=${encodeURIComponent(userEmail)}`,
        { method: 'POST' }
      )
      if (!response.ok) throw new Error('Failed to process folder')
      const data = await response.json()
      setProcessing(prev => ({ ...prev, [selectedFolder.id]: 'done' }))
      // Mark all video files as done
      files.forEach(f => {
        if (isVideoFile(f.name)) {
          setProcessing(prev => ({ ...prev, [f.id]: 'done' }))
        }
      })
    } catch (err) {
      setProcessing(prev => ({ ...prev, [selectedFolder.id]: 'error' }))
      setError(err.message)
    }
  }

  const isVideoFile = (fileName) => {
    const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    return videoExtensions.some(ext => fileName.toLowerCase().endsWith(ext))
  }

  const goBack = () => {
    setView('folders')
    setSelectedFolder(null)
    setFiles([])
  }

  const getStatusIcon = (id) => {
    switch (processing[id]) {
      case 'processing':
        return <span className="status-icon spinning">⏳</span>
      case 'done':
        return <span className="status-icon done">✅</span>
      case 'error':
        return <span className="status-icon error">❌</span>
      default:
        return null
    }
  }

  return (
    <div className="drive-manager-overlay">
      <div className="drive-manager">
        <div className="drive-header">
          <h2>
            {view === 'folders' ? (
              <>📁 Google Drive</>
            ) : (
              <>
                <button className="back-btn" onClick={goBack}>←</button>
                {selectedFolder?.name}
              </>
            )}
          </h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        {error && (
          <div className="drive-error">
            {error}
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        <div className="drive-content">
          {loading ? (
            <div className="drive-loading">Loading...</div>
          ) : view === 'folders' ? (
            <div className="folder-list">
              {folders.length === 0 ? (
                <div className="empty-state">No folders found</div>
              ) : (
                folders.map(folder => (
                  <div
                    key={folder.id}
                    className="folder-item"
                    onClick={() => loadFiles(folder.id, folder.name)}
                  >
                    <span className="folder-icon">📁</span>
                    <span className="folder-name">{folder.name}</span>
                    <span className="folder-arrow">→</span>
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="file-list">
              {selectedFolder && (
                <div className="folder-actions">
                  <button
                    className="process-all-btn"
                    onClick={processFolder}
                    disabled={processing[selectedFolder.id] === 'processing'}
                  >
                    {processing[selectedFolder.id] === 'processing' ? (
                      'Processing...'
                    ) : processing[selectedFolder.id] === 'done' ? (
                      '✅ Folder Processed'
                    ) : (
                      '🚀 Process All Videos'
                    )}
                  </button>
                </div>
              )}

              {files.length === 0 ? (
                <div className="empty-state">No files found</div>
              ) : (
                files.map(file => (
                  <div key={file.id} className="file-item">
                    <span className="file-icon">
                      {isVideoFile(file.name) ? '🎬' : '📄'}
                    </span>
                    <span className="file-name">{file.name}</span>
                    {getStatusIcon(file.id)}
                    {isVideoFile(file.name) && !processing[file.id] && (
                      <button
                        className="process-btn"
                        onClick={() => processFile(file.id, file.name)}
                      >
                        Process
                      </button>
                    )}
                    {processing[file.id] === 'processing' && (
                      <span className="processing-text">Processing...</span>
                    )}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DriveManager
