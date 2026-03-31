import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import InputArea from './components/InputArea'
import Header from './components/Header'
import LoginScreen from './components/LoginScreen'
import DriveManager from './components/DriveManager'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [showDriveManager, setShowDriveManager] = useState(false)

  // User authentication state
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem('video_gpt_user')
    return stored ? JSON.parse(stored) : null
  })

  // Session ID for conversation
  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem('video_gpt_session_id')
    if (stored) return stored
    const newId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    localStorage.setItem('video_gpt_session_id', newId)
    return newId
  })

  const messagesEndRef = useRef(null)

  // Check for OAuth callback on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const userId = params.get('user_id')
    const email = params.get('email')
    const name = params.get('name')

    if (userId && email) {
      const userData = { user_id: userId, email, name: name || '' }
      setUser(userData)
      localStorage.setItem('video_gpt_user', JSON.stringify(userData))
      // Clean up URL
      window.history.replaceState({}, document.title, '/')
    }
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleLogout = () => {
    setUser(null)
    setMessages([])
    localStorage.removeItem('video_gpt_user')
    localStorage.removeItem('video_gpt_session_id')
  }

  const handleSendMessage = async (question) => {
    if (!question.trim() || isLoading || !user) return

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      sources: null
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Add placeholder AI message
    const aiMessageId = Date.now() + 1
    const aiMessage = {
      id: aiMessageId,
      role: 'assistant',
      content: '',
      sources: null,
      isStreaming: true
    }
    setMessages(prev => [...prev, aiMessage])

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          return_sources: true,
          session_id: sessionId,
          user_email: user.email  // Include user email for filtering
        })
      })

      if (!response.ok) {
        throw new Error('Failed to fetch response')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'content') {
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMessageId
                    ? { ...msg, content: msg.content + data.content }
                    : msg
                ))
              } else if (data.type === 'sources') {
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMessageId
                    ? { ...msg, sources: data.sources, isStreaming: false }
                    : msg
                ))
              } else if (data.type === 'done') {
                if (data.session_id && data.session_id !== sessionId) {
                  setSessionId(data.session_id)
                  localStorage.setItem('video_gpt_session_id', data.session_id)
                }
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMessageId
                    ? { ...msg, isStreaming: false }
                    : msg
                ))
              } else if (data.type === 'error') {
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMessageId
                    ? { ...msg, content: data.content, isStreaming: false }
                    : msg
                ))
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessageId
          ? { ...msg, content: 'Sorry, I encountered an error. Please try again.', isStreaming: false }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewConversation = async () => {
    const oldSessionId = sessionId
    try {
      await fetch('/api/conversation/clear', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: oldSessionId })
      })
    } catch (error) {
      console.error('Error clearing conversation:', error)
    }

    setMessages([])
    const newId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    setSessionId(newId)
    localStorage.setItem('video_gpt_session_id', newId)
  }

  // Show login screen if not authenticated
  if (!user) {
    return <LoginScreen />
  }

  return (
    <div className="app">
      <Header
        onNewConversation={handleNewConversation}
        user={user}
        onLogout={handleLogout}
        onOpenDrive={() => setShowDriveManager(true)}
      />

      {showDriveManager && (
        <DriveManager
          userEmail={user.email}
          onClose={() => setShowDriveManager(false)}
        />
      )}

      <div className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-content">
              <div className="logo-large">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect width="64" height="64" rx="14" fill="url(#welcomeGradient)"/>
                  <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" fill="#0a0a0f" opacity="0.3"/>
                  <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" stroke="white" strokeWidth="2" opacity="0.9"/>
                  <path d="M26 28V36L32 40L38 36V28L32 24L26 28Z" fill="white"/>
                  <defs>
                    <linearGradient id="welcomeGradient" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                      <stop stopColor="#00d4ff"/>
                      <stop offset="1" stopColor="#a855f7"/>
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <h1>Welcome, {user.name || user.email.split('@')[0]}!</h1>
              <p>Ask questions about your video content</p>

              <div className="quick-actions">
                <button className="action-btn" onClick={() => setShowDriveManager(true)}>
                  <span className="action-icon">📁</span>
                  <span className="action-text">
                    <strong>Upload Videos</strong>
                    <small>Process videos from Google Drive</small>
                  </span>
                </button>
              </div>

              <div className="example-questions">
                <p className="examples-label">Or try asking:</p>
                <button onClick={() => handleSendMessage("What videos do I have?")}>
                  What videos do I have?
                </button>
                <button onClick={() => handleSendMessage("Summarize my latest video")}>
                  Summarize my latest video
                </button>
                <button onClick={() => handleSendMessage("What topics are covered in my videos?")}>
                  What topics are covered?
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      <InputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  )
}

export default App
