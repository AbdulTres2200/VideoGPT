import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import InputArea from './components/InputArea'
import Header from './components/Header'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (question) => {
    if (!question.trim() || isLoading) return

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
          return_sources: true
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

  return (
    <div className="app">
      <Header />
      <div className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-content">
              <div className="logo-large">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect width="64" height="64" rx="12" fill="url(#gradient)"/>
                  <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" fill="white" opacity="0.9"/>
                  <path d="M26 28V36L32 40L38 36V28L32 24L26 28Z" fill="white"/>
                  <defs>
                    <linearGradient id="gradient" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                      <stop stopColor="#6366f1"/>
                      <stop offset="1" stopColor="#8b5cf6"/>
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <h1>VideoGPT</h1>
              <p>Ask questions about your video content</p>
              <div className="example-questions">
                <button onClick={() => handleSendMessage("What is OnPrintShop?")}>
                  What is OnPrintShop?
                </button>
                <button onClick={() => handleSendMessage("How do I create a product template?")}>
                  How do I create a product template?
                </button>
                <button onClick={() => handleSendMessage("What are the key features?")}>
                  What are the key features?
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

