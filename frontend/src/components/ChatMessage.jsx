import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatMessage.css'

function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`message ${isUser ? 'message-user' : 'message-ai'}`}>
      <div className="message-content">
        {!isUser && (
          <div className="message-avatar">
            <svg width="20" height="20" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
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
        )}
        <div className="message-text">
          <div className="message-body">
            {message.content ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({node, ...props}) => <h1 className="markdown-h1" {...props} />,
                  h2: ({node, ...props}) => <h2 className="markdown-h2" {...props} />,
                  h3: ({node, ...props}) => <h3 className="markdown-h3" {...props} />,
                  p: ({node, ...props}) => <p className="markdown-p" {...props} />,
                  ul: ({node, ...props}) => <ul className="markdown-ul" {...props} />,
                  ol: ({node, ...props}) => <ol className="markdown-ol" {...props} />,
                  li: ({node, ...props}) => <li className="markdown-li" {...props} />,
                  strong: ({node, ...props}) => <strong className="markdown-strong" {...props} />,
                  em: ({node, ...props}) => <em className="markdown-em" {...props} />,
                  code: ({node, inline, ...props}) => 
                    inline ? (
                      <code className="markdown-code-inline" {...props} />
                    ) : (
                      <code className="markdown-code-block" {...props} />
                    ),
                  blockquote: ({node, ...props}) => <blockquote className="markdown-blockquote" {...props} />,
                }}
              >
                {message.content}
              </ReactMarkdown>
            ) : message.isStreaming ? (
              <span className="typing-indicator">
                <span></span><span></span><span></span>
              </span>
            ) : null}
          </div>
          {message.sources && message.sources.length > 0 && (
            <div className="message-sources">
              <div className="sources-header">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="currentColor"/>
                </svg>
                <span>Sources</span>
              </div>
              <div className="sources-list">
                {message.sources.map((source, idx) => (
                  <div key={idx} className="source-item">
                    <div className="source-name">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2ZM16 18H8V16H16V18ZM16 14H8V12H16V14ZM13 9V3.5L18.5 9H13Z" fill="currentColor"/>
                      </svg>
                      {source.video_name}
                    </div>
                    <div className="source-file">{source.source_file}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        {isUser && (
          <div className="message-avatar user-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 12C14.7614 12 17 9.76142 17 7C17 4.23858 14.7614 2 12 2C9.23858 2 7 4.23858 7 7C7 9.76142 9.23858 12 12 12Z" fill="currentColor"/>
              <path d="M12.0002 14.5C7.99016 14.5 4.75016 17.15 4.75016 20.5C4.75016 20.78 4.97016 21 5.25016 21H18.7502C19.0302 21 19.2502 20.78 19.2502 20.5C19.2502 17.15 16.0102 14.5 12.0002 14.5Z" fill="currentColor"/>
            </svg>
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage

