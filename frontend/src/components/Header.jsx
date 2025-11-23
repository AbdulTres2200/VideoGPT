import './Header.css'

function Header({ onNewConversation }) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <svg width="32" height="32" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
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
          <span className="logo-text">VideoGPT</span>
        </div>
        {onNewConversation && (
          <button className="new-conversation-btn" onClick={onNewConversation} title="Start new conversation">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            New Chat
          </button>
        )}
      </div>
    </header>
  )
}

export default Header

