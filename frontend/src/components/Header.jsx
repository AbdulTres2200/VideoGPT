import './Header.css'

function Header({ onNewConversation, user, onLogout, onOpenDrive }) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <svg width="32" height="32" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="64" height="64" rx="14" fill="url(#headerGradient)"/>
            <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" fill="#0a0a0f" opacity="0.3"/>
            <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" stroke="white" strokeWidth="2" opacity="0.9"/>
            <path d="M26 28V36L32 40L38 36V28L32 24L26 28Z" fill="white"/>
            <defs>
              <linearGradient id="headerGradient" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                <stop stopColor="#00d4ff"/>
                <stop offset="1" stopColor="#a855f7"/>
              </linearGradient>
            </defs>
          </svg>
          <span className="logo-text">VideoGPT</span>
        </div>

        <div className="header-actions">
          {onOpenDrive && (
            <button className="drive-btn" onClick={onOpenDrive} title="Manage Google Drive Videos">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Drive
            </button>
          )}

          {onNewConversation && (
            <button className="new-conversation-btn" onClick={onNewConversation} title="Start new conversation">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              New Chat
            </button>
          )}

          {user && (
            <div className="user-menu">
              <div className="user-avatar">
                {user.name ? user.name.charAt(0).toUpperCase() : user.email.charAt(0).toUpperCase()}
              </div>
              <div className="user-dropdown">
                <div className="user-info">
                  <span className="user-name">{user.name || 'User'}</span>
                  <span className="user-email">{user.email}</span>
                </div>
                <button className="logout-btn" onClick={onLogout}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M16 17L21 12L16 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M21 12H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Logout
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header
