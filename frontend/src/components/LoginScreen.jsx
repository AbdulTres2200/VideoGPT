import './LoginScreen.css'

function LoginScreen({ onLogin }) {
  const handleGoogleLogin = () => {
    // Redirect to Google OAuth
    window.location.href = '/api/auth/google/login'
  }

  return (
    <div className="login-screen">
      <div className="login-content">
        <div className="logo-large">
          <svg width="80" height="80" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="64" height="64" rx="14" fill="url(#loginGradient)"/>
            <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" fill="#0a0a0f" opacity="0.3"/>
            <path d="M20 24L32 16L44 24V40L32 48L20 40V24Z" stroke="white" strokeWidth="2" opacity="0.9"/>
            <path d="M26 28V36L32 40L38 36V28L32 24L26 28Z" fill="white"/>
            <defs>
              <linearGradient id="loginGradient" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                <stop stopColor="#00d4ff"/>
                <stop offset="1" stopColor="#a855f7"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <h1>VideoGPT</h1>
        <p>AI-powered video insights from your Google Drive</p>

        <button className="google-login-btn" onClick={handleGoogleLogin}>
          <svg width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
          </svg>
          Sign in with Google
        </button>

        <div className="login-features">
          <div className="feature">
            <span className="feature-icon">📁</span>
            <span>Access your Google Drive videos</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🎬</span>
            <span>Automatic transcription & analysis</span>
          </div>
          <div className="feature">
            <span className="feature-icon">💬</span>
            <span>Chat with your video content</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoginScreen
