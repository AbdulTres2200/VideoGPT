# VideoGPT Frontend

A ChatGPT-like interface for querying video insights using RAG (Retrieval-Augmented Generation).

## Features

- 🎨 Modern, ChatGPT-inspired UI
- ⚡ Real-time streaming responses
- 📱 Responsive design
- 🎯 Source references display
- 🚀 Fast and lightweight

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Build

To build for production:
```bash
npm run build
```

## Configuration

The frontend is configured to proxy API requests to `http://localhost:8000` (the FastAPI backend).

To change the API URL, edit `vite.config.js`:

```javascript
proxy: {
  '/api': {
    target: 'http://your-api-url:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

