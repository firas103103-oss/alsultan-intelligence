# 🏗️ Build & Deployment Guide

## Local Development

### Prerequisites
- Node.js 20+
- npm 10+
- Docker (optional, for containerized development)

### Setup

1. **Clone and Install Dependencies**
```bash
git clone <repo>
cd THE-SULTAN
npm run setup
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your actual configuration
```

3. **Run Development Servers**
```bash
npm run dev
```

This will start:
- Backend API: http://localhost:5000
- Frontend: http://localhost:5173

## Docker Build & Run

### Build Docker Image
```bash
npm run docker:build
```

### Run with Docker
```bash
npm run docker:run
```

### Run with Docker Compose
```bash
npm run docker:compose
```

## Railway Deployment

### Configuration Files
- `railway.json` - Railway deployment configuration
- `Dockerfile` - Multi-stage build (Backend + Frontend)
- `.env.example` - Environment variables template

### Environment Variables Required
- `GOOGLE_GENERATIVE_AI_API_KEY` - Google Gemini API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `APP_URL` - Application URL (e.g., https://sultan.mrf103.com)

### Deployment Steps
1. Connect GitHub repo to Railway
2. Configure environment variables in Railway dashboard
3. Set root directory if needed
4. Push to main branch - Railway will auto-deploy

### Health Checks
- Endpoint: `/health`
- Timeout: 300 seconds
- Restart Policy: ON_FAILURE (max 10 retries)

### Port Configuration
- Internal Port: 8080
- Public Domain: sultan.mrf103.com:8080

## Build Output

### Backend
- Source: `backend/src/**/*.ts`
- Build Output: `backend/dist/`
- Entry: `dist/server.js`

### Frontend
- Source: `frontend/src/**/*.tsx`
- Build Output: `frontend/dist/`
- Framework: React + Vite + Tailwind CSS

## Production Build

The Docker image includes:
1. **Backend Build Stage**
   - Installs all dependencies
   - Compiles TypeScript to JavaScript
   
2. **Frontend Build Stage**
   - Installs dependencies
   - Builds optimized Vite bundle
   
3. **Production Runtime**
   - Minimal Alpine Linux image
   - Non-root user for security
   - Health checks enabled

## Troubleshooting

### Build Fails
- Ensure Node.js version is 20+
- Check all environment variables are set
- Verify .env file exists and is readable

### Deployment Issues
- Check Railway logs: `railway logs`
- Verify health check endpoint: `curl https://sultan.mrf103.com/health`
- Check resource limits in Railway dashboard

### Port Issues
- Ensure port 8080 is not in use
- For local testing: `npm run dev` runs on different ports
- Railway uses port 8080 internally
