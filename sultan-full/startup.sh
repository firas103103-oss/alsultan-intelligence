#!/bin/bash

# The Sultan AI - Startup Script

set -e

echo "🚀 Starting The Sultan AI..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${BLUE}ℹ️  No .env file found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${BLUE}⚠️  Please update .env with your actual values${NC}"
fi

# Install dependencies
echo -e "${BLUE}📦 Installing backend dependencies...${NC}"
cd backend
npm ci
npm run build
cd ..

echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd frontend
npm ci
npm run build
cd ..

echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${GREEN}🎉 The Sultan AI is ready to deploy${NC}"
