# Multi-stage build for The Sultan AI (Vite + React + Express)

# Stage 1: Build Backend
FROM node:20.18.1-alpine AS backend-builder
WORKDIR /app/backend

COPY backend/package*.json ./
RUN npm ci

COPY backend/ ./
RUN npm run build

# Stage 2: Build Frontend
FROM node:20.18.1-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 3: Production
FROM node:20.18.1-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser

# Copy backend
COPY --from=backend-builder --chown=appuser:nodejs /app/backend/dist ./backend/dist
COPY --from=backend-builder --chown=appuser:nodejs /app/backend/node_modules ./backend/node_modules
COPY --from=backend-builder --chown=appuser:nodejs /app/backend/package.json ./backend/

# Copy frontend build
COPY --from=frontend-builder --chown=appuser:nodejs /app/frontend/dist ./frontend/dist

USER appuser

EXPOSE 8080

WORKDIR /app/backend
CMD ["node", "dist/server.js"]
