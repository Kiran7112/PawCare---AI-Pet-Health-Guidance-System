# PawCare+ frontend — build with Vite, serve static via nginx
# ---------- Stage 1: build ----------
FROM node:20-alpine AS build

WORKDIR /app

# Install deps (use lockfile if present for reproducible builds)
COPY package*.json ./
RUN npm ci

# Build the production bundle
COPY . .
RUN npm run build

# ---------- Stage 2: serve ----------
FROM nginx:1.27-alpine AS runtime

# SPA-aware config that also proxies /api to the backend container
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
