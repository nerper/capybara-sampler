# Docker image (private Docker Hub)

Production image runs **FastAPI** with **uvicorn** on port **8080** (override with `PORT`).

## Required runtime configuration

- **`OPENAI_API_KEY`** – required for cognate validation (set in your hosting provider’s env/secrets).
- Optional: **`PRELOAD_LANGUAGES`** – comma-separated canonical or alias codes (e.g. `spa,eng`, `en-US,jpn`) to preload Stanza at startup (uses more RAM). Values are normalized the same way as request body language fields.

Do **not** bake `.env` or API keys into the image.

## 1. Build locally

```bash
docker build -t <dockerhub_username>/familiarity-model:latest .
```

If your cloud runs **Linux AMD64** (most VPS / Kubernetes nodes) but you build on Apple Silicon, build for that platform before push:

```bash
docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 -t <dockerhub_username>/familiarity-model:latest --load .
```

## 2. Private repository on Docker Hub

1. Sign in at [hub.docker.com](https://hub.docker.com).
2. **Repositories → Create repository**.
3. Name: e.g. `familiarity-model`.
4. Visibility: **Private**.
5. Create.

## 3. Access token (recommended instead of password)

1. **Account settings → Security → New access token**.
2. Description: e.g. `familiarity-model-push`.
3. Access permissions: **Read, Write, Delete** (or the minimum your org allows for push).
4. Copy the token once; store it in a password manager.

## 4. Log in and push

```bash
docker login -u <dockerhub_username>
# Password: paste the access token (not your account password)

docker tag <dockerhub_username>/familiarity-model:latest docker.io/<dockerhub_username>/familiarity-model:latest
docker push docker.io/<dockerhub_username>/familiarity-model:latest
```

Or one-step build + push for AMD64:

```bash
docker buildx build --platform linux/amd64 \
  -t docker.io/<dockerhub_username>/familiarity-model:latest \
  --push .
```

## 5. “Existing image” deploy UI

| Field | Value |
|--------|--------|
| **Image URL** | `docker.io/<dockerhub_username>/familiarity-model:latest` |
| **Credential** | Registry login for Docker Hub: **username** = your Docker Hub username, **password/secret** = the **access token** |

Then set **`OPENAI_API_KEY`** (and optional `PRELOAD_LANGUAGES`, `PORT` if the platform does not map port 8080 automatically) in the service environment.

## 6. Smoke test

```bash
docker run --rm -e OPENAI_API_KEY=your-key -p 8080:8080 docker.io/<dockerhub_username>/familiarity-model:latest
curl http://localhost:8080/health
# {"status":"healthy"}
```
