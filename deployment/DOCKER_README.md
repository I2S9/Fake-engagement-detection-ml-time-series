# Docker Deployment Guide

This guide explains how to build and run the Fake Engagement Detection API using Docker.

## Prerequisites

- Docker installed and running
- Docker Compose (optional, for easier deployment)

## Building the Docker Image

### Option 1: Using Docker directly

```bash
# From project root
docker build -t fake-engagement-api:latest -f deployment/Dockerfile .
```

### Option 2: Using Docker Compose

```bash
# From project root
docker-compose -f deployment/docker-compose.yml build
```

## Running the Container

### Option 1: Using Docker directly

```bash
# Basic run
docker run -p 8000:8000 fake-engagement-api:latest

# With environment variables
docker run -p 8000:8000 \
  -e MODEL_PATH="models/baselines/random_forest.pkl" \
  -e MODEL_TYPE="random_forest" \
  -e CONFIG_PATH="config/config.yaml" \
  fake-engagement-api:latest

# With volume mounts (to persist models and data)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  fake-engagement-api:latest
```

### Option 2: Using Docker Compose

```bash
# From project root
docker-compose -f deployment/docker-compose.yml up

# In detached mode
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop
docker-compose -f deployment/docker-compose.yml down
```

## Testing the Container

### Option 1: Using test script (Linux/Mac)

```bash
chmod +x deployment/test_docker.sh
./deployment/test_docker.sh
```

### Option 2: Using test script (Windows PowerShell)

```powershell
.\deployment\test_docker.ps1
```

### Option 3: Manual testing

```bash
# Build and run
docker build -t fake-engagement-api:latest -f deployment/Dockerfile .
docker run -d --name test-api -p 8000:8000 fake-engagement-api:latest

# Wait a few seconds for startup
sleep 5

# Test health endpoint
curl http://localhost:8000/health

# Test API documentation
# Open browser: http://localhost:8000/docs

# Cleanup
docker rm -f test-api
```

## Environment Variables

You can configure the API using environment variables:

- `MODEL_PATH`: Path to model file (default: none, must be set or loaded via API)
- `MODEL_TYPE`: Type of model (auto-detected if not provided)
- `CONFIG_PATH`: Path to config file
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Volume Mounts

For production, mount these directories:

- `models/`: Trained model files
- `config/`: Configuration files
- `data/`: Data files (if needed)

Example:
```bash
docker run -p 8000:8000 \
  -v /path/to/models:/app/models \
  -v /path/to/config:/app/config \
  fake-engagement-api:latest
```

## Health Check

The container includes a health check that verifies the API is responding:

```bash
# Check container health
docker ps  # Look for "healthy" status

# Inspect health check
docker inspect --format='{{.State.Health.Status}}' <container_id>
```

## Troubleshooting

### Container exits immediately

Check logs:
```bash
docker logs <container_id>
```

### API returns 503 (Model not loaded)

Load a model via the API:
```bash
curl -X POST "http://localhost:8000/load_model?model_path=models/baselines/random_forest.pkl&model_type=random_forest"
```

Or set MODEL_PATH environment variable when running:
```bash
docker run -p 8000:8000 \
  -e MODEL_PATH="models/baselines/random_forest.pkl" \
  -v $(pwd)/models:/app/models \
  fake-engagement-api:latest
```

### Port already in use

Use a different port:
```bash
docker run -p 8001:8000 fake-engagement-api:latest
```

## Production Deployment

For production, consider:

1. Using a reverse proxy (nginx, traefik)
2. Setting up proper logging
3. Using secrets management for sensitive data
4. Implementing resource limits
5. Setting up monitoring and alerting

Example with resource limits:
```bash
docker run -p 8000:8000 \
  --memory="2g" \
  --cpus="2" \
  fake-engagement-api:latest
```

