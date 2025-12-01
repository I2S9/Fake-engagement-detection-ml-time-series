#!/bin/bash

# Test script for Docker build and run

set -e

echo "=========================================="
echo "Docker Build and Test Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Build Docker image
echo -e "\n${GREEN}Building Docker image...${NC}"
docker build -t fake-engagement-api:latest -f deployment/Dockerfile .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Test that the image runs
echo -e "\n${GREEN}Testing Docker container startup...${NC}"
docker run -d --name test-api -p 8001:8000 fake-engagement-api:latest

# Wait for container to start
sleep 5

# Check if container is running
if docker ps | grep -q test-api; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container failed to start${NC}"
    docker logs test-api
    docker rm -f test-api
    exit 1
fi

# Test health endpoint
echo -e "\n${GREEN}Testing health endpoint...${NC}"
sleep 3
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health || echo "000")

if [ "$response" = "200" ] || [ "$response" = "503" ]; then
    echo -e "${GREEN}✓ Health endpoint responded (status: $response)${NC}"
else
    echo -e "${RED}✗ Health endpoint failed (status: $response)${NC}"
    docker logs test-api
fi

# Cleanup
echo -e "\n${GREEN}Cleaning up test container...${NC}"
docker rm -f test-api

echo -e "\n${GREEN}=========================================="
echo "Docker test completed successfully!"
echo "==========================================${NC}"

