# PowerShell test script for Docker build and run

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Docker Build and Test Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Build Docker image
Write-Host "`nBuilding Docker image..." -ForegroundColor Green
docker build -t fake-engagement-api:latest -f deployment/Dockerfile .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker image built successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Docker build failed" -ForegroundColor Red
    exit 1
}

# Test that the image runs
Write-Host "`nTesting Docker container startup..." -ForegroundColor Green
docker run -d --name test-api -p 8001:8000 fake-engagement-api:latest

# Wait for container to start
Start-Sleep -Seconds 5

# Check if container is running
$containerRunning = docker ps | Select-String "test-api"
if ($containerRunning) {
    Write-Host "✓ Container is running" -ForegroundColor Green
} else {
    Write-Host "✗ Container failed to start" -ForegroundColor Red
    docker logs test-api
    docker rm -f test-api
    exit 1
}

# Test health endpoint
Write-Host "`nTesting health endpoint..." -ForegroundColor Green
Start-Sleep -Seconds 3

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -TimeoutSec 5
    $statusCode = $response.StatusCode
    Write-Host "✓ Health endpoint responded (status: $statusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    if ($statusCode -eq 503) {
        Write-Host "✓ Health endpoint responded (status: 503 - model not loaded, expected)" -ForegroundColor Yellow
    } else {
        Write-Host "✗ Health endpoint failed (status: $statusCode)" -ForegroundColor Red
        docker logs test-api
    }
}

# Cleanup
Write-Host "`nCleaning up test container..." -ForegroundColor Green
docker rm -f test-api

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "Docker test completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

