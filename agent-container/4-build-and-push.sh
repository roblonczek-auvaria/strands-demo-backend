#!/bin/bash
# Alternative build approach using native Docker (works better on Apple Silicon)

# Build for ARM64 using native Docker
docker build \
    --platform linux/arm64 \
    -f Dockerfile.agentcore \
    -t 081302066317.dkr.ecr.eu-central-1.amazonaws.com/streaming-agent:latest \
    .

# Push to ECR
docker push 081302066317.dkr.ecr.eu-central-1.amazonaws.com/streaming-agent:latest
