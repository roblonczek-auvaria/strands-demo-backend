#!/bin/bash
# Alternative build approach using native Docker (works better on Apple Silicon)

# Set tag (default to staging)
TAG=${1:-staging}
REPO_URI="081302066317.dkr.ecr.eu-central-1.amazonaws.com/streaming-agent"

echo "Building and pushing with tag: $TAG"

# Build for ARM64 using native Docker
docker build \
    --platform linux/arm64 \
    -f Dockerfile.agentcore \
    -t $REPO_URI:$TAG \
    .

# Push to ECR
docker push $REPO_URI:$TAG
