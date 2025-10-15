# Setup ECR Directory

This directory contains the necessary files and scripts to containerize and deploy a RAG (Retrieval-Augmented Generation) agent to AWS ECR (Elastic Container Registry) for use with Amazon Bedrock AgentCore.

## Overview

The setup creates a Docker container with a FastAPI server that hosts a RAG agent with access to documentation documents. The container is built for ARM64 architecture and deployed to AWS ECR.

## Files

### Deployment Scripts (run in order)
- **`1-create-ecr.sh`** - Creates the ECR repository `streaming-agent` with image scanning enabled
- **`2-login-ecr.sh`** - Authenticates Docker with AWS ECR
- **`4-build-and-push.sh`** - Builds the Docker image for ARM64 and pushes to ECR
- **`5-verify.sh`** - Verifies the image was successfully pushed to ECR

### Application Files
- **`Dockerfile.agentcore`** - Multi-stage Docker build for ARM64 deployment to Bedrock AgentCore
- **`server.py`** - FastAPI server with endpoints for Bedrock AgentCore integration
- **`rag_agent.py`** - RAG agent implementation for documentation document queries
- **`ask_knowledgebase.py`** - Knowledge base interaction utilities
- **`requirements.txt`** - Python dependencies including Strands Agents framework

## Usage

1. Ensure AWS CLI is configured with appropriate permissions
2. Run scripts in numerical order:
   ```bash
   ./1-create-ecr.sh
   ./2-login-ecr.sh
   ./4-build-and-push.sh
   ./5-verify.sh
   ```

## Prerequisites

- AWS CLI configured
- Docker installed
- Appropriate IAM permissions for ECR operations
- Apple Silicon Mac (ARM64 build optimized)

## Target Deployment

The containerized agent is designed for deployment to Amazon Bedrock AgentCore Runtime in the `eu-central-1` region.