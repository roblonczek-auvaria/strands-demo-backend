## Deployment Process

This backend has a **2-step deployment process**:

### 1. Container Setup (ECR)
- Builds a Docker container with the RAG agent code
- Pushes it to AWS ECR (Elastic Container Registry)  
- Uses scripts in `agent-container/` folder

### 2. Agent Deployment (Bedrock AgentCore)
- Takes the container and deploys it as a chat agent on AWS Bedrock
- Connects to Cognito for user authentication
- Has separate scripts for staging (`deploy_agent-staging.py`) and production (`deploy_agent-prod.py`)

**In simple terms**: Packages your AI chat agent into a container, uploads it to AWS, then makes it available as a live chat service that users can authenticate with and use.

## Core Components

### RAG Agent (`agent-container/`)
- **Agent**: Strands-powered AI using Amazon Nova Pro with knowledge base access
- **Knowledge Tool**: Queries ATP documentation via S3 Vectors + Titan embeddings  
- **Server**: FastAPI service with Bedrock AgentCore-compatible endpoints. This server is manually setup and could be simplified within the strands agent using the bedrock-agentcore package. However, this was faster to implement as it was an artifact from a former project.

### Testing (`test.sh`)
End-to-end test script that authenticates via Cognito and calls the deployed agent with a sample query. To test this, register as a user in the prod fronend and use your own credentials.

---

