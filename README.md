## Deployment Process

This backend has a **2-step deployment process**:

### 1. Container Setup (ECR)
- Builds a Docker container with the RAG agent code
- Pushes it to AWS ECR (Elastic Container Registry)  
- Uses scripts in `setup-ecr/` folder

### 2. Agent Deployment (Bedrock AgentCore)
- Takes the container and deploys it as a chat agent on AWS Bedrock
- Connects to Cognito for user authentication
- Has separate scripts for staging (`deploy_agent-staging.py`) and production (`deploy_agent-prod.py`)

**In simple terms**: Packages your AI chat agent into a container, uploads it to AWS, then makes it available as a live chat service that users can authenticate with and use.

---

